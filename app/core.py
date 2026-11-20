"""Corpus preparation, byte-pair encoding, and a compact GPT."""
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
import math
import random
import re
import unicodedata

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _VisibleText(HTMLParser):
    """Extract text while dropping non-article markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer", "header"}:
            self.hidden.append(tag)
        elif not self.hidden and tag in {"p", "div", "br", "li", "h1", "h2"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def clean_lines(text: str) -> list[str]:
    """Remove markup, boilerplate, short lines and case-insensitive duplicates."""
    parser = _VisibleText()
    parser.feed(text)
    parser.close()
    quotes = str.maketrans("‘’“”", "''\"\"")
    lines = unicodedata.normalize("NFKC", "".join(parser.parts)).translate(quotes).splitlines()
    unique: dict[str, str] = {}
    for raw in lines:
        line = " ".join(raw.split())
        key = line.casefold()
        if len(line) >= 20 and not re.match(
            r"^(copyright\b|all rights reserved\b|subscribe\b|privacy policy\b|navigation\b|terms of use\b)", key
        ):
            unique.setdefault(key, line)
    return list(unique.values())


def split_lines(lines: list[str], seed: int = 42) -> tuple[list[str], list[str]]:
    """Hold out one tenth of unique lines under a stable seed."""
    if len(lines) < 2:
        raise ValueError("Corpus needs two or more usable lines.")
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    count = max(1, len(shuffled) // 10)
    return shuffled[count:], shuffled[:count]


BOS, EOS, PAD = 256, 257, 258
FIRST_MERGE = 259
_CHUNKS = re.compile(r" ?\w+|[^\w\s]|\s+")


def _merge(tokens: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    result = []
    position = 0
    while position < len(tokens):
        if tokens[position:position + 2] == pair:
            result.append(new_id)
            position += 2
        else:
            result.append(tokens[position])
            position += 1
    return tuple(result)


class BytePairTokenizer:
    """Learn ranked byte merges within chunks and preserve all UTF-8 input."""

    def __init__(self, merges: Sequence[tuple[int, int]] = ()) -> None:
        self.merges = tuple(merges)
        self.ranks = {pair: FIRST_MERGE + i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, token in self.ranks.items():
            self.vocab[token] = self.vocab[pair[0]] + self.vocab[pair[1]]

    @property
    def size(self) -> int:
        return FIRST_MERGE + len(self.merges)

    @staticmethod
    def _chunks(text: str) -> list[tuple[int, ...]]:
        return [tuple(chunk.encode("utf-8")) for chunk in _CHUNKS.findall(text)]

    @classmethod
    def fit(cls, texts: Iterable[str], vocab_size: int = 512) -> "BytePairTokenizer":
        """Merge the most frequent adjacent symbol pair with a stable tie break."""
        if not FIRST_MERGE <= vocab_size <= 4096:
            raise ValueError("Vocabulary size must be between 259 and 4096.")
        chunks = Counter(chunk for text in texts for chunk in cls._chunks(text))
        merges = []
        for token in range(FIRST_MERGE, vocab_size):
            counts: Counter[tuple[int, int]] = Counter()
            for chunk, frequency in chunks.items():
                for pair in zip(chunk, chunk[1:]):
                    counts[pair] += frequency
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            if counts[pair] < 2:
                break
            merges.append(pair)
            merged: Counter[tuple[int, ...]] = Counter()
            for chunk, frequency in chunks.items():
                merged[_merge(chunk, pair, token)] += frequency
            chunks = merged
        return cls(merges)

    def encode(self, text: str, *, boundaries: bool = False) -> list[int]:
        ids = []
        for chunk in self._chunks(text):
            for pair, token in self.ranks.items():
                if len(chunk) < 2:
                    break
                chunk = _merge(chunk, pair, token)
            ids.extend(chunk)
        return [BOS, *ids, EOS] if boundaries else ids

    def decode(self, ids: Iterable[int], *, errors: str = "strict") -> str:
        """Decode raw bytes, skipping the three reserved structural tokens."""
        return b"".join(self.vocab[i] for i in ids if i not in (BOS, EOS, PAD)).decode(
            "utf-8", errors=errors
        )

    def to_record(self) -> dict:
        """Serialize ranked merges and each vocabulary item as hex bytes."""
        return {"merges": self.merges, "vocab": {
            str(i): value.hex() for i, value in self.vocab.items()
        }, "special_tokens": {"BOS": BOS, "EOS": EOS, "PAD": PAD}}

    @classmethod
    def from_record(cls, record: dict) -> "BytePairTokenizer":
        tokenizer = cls([tuple(pair) for pair in record["merges"]])
        if tokenizer.to_record()["vocab"] != record["vocab"]:
            raise ValueError("Tokenizer merges and vocabulary disagree.")
        return tokenizer


class InputEmbeddings(nn.Module):
    """Add learned absolute positions to learned token vectors."""

    def __init__(self, vocab_size: int, context: int, width: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context, width)

    def forward(self, ids: Tensor) -> Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.tokens(ids) + self.positions(positions)


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context: int = 64
    width: int = 96
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.context, self.width, self.heads, self.layers) < 1:
            raise ValueError("GPT dimensions must be positive.")
        if self.width % self.heads:
            raise ValueError("Model width must be divisible by head count.")


class CausalAttention(nn.Module):
    """Scaled dot-product attention with an explicit triangular mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)
        self.register_buffer(
            "mask", torch.ones(config.context, config.context, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        dimension = width // self.heads
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, dimension).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / dimension**0.5
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        values = scores.softmax(dim=-1) @ v
        return self.output(values.transpose(1, 2).reshape(batch, length, width))


class TransformerBlock(nn.Module):
    """Pre-normalized attention and MLP with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    """Decoder-only transformer for next-token prediction."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = InputEmbeddings(config.vocab_size, config.context, config.width)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.head.weight = self.embedding.tokens.weight

    @staticmethod
    def _initialize(layer: nn.Module) -> None:
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            nn.init.normal_(layer.weight, std=0.02)
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.context:
            raise ValueError("Expected a batch of nonempty token sequences within the context limit.")
        return self.head(self.norm(self.blocks(self.embedding(ids))))

    def parameter_count(self) -> int:
        """Count the tied embedding/head matrix once."""
        return sum(p.numel() for p in self.parameters())


def _windows(rows: Sequence[Sequence[int]], context: int, rng: random.Random) -> tuple[Tensor, Tensor]:
    """Sample a context window from each selected document and pad on the right."""
    windows = []
    for row in rows:
        start = rng.randrange(max(1, len(row) - context))
        part = list(row[start:start + context + 1])
        windows.append(part + [PAD] * (context + 1 - len(part)))
    tokens = torch.tensor(windows, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]


def _loss(model: GPT, x: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    return F.cross_entropy(
        model(x).flatten(0, 1), targets.flatten(), ignore_index=PAD, reduction=reduction,
    )


@torch.no_grad()
def evaluate(model: GPT, rows: Sequence[Sequence[int]]) -> float:
    """Evaluate fixed windows with each non-padding token weighted equally."""
    model.eval()
    rng = random.Random(761)
    loss = 0.0
    count = 0
    for offset in range(0, len(rows), 32):
        x, y = _windows(rows[offset:offset + 32], model.config.context, rng)
        loss += _loss(model, x, y, reduction="sum").item()
        count += int(y.ne(PAD).sum())
    return loss / count


def train_gpt(
    model: GPT, train: Sequence[Sequence[int]], val: Sequence[Sequence[int]],
    steps: int, batch_size: int, seed: int, *, log: bool = True,
) -> list[dict[str, float | int]]:
    """Train on seeded context windows with clipped AdamW and cosine decay."""
    if min(steps, batch_size, len(train), len(val)) < 1:
        raise ValueError("Training requires a positive step and batch count and both data splits.")
    if any(len(row) < 2 for row in (*train, *val)):
        raise ValueError("Training sequences need at least two tokens.")
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.0003)
    history = []
    recent_losses = []
    for step in range(1, steps + 1):
        model.train()
        x, y = _windows(rng.choices(train, k=batch_size), model.config.context, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        recent_losses.append(loss.item())
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            train_loss = sum(recent_losses) / len(recent_losses)
            validation_loss = evaluate(model, val)
            history.append({"step": step, "train_loss": train_loss, "val_loss": validation_loss})
            recent_losses.clear()
            if log:
                print(f"step {step:4d}  train {train_loss:.4f}  val {validation_loss:.4f}")
    return history


def sample_token(logits: Tensor, temperature: float, top_k: int, rng: torch.Generator) -> int:
    """Draw from the temperature-scaled top-k entries of one logits vector."""
    if not math.isfinite(temperature) or temperature <= 0 or top_k < 1:
        raise ValueError("Temperature and top-k must be positive, with finite temperature.")
    scores, candidates = logits.topk(min(top_k, logits.numel()))
    probabilities = ((scores - scores[0]) / temperature).softmax(dim=0)
    choice = torch.multinomial(probabilities, 1, generator=rng)
    return int(candidates[choice].item())


@torch.inference_mode()
def generate_ids(
    model: GPT, tokenizer: BytePairTokenizer, prefix: list[int], max_new_tokens: int = 32,
    temperature: float = 0.8, top_k: int = 12, seed: int = 42,
) -> list[int]:
    """Decode one token at a time, stopping on EOS, sentence punctuation, or budget."""
    if not prefix or max_new_tokens < 1:
        raise ValueError("Provide a prompt and at least one new token.")
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    ids = list(prefix)
    generated = []
    for _ in range(max_new_tokens):
        window = torch.tensor([ids[-model.config.context:]], dtype=torch.long)
        logits = model(window)[0, -1].clone()
        logits[[BOS, PAD]] = float("-inf")
        token = sample_token(logits, temperature, top_k, rng)
        if token == EOS:
            break
        generated.append(token)
        ids.append(token)
        if tokenizer.vocab[token].endswith((b".", b"!", b"?")):
            break
    return generated


def commentary(
    model: GPT, tokenizer: BytePairTokenizer, prompt: str, max_new_tokens: int = 32,
    temperature: float = 0.8, top_k: int = 12, seed: int = 42,
) -> str:
    """Generate one fictional commentary sentence from a scenario prompt."""
    prompt = " ".join(prompt.split())
    if not prompt:
        raise ValueError("Enter a nonempty commentary prompt.")
    prefix = prompt if prompt.endswith((",", ".", "!", "?", ":")) else prompt + ","
    ids = generate_ids(
        model, tokenizer, [BOS, *tokenizer.encode(prefix)], max_new_tokens,
        temperature, top_k, seed,
    )
    suffix = tokenizer.decode(ids, errors="ignore")
    suffix = "".join(c if c.isprintable() else " " for c in suffix)
    complete = re.match(r".*?[.!?](?=\s|$)", suffix)
    if complete:
        suffix = complete[0]
    elif ids and len(ids) == max_new_tokens and len(tokenizer.vocab[ids[-1]]) == 1:
        trimmed, _, _ = suffix.rstrip().rpartition(" ")
        suffix = trimmed
    return f"{prefix} {suffix.strip()}".strip() if suffix.strip() else prompt


# --- Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
# Hint: User input handling
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
