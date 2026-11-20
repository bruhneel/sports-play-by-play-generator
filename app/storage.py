"""A tiny key-value store in one JSON file (no server, easy to read). Values can be anything JSON can hold."""
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core import BytePairTokenizer, GPT


class Store:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _read(self) -> dict:
        return json.loads(self.path.read_text() or "{}")

    def put(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str, default=None):
        return self._read().get(key, default)

    def all(self) -> dict:
        return dict(sorted(self._read().items()))


def prepare(destination: Path = Path("artifacts/corpus.json"), seed: int = 42) -> dict:
    """Build reproducible train and validation records from local data."""
    from app.core import clean_lines, split_lines

    train, val = split_lines(clean_lines(Path("data/commentary.txt").read_text(encoding="utf-8")), seed)
    news = clean_lines(Path("data/news.txt").read_text(encoding="utf-8"))
    result = {"train": train + news, "val": val, "news_lines": len(news), "seed": seed}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def save_checkpoint(path: Path, model: "GPT", tokenizer: "BytePairTokenizer") -> None:
    """Save the tokenizer and model state together for consistent reloading."""
    from dataclasses import asdict

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.config), "state": model.state_dict(),
         "tokenizer": tokenizer.to_record()}, path,
    )


def load_checkpoint(path: Path = Path("artifacts/model.pt")) -> tuple["GPT", "BytePairTokenizer"]:
    """Reload a CPU checkpoint using tensors and primitive metadata only."""
    import torch

    from app.core import BytePairTokenizer, GPT, GPTConfig

    record = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = BytePairTokenizer.from_record(record["tokenizer"])
    config = GPTConfig(**record["config"])
    if config.vocab_size != tokenizer.size:
        raise ValueError("Model and tokenizer vocabulary sizes differ.")
    model = GPT(config)
    model.load_state_dict(record["state"])
    return model.eval(), tokenizer


def train(
    steps: int = 700, seed: int = 42, width: int = 96, layers: int = 2,
    context: int = 64, batch_size: int = 16, vocab_size: int = 512,
) -> dict:
    """Prepare a seeded corpus, train the tokenizer and GPT, and persist results."""
    from dataclasses import asdict
    from math import exp

    import torch

    from app.core import BytePairTokenizer, GPT, GPTConfig, train_gpt

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    corpus = prepare(seed=seed)
    tokenizer = BytePairTokenizer.fit(corpus["train"], vocab_size)
    model = GPT(GPTConfig(tokenizer.size, context=context, width=width, layers=layers))
    training, validation = (
        [tokenizer.encode(line, boundaries=True) for line in corpus[key]]
        for key in ("train", "val")
    )
    print(f"Training {model.parameter_count():,} parameters on {len(training)} lines.")
    history = train_gpt(model, training, validation, steps, batch_size, seed)
    save_checkpoint(Path("artifacts/model.pt"), model, tokenizer)
    Path("artifacts/tokenizer.json").write_text(
        json.dumps(tokenizer.to_record(), indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "seed": seed, "config": asdict(model.config), "batch_size": batch_size,
        "parameter_count": model.parameter_count(), "train_lines": len(training),
        "val_lines": len(validation), "news_lines": corpus["news_lines"], "loss": history,
        "validation_perplexity": exp(history[-1]["val_loss"]),
    }
    path = Path("results/training.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
