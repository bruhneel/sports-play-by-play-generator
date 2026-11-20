"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

def test_cleaning_and_split():
    from app.core import clean_lines, split_lines

    assert clean_lines(
        '<nav>Unwanted navigation link</nav><p>“What  a finish!” says the caller.</p>'
        '<p>“WHAT a finish!” says the caller.</p><p>Hi</p>\n'
        '<script>Fake long script content</script>\nCopyright 2024, all rights reserved'
    ) == ['"What a finish!" says the caller.']
    lines = [f"Example sports sentence {i}." for i in range(30)]
    train, val = split_lines(lines)
    assert (train, val) == split_lines(lines)
    assert (len(train), len(val)) == (27, 3)
    assert set(train).isdisjoint(val) and set(train + val) == set(lines)


def test_bpe_roundtrip_determinism_and_embeddings():
    import json

    import torch

    from app.core import BOS, EOS, BytePairTokenizer, InputEmbeddings

    source = ["The ball rockets through cover! 🎾", "The ball goes over cover."] * 5
    bpe = BytePairTokenizer.fit(source, 290)
    assert bpe.merges and bpe.merges == BytePairTokenizer.fit(reversed(source), 290).merges
    text = "  Élan—🏀\nThe ball\tgoes through cover!"
    ids = bpe.encode(text, boundaries=True)
    assert ids[0] == BOS and ids[-1] == EOS
    assert bpe.decode(ids) == text
    restored = BytePairTokenizer.from_record(json.loads(json.dumps(bpe.to_record())))
    assert restored.encode(text, boundaries=True) == ids
    assert InputEmbeddings(bpe.size, len(ids), 16)(torch.tensor([ids])).shape == (1, len(ids), 16)


def test_gpt_shapes_causality_and_parameters():
    import torch

    from app.core import GPT, GPTConfig

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = GPT(GPTConfig(20, context=8, width=16, layers=2)).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = ids.clone()
    changed[:, 4:] = 7
    before, after = model(ids), model(changed)
    assert before.shape == (1, 6, 20)
    torch.testing.assert_close(before[:, :4], after[:, :4], atol=0, rtol=0)
    assert not torch.allclose(before[:, 4:], after[:, 4:])
    assert model.head.weight is model.embedding.tokens.weight
    width = 16
    assert model.parameter_count() == (20 + 8 + 2) * width + 2 * (12 * width**2 + 13 * width)


def test_overfit_and_checkpoint(tmp_path):
    import torch

    from app.core import BytePairTokenizer, GPT, GPTConfig, train_gpt
    from app.storage import load_checkpoint, save_checkpoint

    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = BytePairTokenizer.fit(["Goal for the home side!"] * 8)
    model = GPT(GPTConfig(tokenizer.size, context=16, width=16, layers=1))
    ids = tokenizer.encode("Goal for the home side!", boundaries=True)
    history = train_gpt(model, [ids] * 8, [ids] * 2, 120, 4, 42, log=False)
    assert history[-1]["val_loss"] < history[0]["val_loss"] * 0.6
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, tokenizer)
    loaded, bpe = load_checkpoint(path)
    x = torch.tensor([ids[:16]])
    torch.testing.assert_close(loaded(x), model.eval()(x), atol=0, rtol=0)
    assert bpe.encode("Goal!") == tokenizer.encode("Goal!")


def test_generation_limit_seed_and_top_k():
    import torch

    from app.core import BOS, BytePairTokenizer, GPT, GPTConfig, generate_ids, sample_token

    torch.set_num_threads(1)
    torch.manual_seed(2)
    tokenizer = BytePairTokenizer()
    model = GPT(GPTConfig(tokenizer.size, context=8, width=16, layers=1))
    prefix = [BOS, *tokenizer.encode("Goal")]
    a = generate_ids(model, tokenizer, prefix, 3, top_k=4, seed=22)
    assert a == generate_ids(model, tokenizer, prefix, 3, top_k=4, seed=22)
    assert 1 <= len(a) <= 3
    assert all(i in tokenizer.vocab for i in a)
    rng = torch.Generator().manual_seed(3)
    scores = torch.tensor([0.0, 8.0, 1.0, 7.5])
    assert {sample_token(scores, 1, 2, rng) for _ in range(30)} == {1, 3}
    assert sample_token(scores, 1, 1, rng) == 1


def test_generation_stops_on_sentence_end_or_eos():
    import torch

    from app.core import BOS, EOS, BytePairTokenizer, GPT, GPTConfig, generate_ids

    class FixedGPT(GPT):
        def forward(self, ids):
            logits = torch.zeros((1, ids.shape[1], self.config.vocab_size))
            logits[..., self.token] = 10
            return logits

    tokenizer = BytePairTokenizer()
    model = FixedGPT(GPTConfig(tokenizer.size, context=8, width=16, layers=1))
    model.token = ord("!")
    assert generate_ids(model, tokenizer, [BOS], 10, top_k=1) == [ord("!")]
    model.token = EOS
    assert generate_ids(model, tokenizer, [BOS], 10, top_k=1) == []


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

