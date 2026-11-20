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


# Milestone 4: Train the GPT model using the sports text data.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Develop a text generation function to produce short commentary snippets.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

