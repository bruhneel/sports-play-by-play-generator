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


# Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


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

