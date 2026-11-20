# Sports Play-by-Play Generator

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/qrqptsmr28/badge.svg)](https://msrit.ask.thili.ai/b/qrqptsmr28)

A command-line tool that generates short fictional sports commentary with a 279,168-parameter GPT trained from scratch on a CPU.

## Install

From the repository root, with Python 3.12 and Git:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# On Linux, install the CPU-only PyTorch wheel first to avoid CUDA packages.
# pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
python -m app.cli hello
pytest -q
```

The generator works offline with no API key. The starter's separate Ollama, Groq, and Gemini helper remains optional. Keep `.env` local.

## Reproduce the six milestones

1. **Corpus.** Run `python -m app.cli prepare` to strip markup and boilerplate, normalize punctuation and whitespace, remove duplicates, and save seeded splits. To regenerate the fictional lines, run `python data/build_commentary.py`.
2. **Tokenizer and embeddings.** `train` learns up to 512 byte-pair tokens from training lines and saves the merge record and byte vocabulary in `artifacts/tokenizer.json`. The GPT adds learned token and position vectors.
3. **Transformer.** The default configuration has two blocks, width 96, four heads, and context 64. Run `pytest -q` to verify output shape, parameter count, and causal masking.
4. **Training.** Run `python -m app.cli train` to regenerate the ignored CPU checkpoint at `artifacts/model.pt`. Use `--steps`, `--seed`, `--width`, `--layers`, `--context`, `--batch-size`, or `--vocab-size` to change the run. The committed [training log](results/training.json) records 700 steps, final train loss **0.2142**, and final validation loss **0.2280** nats per token. Logged training loss is the mean across the batches since the preceding log entry.
5. **Sample.** Run `python -m app.cli commentate "Last over, 12 needed off 6 balls" --seed 42`. Tune `--temperature`, `--top-k`, `--max-tokens`, and `--num-snippets` as needed.
6. **Interactive CLI.** Run `python -m app.cli interactive`, enter fictional prompts, and type `/quit` when done. Each command supports `--help`.

## Example session

```text
$ python -m app.cli prepare
Prepared 515 training lines and 56 validation lines.

$ python -m app.cli commentate "Last over, 12 needed off 6 balls" --seed 42
Last over, 12 needed off 6 balls, the slow ball fools the batter, but they sneak through for a single.

$ python -m app.cli commentate "Penalty in the 89th minute" --seed 42
Penalty in the 89th minute, the ball clips the underside of the bar and bounces over the line!
```

These examples were generated after training, with the default CPU settings. [Recorded session](results/example-session.txt)

## Data and model

The dataset combines three short [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) Wikinews extracts with 568 original fictional lines for cricket, football, basketball, and tennis. The fictional data is released under CC0. The news lines are kept in the training split. See [source links and attribution](data/SOURCES.md).

Byte-pair encoding starts with UTF-8 bytes and repeatedly merges the most frequent adjacent pair inside a chunk. The learned vocabulary can encode unseen Unicode without an unknown token. In each transformer block, multi-head attention scores queries against keys and uses a lower-triangular mask so each position sees only its own token and earlier ones. An MLP, layer normalization, and residual connections complete the block.

Training uses AdamW, clipped gradients, and cosine learning-rate decay. Generation feeds sampled tokens back into the model, ends at sentence punctuation or the token budget, and trims a trailing partial byte-sized word when possible. Lower temperatures make the distribution more peaked. A smaller top-k excludes more candidates, with top-k 1 selecting the most likely token every time.

## Limitations

The corpus is small and predominantly templated. The validation split holds out lines, but it shares scenario and play templates with training, so the loss overstates how well the model generalizes to novel sports situations. Prompts can yield repetitive, rough, or incoherent text. The model does not track a score or validate sports rules. It is designed for fictional commentary.
