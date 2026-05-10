# BabyMind

BabyMind is a small language-model training project built on the nanoGPT codebase. It is set up for experiments where a model starts from random weights, trains only on the data you provide, and is then sampled through a simple terminal chat loop.

The current repo is intentionally compact: one GPT model implementation, one training script, dataset preparation helpers, a sampler, and a small `chat.py` wrapper for the character-level checkpoint.

## What This Project Does

- Trains a GPT-style causal language model from scratch or from GPT-2 weights.
- Supports tiny character-level experiments with `data/shakespeare_char`.
- Supports GPT-2 BPE/token-level experiments with `data/shakespeare` and `data/openwebtext`.
- Saves checkpoints to an output folder such as `out-shakespeare-char`.
- Lets you generate text with `sample.py` or interactively continue prompts with `chat.py`.

BabyMind is not a production chatbot. It is a learning and research playground for seeing how a small transformer picks up patterns from a controlled dataset.

## Repository Layout

```text
BabyMind/
  chat.py                         Interactive prompt continuation for the char model
  configurator.py                 Simple config-file and --key=value override loader
  model.py                        GPT model, attention blocks, optimizer setup, generation
  sample.py                       General text sampling from a checkpoint or GPT-2 weights
  train.py                        Training loop with checkpointing, eval, AMP, and DDP support
  config/
    train_shakespeare_char.py     Small character-level training config
    finetune_shakespeare.py       GPT-2 XL fine-tune config for Shakespeare
    train_gpt2.py                 GPT-2 124M OpenWebText training config
    eval_gpt2*.py                 GPT-2 evaluation configs
  data/
    shakespeare_char/             Character-level Tiny Shakespeare preparation
    shakespeare/                  GPT-2 BPE Tiny Shakespeare preparation
    openwebtext/                  OpenWebText preparation for larger training
```

Generated files are intentionally ignored by Git, including `*.bin`, `*.pkl`, `*.pt`, root-level `input.txt`, virtual environments, and Python caches.

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- tiktoken
- requests
- transformers, needed when loading GPT-2 weights
- datasets and tqdm, needed for OpenWebText preparation
- wandb, optional logging

Install the common dependencies:

```bash
pip install torch numpy tiktoken requests transformers datasets tqdm wandb
```

For CUDA training, install the PyTorch build that matches your GPU and CUDA version from the official PyTorch instructions.

## Quick Start: Character-Level BabyMind

This is the smallest experiment and the one `chat.py` expects.

1. Prepare the character-level data:

```bash
python data/shakespeare_char/prepare.py
```

This creates:

- `data/shakespeare_char/train.bin`
- `data/shakespeare_char/val.bin`
- `data/shakespeare_char/meta.pkl`

2. Train the small model:

```bash
python train.py config/train_shakespeare_char.py --compile=False --always_save_checkpoint=True
```

On CPU, use:

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --always_save_checkpoint=True
```

The config saves checkpoints in:

```text
out-shakespeare-char/ckpt.pt
```

3. Chat with the trained checkpoint:

```bash
python chat.py
```

`chat.py` loads `out-shakespeare-char/ckpt.pt` and the character vocabulary from `data/shakespeare_char/meta.pkl`. It encodes your prompt, lets the model continue it, and prints the first few generated lines.

## Training On Your Own Curriculum

To teach the model from your own text, replace or create:

```text
data/shakespeare_char/input.txt
```

Then rerun:

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --compile=False --always_save_checkpoint=True
python chat.py
```

The character-level pipeline learns exactly from the characters in `input.txt`. If your curriculum includes Hindi, Hinglish, or other Unicode text, read and write `input.txt` as UTF-8 before training.

## Sampling

For raw generation from a checkpoint:

```bash
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:" --num_samples=3 --max_new_tokens=200 --device=cpu
```

If the checkpoint has dataset metadata, `sample.py` uses that dataset's encoder and decoder. Otherwise it falls back to GPT-2 tokenization through `tiktoken`.

## Larger Training Modes

Tiny Shakespeare with GPT-2 BPE tokens:

```bash
python data/shakespeare/prepare.py
python train.py config/finetune_shakespeare.py
```

OpenWebText pretraining:

```bash
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

OpenWebText preparation is large. It downloads and tokenizes millions of documents and can require tens of gigabytes of disk space.

## Configuration

Training options are plain Python globals in `train.py` and config files under `config/`. `configurator.py` applies overrides in this order:

1. Config file, for example `config/train_shakespeare_char.py`
2. Command-line overrides, for example `--batch_size=32`

Example:

```bash
python train.py config/train_shakespeare_char.py --batch_size=32 --max_iters=1000 --compile=False
```

## Checkpoints And Outputs

Important generated files:

- `train.bin` and `val.bin`: tokenized dataset files
- `meta.pkl`: vocabulary metadata for character-level datasets
- `ckpt.pt`: model checkpoint
- `out-shakespeare-char/`: default output directory for the small BabyMind run

These files can be large or machine-specific, so they are ignored by Git.

## Notes And Limitations

- The chat mode is prompt continuation, not instruction following.
- A tiny dataset usually causes memorization before broad generalization.
- The model only knows patterns present in the training data.
- Better behavior comes from better curriculum design, more examples, and repeated evaluation.
- GPU training is strongly recommended for anything beyond the smallest character model.

## Credits

BabyMind is based on the nanoGPT architecture and training style by Andrej Karpathy, adapted here as a compact local playground for controlled language-learning experiments.

## License

MIT License. See `LICENSE`.
