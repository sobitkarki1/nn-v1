> **[ARCHIVED]** This repository is kept for historical reference. It represents an early experiment (~1394 parameters, NumPy/CuPy).
> The current generation neural network project is **[nn-v4](https://github.com/sobitkarki1/nn-v4)** — a 1.5B-parameter GPT-style transformer.

---

# Tiny Language Model (~1000 parameters)

A primitive character-level language model using NumPy with GPU acceleration (CuPy).
Part of a learning progression: **nn-v1 → nn-v2 → nn-v4**.

| Version | Params | Architecture | Framework | Notes |
|---------|--------|--------------|-----------|-------|
| **nn-v1** (this) | ~1.4K | Embedding + MLP | NumPy/CuPy | Character-level; ultra-lightweight |
| nn-v2 | ~6.8K | Bigram + transformer block | PyTorch | Context window; GPU-accelerated |
| nn-v4 | ~1.45B | GPT decoder (24 layers) | PyTorch | Mixed precision, Flash Attention 2 |

## Features
- **~1000 parameters**: Ultra-lightweight neural network
- **GPU acceleration**: Uses CuPy for NVIDIA GPU support (falls back to NumPy on CPU)
- **Character-level**: Learns patterns at character granularity
- **Simple architecture**: Embedding → Average Pooling → Hidden Layer (ReLU) → Output

## Usage
`ash
pip install -r requirements.txt
python train.py    # train + save model
python generate.py # interactive text generation
`
"@

 = @"
> **[ARCHIVED]** This repository is kept for historical reference. It represents an intermediate experiment (~6.8K parameters, PyTorch bigram model).
> The current generation neural network project is **[nn-v4](https://github.com/sobitkarki1/nn-v4)** — a 1.5B-parameter GPT-style transformer.

---

# Simple Text Inference Model

A minimal character-level bigram language model using PyTorch.
Part of a learning progression: **nn-v1 → nn-v2 → nn-v4**.

| Version | Params | Architecture | Framework | Notes |
|---------|--------|--------------|-----------|-------|
| nn-v1 | ~1.4K | Embedding + MLP | NumPy/CuPy | Character-level; ultra-lightweight |
| **nn-v2** (this) | ~6.8K | Bigram + transformer block | PyTorch | 8-char context window |
| nn-v4 | ~1.45B | GPT decoder (24 layers) | PyTorch | Mixed precision, Flash Attention 2 |

## Files
- model.py — BigramModel class
- 	rain.py — Training script (creates igram_model.pth)
- generate.py — Interactive text generation
- data.txt — Shakespeare training text

## Usage
`ash
pip install -r requirements.txt
python train.py    # train for 5000 iterations
python generate.py # interactive generation
`
"@

 = @"
> **[ARCHIVED]** This is a legacy copy of Andrej Karpathy's nanoGPT used for early experiments.
> The current generation neural network project is **[nn-v4](https://github.com/sobitkarki1/nn-v4)** — a 1.5B-parameter GPT-style transformer built from scratch.

---

# nanogpt-legacy

An early experiment following Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) to understand GPT-style transformer training.

Key learnings from this stage fed into the design of [nn-v4](https://github.com/sobitkarki1/nn-v4).

## Learning Progression
| Repo | What was learned |
|------|-----------------|
| nn-v1 | Forward pass, backprop by hand (NumPy) |
| nn-v2 | PyTorch autograd, bigram model |
| nanogpt-legacy | Attention mechanism, transformer blocks |
| **nn-v4** | Full GPT at scale (1.5B params, The Pile, Flash Attention 2) |