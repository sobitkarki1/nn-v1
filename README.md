# Tiny Language Model (~1000 parameters)

A primitive character-level language model using NumPy with GPU acceleration (CuPy).

## Features
- **~1000 parameters**: Ultra-lightweight neural network
- **GPU acceleration**: Uses CuPy for NVIDIA GPU support (falls back to NumPy on CPU)
- **Character-level**: Learns patterns at character granularity
- **Simple architecture**: Embedding → Average Pooling → Hidden Layer (ReLU) → Output

## Model Architecture
```
Input (character indices)
    ↓
Embedding Layer (50 × 8 = 400 params)
    ↓
Average Pooling
    ↓
Hidden Layer (8 × 16 + 16 bias = 144 params)
    ↓
ReLU Activation
    ↓
Output Layer (16 × 50 + 50 bias = 850 params)
    ↓
Softmax
```
**Total: ~1394 parameters**

## Installation

```powershell
pip install -r requirements.txt
```

For GPU support, you need CUDA 11.x installed. If CuPy installation fails, the model will fall back to CPU mode.

## Usage

### Training
```powershell
python train.py
```

This will:
- Train the model on Shakespeare text for 100 epochs
- Display loss, perplexity, and accuracy metrics
- Save the trained model to `tiny_lm_model.npz`
- Generate sample text

### Text Generation
```powershell
python generate.py
```

Interactive text generation from prompts.

### Evaluation Metrics

The model tracks three key metrics:

1. **Loss (Cross-Entropy)**: Lower is better
   - Measures how well the model predicts the next character
   - Range: 0 to ∞ (typically 0-4 for this task)

2. **Perplexity**: Lower is better
   - Perplexity = exp(loss)
   - Interpretable as "average branching factor"
   - Range: 1 to ∞ (1 is perfect)

3. **Accuracy**: Higher is better
   - Percentage of correctly predicted next characters
   - Range: 0% to 100%

## Configuration

Edit `config.py` to adjust:
- `VOCAB_SIZE`: Number of unique characters
- `EMBED_DIM`: Embedding dimension
- `HIDDEN_DIM`: Hidden layer size
- `SEQ_LENGTH`: Input sequence length
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Learning rate
- `EPOCHS`: Number of training epochs

## Dataset

Currently uses Shakespeare's "To be or not to be" soliloquy. You can easily swap this with any text data by modifying the `DATASET` variable in `train.py`.

## GPU Requirements

- NVIDIA GPU with CUDA support (GTX 1650 Ti supported)
- CUDA 11.x
- CuPy package

If GPU is not available, the code automatically falls back to CPU with NumPy.
