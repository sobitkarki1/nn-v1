"""
Training script for Tiny Language Model
"""

import torch
import numpy as np
from model import TinyLM, CharTokenizer, calculate_perplexity, device, GPU_AVAILABLE
from config import *

# Load dataset from file
with open('t8.shakespear.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
    # Use first 100K characters for manageable training
    DATASET_TEXT = text[:100000]


def prepare_data(text, tokenizer, seq_length):
    """Prepare training data"""
    tokens = tokenizer.encode(text)
    
    # Create sequences
    X, y = [], []
    for i in range(len(tokens) - seq_length):
        X.append(tokens[i:i + seq_length])
        y.append(tokens[i + seq_length])  # Predict next character
    
    return torch.tensor(X, dtype=torch.int64, device=device), torch.tensor(y, dtype=torch.int64, device=device)


def create_batches(X, y, batch_size):
    """Create mini-batches - simple and robust"""
    n_samples = X.shape[0]
    
    # Shuffle indices once at the start
    indices = torch.randperm(n_samples, device=device)
    
    # Create batches sequentially - simpler and more robust
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_indices = indices[i:end_idx]
        yield X[batch_indices], y[batch_indices]


def train():
    """Main training loop"""
    print("=" * 60)
    print("TINY LANGUAGE MODEL TRAINING")
    print("=" * 60)
    print(f"GPU: {GPU_AVAILABLE} | Device: {device}")
    if GPU_AVAILABLE:
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"Dataset: {len(DATASET_TEXT):,} chars | Seq: {SEQ_LENGTH} | Batch: {BATCH_SIZE}")
    print(f"LR: {LEARNING_RATE} | Epochs: {EPOCHS}")
    print("=" * 60)
    
    # Initialize tokenizer and prepare data
    tokenizer = CharTokenizer(DATASET_TEXT, vocab_size=VOCAB_SIZE)
    X_train, y_train = prepare_data(DATASET_TEXT, tokenizer, SEQ_LENGTH)
    print(f"Training samples: {X_train.shape[0]:,}")
    
    # Initialize model
    model = TinyLM(
        vocab_size=len(tokenizer.char_to_idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    # Training history
    best_loss = float('inf')
    history = {'epoch': [], 'loss': [], 'perplexity': [], 'accuracy': []}
    
    print("\nTraining started...")
    print("-" * 60)
    
    if GPU_AVAILABLE:
        torch.cuda.synchronize()
    
    for epoch in range(EPOCHS):
        epoch_losses = []
        correct = 0
        total = 0
        
        # Batch processing
        for batch_X, batch_y in create_batches(X_train, y_train, BATCH_SIZE):
            logits, hidden, pooled = model.forward(batch_X)
            loss, probs = model.compute_loss(logits, batch_y)
            model.backward(batch_X, batch_y, logits, probs, hidden, pooled, LEARNING_RATE)
            
            # Track metrics
            predictions = torch.argmax(probs, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.shape[0]
            epoch_losses.append(loss.item())
        
        if GPU_AVAILABLE:
            torch.cuda.synchronize()
        
        # Compute statistics
        avg_loss = np.mean(epoch_losses)
        perplexity = calculate_perplexity(avg_loss)
        accuracy = 100 * correct / total
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['perplexity'].append(perplexity)
        history['accuracy'].append(accuracy)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Perplexity: {perplexity:.2f} | Acc: {accuracy:.2f}%")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print("-" * 60)
    print("Training completed!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Perplexity: {perplexity:.2f}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    
    # Generate sample text
    print("\nGenerating sample text...")
    print("-" * 60)
    
    start_text = "To be"
    start_tokens = tokenizer.encode(start_text)
    generated_tokens = model.generate(start_tokens, max_length=100, temperature=0.8)
    generated_text = tokenizer.decode(generated_tokens)
    
    print(f"Prompt: '{start_text}'")
    print(f"Generated: '{generated_text}'")
    print("-" * 60)
    
    # Save model parameters
    model_params = {
        'embedding': model.embedding.cpu().numpy(),
        'W1': model.W1.cpu().numpy(),
        'b1': model.b1.cpu().numpy(),
        'W2': model.W2.cpu().numpy(),
        'b2': model.b2.cpu().numpy(),
    }
    
    np.savez('tiny_lm_model.npz', **model_params)
    print("\n✓ Model saved to 'tiny_lm_model.npz'")
    
    # Save tokenizer
    import json
    with open('tokenizer.json', 'w') as f:
        json.dump({
            'char_to_idx': tokenizer.char_to_idx,
            'idx_to_char': {int(k): v for k, v in tokenizer.idx_to_char.items()}
        }, f)
    print("✓ Tokenizer saved to 'tokenizer.json'")
    
    return model, tokenizer, history


if __name__ == "__main__":
    model, tokenizer, history = train()
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("TRAINING METRICS SUMMARY")
    print("=" * 60)
    print(f"Model Parameters: {model.total_params}")
    print(f"Initial Loss: {history['loss'][0]:.4f}")
    print(f"Final Loss: {history['loss'][-1]:.4f}")
    print(f"Loss Improvement: {(history['loss'][0] - history['loss'][-1]):.4f}")
    print(f"Initial Perplexity: {history['perplexity'][0]:.2f}")
    print(f"Final Perplexity: {history['perplexity'][-1]:.2f}")
    print(f"Initial Accuracy: {history['accuracy'][0]:.2f}%")
    print(f"Final Accuracy: {history['accuracy'][-1]:.2f}%")
    print("=" * 60)
