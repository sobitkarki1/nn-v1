"""
Training script for Tiny Language Model
"""

import torch
import numpy as np
from model import TinyLM, CharTokenizer, calculate_perplexity, device, GPU_AVAILABLE
from config import *


# Simple dataset - Shakespeare-like text for demonstration
DATASET = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action.
"""


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
    """Create mini-batches"""
    n_samples = X.shape[0]
    indices = torch.randperm(n_samples, device=device)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def train():
    """Main training loop"""
    print("=" * 60)
    print("TINY LANGUAGE MODEL TRAINING")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Dataset length: {len(DATASET)} characters")
    print(f"Sequence length: {SEQ_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)
    
    # Initialize tokenizer
    tokenizer = CharTokenizer(DATASET, vocab_size=VOCAB_SIZE)
    
    # Prepare data
    X_train, y_train = prepare_data(DATASET, tokenizer, SEQ_LENGTH)
    print(f"Training samples: {X_train.shape[0]}")
    
    # Initialize model
    model = TinyLM(
        vocab_size=len(tokenizer.char_to_idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    # Training loop
    best_loss = float('inf')
    history = {'epoch': [], 'loss': [], 'perplexity': [], 'accuracy': []}
    
    print("\nTraining started...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        epoch_losses = []
        correct_predictions = 0
        total_predictions = 0
        
        for batch_X, batch_y in create_batches(X_train, y_train, BATCH_SIZE):
            # Forward pass
            logits, hidden, pooled = model.forward(batch_X)
            loss, probs = model.compute_loss(logits, batch_y)
            
            # Backward pass
            model.backward(batch_X, batch_y, logits, probs, hidden, pooled, LEARNING_RATE)
            
            # Calculate accuracy
            predictions = torch.argmax(probs, dim=1)
            correct_predictions += torch.sum(predictions == batch_y).item()
            total_predictions += batch_y.shape[0]
            
            epoch_losses.append(float(loss.cpu().item()))
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        perplexity = calculate_perplexity(avg_loss)
        accuracy = float(correct_predictions) / total_predictions * 100
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['perplexity'].append(perplexity)
        history['accuracy'].append(accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Perplexity: {perplexity:.2f} | Accuracy: {accuracy:.2f}%")
        
        # Save best model
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
