"""
Tiny Language Model - GPU Accelerated with PyTorch
A primitive character-level language model with ~1000 parameters
"""

import torch
import numpy as np
from config import *

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ GPU mode enabled (PyTorch CUDA)")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    GPU_AVAILABLE = True
else:
    device = torch.device('cpu')
    print("⚠ Falling back to CPU (PyTorch)")
    GPU_AVAILABLE = False


class TinyLM:
    """Tiny Language Model with ~1000 parameters"""
    
    def __init__(self, vocab_size=50, embed_dim=8, hidden_dim=16):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Initialize parameters (Xavier initialization) - numpy-like syntax with PyTorch
        scale_embed = np.sqrt(2.0 / vocab_size)
        scale_w1 = np.sqrt(2.0 / embed_dim)
        scale_w2 = np.sqrt(2.0 / hidden_dim)
        
        self.embedding = torch.randn(vocab_size, embed_dim, device=device, dtype=torch.float32) * scale_embed
        self.W1 = torch.randn(embed_dim, hidden_dim, device=device, dtype=torch.float32) * scale_w1
        self.b1 = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
        self.W2 = torch.randn(hidden_dim, vocab_size, device=device, dtype=torch.float32) * scale_w2
        self.b2 = torch.zeros(vocab_size, device=device, dtype=torch.float32)
        
        # Calculate total parameters
        self.total_params = (vocab_size * embed_dim + 
                            embed_dim * hidden_dim + hidden_dim +
                            hidden_dim * vocab_size + vocab_size)
        print(f"Total parameters: {self.total_params}")
    
    def relu(self, x):
        return torch.maximum(torch.tensor(0.0, device=device), x)
    
    def softmax(self, x):
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, seq_length) - integer indices
        """
        # Embedding lookup: (batch_size, seq_length, embed_dim)
        embedded = self.embedding[x]
        
        # Average pooling over sequence: (batch_size, embed_dim)
        pooled = torch.mean(embedded, dim=1)
        
        # Hidden layer with ReLU: (batch_size, hidden_dim)
        hidden = self.relu(torch.matmul(pooled, self.W1) + self.b1)
        
        # Output layer: (batch_size, vocab_size)
        logits = torch.matmul(hidden, self.W2) + self.b2
        
        return logits, hidden, pooled
    
    def compute_loss(self, logits, targets):
        """Cross-entropy loss"""
        probs = self.softmax(logits)
        batch_size = targets.shape[0]
        
        # Clip probabilities to avoid log(0)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        
        # Cross-entropy
        loss = -torch.mean(torch.log(probs[torch.arange(batch_size, device=device), targets]))
        return loss, probs
    
    def backward(self, x, targets, logits, probs, hidden, pooled, learning_rate):
        """Backward pass and parameter update"""
        batch_size = x.shape[0]
        
        # Gradient of loss w.r.t logits
        dlogits = probs.clone()
        dlogits[torch.arange(batch_size, device=device), targets] -= 1
        dlogits /= batch_size
        
        # Gradients for W2 and b2
        dW2 = torch.matmul(hidden.T, dlogits)
        db2 = torch.sum(dlogits, dim=0)
        
        # Gradient w.r.t hidden
        dhidden = torch.matmul(dlogits, self.W2.T)
        dhidden[hidden <= 0] = 0  # ReLU gradient
        
        # Gradients for W1 and b1
        dW1 = torch.matmul(pooled.T, dhidden)
        db1 = torch.sum(dhidden, dim=0)
        
        # Gradient w.r.t pooled
        dpooled = torch.matmul(dhidden, self.W1.T)
        
        # Gradient w.r.t embedding (broadcasting back to sequence)
        dembedded = dpooled[:, None, :] / x.shape[1]  # (batch_size, 1, embed_dim)
        
        # Update embedding
        for i in range(batch_size):
            for j in range(x.shape[1]):
                idx = x[i, j]
                self.embedding[idx] -= learning_rate * dembedded[i, 0]
        
        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def generate(self, start_tokens, max_length=50, temperature=1.0):
        """Generate text given start tokens"""
        generated = list(start_tokens)
        
        for _ in range(max_length):
            # Take last SEQ_LENGTH tokens
            context = generated[-SEQ_LENGTH:]
            if len(context) < SEQ_LENGTH:
                context = [0] * (SEQ_LENGTH - len(context)) + context
            
            x = torch.tensor([context], dtype=torch.int64, device=device)
            logits, _, _ = self.forward(x)
            
            # Apply temperature
            logits = logits[0] / temperature
            probs = self.softmax(logits)
            
            # Sample next token
            probs_np = probs.cpu().numpy()
            next_token = np.random.choice(self.vocab_size, p=probs_np)
            generated.append(int(next_token))
        
        return generated


class CharTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self, text, vocab_size=50):
        # Get most common characters
        from collections import Counter
        char_counts = Counter(text)
        
        # Reserve special tokens
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        
        # Add most common characters
        most_common = [char for char, _ in char_counts.most_common(vocab_size - 2)]
        for idx, char in enumerate(most_common, start=2):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
        
        print(f"Vocabulary size: {len(self.char_to_idx)}")
    
    def encode(self, text):
        return [self.char_to_idx.get(c, 1) for c in text]
    
    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(t, '<UNK>') for t in tokens])


def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss"""
    if isinstance(loss, torch.Tensor):
        return float(torch.exp(loss).cpu().item())
    else:
        return float(np.exp(loss))


if __name__ == "__main__":
    print("Tiny Language Model - Test")
    model = TinyLM()
    print(f"Model initialized with {model.total_params} parameters")
