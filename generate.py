"""
Inference script for Tiny Language Model
"""

try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    xp = np
    GPU_AVAILABLE = False

import numpy as np
import json
from model import TinyLM, CharTokenizer
from config import SEQ_LENGTH


def load_model_and_tokenizer():
    """Load saved model and tokenizer"""
    # Load tokenizer
    with open('tokenizer.json', 'r') as f:
        tokenizer_data = json.load(f)
    
    # Reconstruct tokenizer
    class LoadedTokenizer:
        def __init__(self, char_to_idx, idx_to_char):
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
        
        def encode(self, text):
            return [self.char_to_idx.get(c, 1) for c in text]
        
        def decode(self, tokens):
            return ''.join([self.idx_to_char.get(str(t), '<UNK>') for t in tokens])
    
    tokenizer = LoadedTokenizer(
        tokenizer_data['char_to_idx'],
        tokenizer_data['idx_to_char']
    )
    
    # Load model parameters
    params = np.load('tiny_lm_model.npz')
    
    vocab_size = params['embedding'].shape[0]
    embed_dim = params['embedding'].shape[1]
    hidden_dim = params['W1'].shape[1]
    
    # Initialize model
    model = TinyLM(vocab_size, embed_dim, hidden_dim)
    
    # Load parameters
    if GPU_AVAILABLE:
        model.embedding = cp.array(params['embedding'])
        model.W1 = cp.array(params['W1'])
        model.b1 = cp.array(params['b1'])
        model.W2 = cp.array(params['W2'])
        model.b2 = cp.array(params['b2'])
    else:
        model.embedding = params['embedding']
        model.W1 = params['W1']
        model.b1 = params['b1']
        model.W2 = params['W2']
        model.b2 = params['b2']
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.8):
    """Generate text from a prompt"""
    start_tokens = tokenizer.encode(prompt)
    generated_tokens = model.generate(start_tokens, max_length=max_length, temperature=temperature)
    return tokenizer.decode(generated_tokens)


if __name__ == "__main__":
    print("=" * 60)
    print("TINY LANGUAGE MODEL - TEXT GENERATION")
    print("=" * 60)
    
    try:
        model, tokenizer = load_model_and_tokenizer()
        print(f"✓ Model loaded ({model.total_params} parameters)")
        print(f"✓ GPU mode: {GPU_AVAILABLE}")
        print("=" * 60)
        
        # Interactive generation
        while True:
            print("\nEnter a prompt (or 'quit' to exit):")
            prompt = input("> ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                prompt = "To be"
            
            print(f"\nGenerating text from: '{prompt}'")
            print("-" * 60)
            
            generated = generate_text(model, tokenizer, prompt, max_length=150, temperature=0.8)
            print(generated)
            print("-" * 60)
        
        print("\nGoodbye!")
        
    except FileNotFoundError:
        print("Error: Model files not found. Please run train.py first.")
