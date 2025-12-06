"""Generate samples from trained model"""

import json
import numpy as np
import torch
from model import TinyLM, CharTokenizer, device

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    tok_data = json.load(f)

tokenizer = CharTokenizer.__new__(CharTokenizer)
tokenizer.char_to_idx = tok_data['char_to_idx']
tokenizer.idx_to_char = {int(k): v for k, v in tok_data['idx_to_char'].items()}

# Load model
model_data = np.load('tiny_lm_model.npz')
model = TinyLM(
    vocab_size=len(tokenizer.char_to_idx),
    embed_dim=8,
    hidden_dim=16
)

model.embedding = torch.tensor(model_data['embedding'], device=device, dtype=torch.float32)
model.W1 = torch.tensor(model_data['W1'], device=device, dtype=torch.float32)
model.b1 = torch.tensor(model_data['b1'], device=device, dtype=torch.float32)
model.W2 = torch.tensor(model_data['W2'], device=device, dtype=torch.float32)
model.b2 = torch.tensor(model_data['b2'], device=device, dtype=torch.float32)

print('=' * 70)
print('GENERATED SAMPLES (Temperature variations)')
print('=' * 70)

prompts = ['To be', 'The', 'And', 'Love', 'Death']
temperatures = [0.5, 0.8, 1.2]

for prompt in prompts:
    print(f'\nPrompt: "{prompt}"')
    print('-' * 70)
    start_tokens = tokenizer.encode(prompt)
    
    for temp in temperatures:
        generated_tokens = model.generate(start_tokens, max_length=80, temperature=temp)
        generated_text = tokenizer.decode(generated_tokens)
        print(f'  [T={temp:.1f}] {generated_text}')

print('\n' + '=' * 70)
