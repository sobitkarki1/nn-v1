"""
Configuration for Tiny Language Model
"""

# Model architecture
VOCAB_SIZE = 50      # Character vocabulary size
EMBED_DIM = 8        # Embedding dimension
HIDDEN_DIM = 16      # Hidden layer dimension

# Training parameters
SEQ_LENGTH = 20      # Sequence length for training
BATCH_SIZE = 32      # Batch size
LEARNING_RATE = 0.01 # Learning rate
EPOCHS = 100         # Number of training epochs
