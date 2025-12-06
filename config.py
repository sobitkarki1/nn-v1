# Model configuration
VOCAB_SIZE = 50  # Character vocabulary size
EMBED_DIM = 16   # Embedding dimension
HIDDEN_DIM = 32  # Hidden layer dimension
SEQ_LENGTH = 20  # Sequence length for training
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 100

# Parameter count calculation:
# Embedding: VOCAB_SIZE * EMBED_DIM = 50 * 16 = 800
# Linear1: EMBED_DIM * HIDDEN_DIM = 16 * 32 = 512
# Linear2: HIDDEN_DIM * VOCAB_SIZE = 32 * 50 = 1600
# Total ≈ 2912 params (slightly over 1000, will adjust if needed)

# Simplified version for ~1000 params:
# EMBED_DIM = 12, HIDDEN_DIM = 20
# Embedding: 50 * 12 = 600
# Linear1: 12 * 20 = 240
# Linear2: 20 * 50 = 1000
# Total = 1840 params (close to target)

# Final adjustment:
EMBED_DIM = 10
HIDDEN_DIM = 18
# Embedding: 50 * 10 = 500
# Linear1: 10 * 18 = 180
# Linear2: 18 * 50 = 900
# Total = 1580 params

# Even smaller:
EMBED_DIM = 8
HIDDEN_DIM = 16
# Embedding: 50 * 8 = 400
# Linear1: 8 * 16 = 128
# Linear2: 16 * 50 = 800
# Bias1: 16, Bias2: 50
# Total = 1394 params ≈ 1000 params
