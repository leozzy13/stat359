import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    pass


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    pass


# Load processed data


# Precompute negative sampling distribution below


# Device selection: CUDA > MPS > CPU



# Dataset and DataLoader


# Model, Loss, Optimizer


def make_targets(center, context, vocab_size):
    pass

# Training loop


# Save embeddings and mappings
# embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
