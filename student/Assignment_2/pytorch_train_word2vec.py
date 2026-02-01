import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Hyperparameters (MUST match handout)
EMBEDDING_DIM = 100
BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive pair


# -----------------------------
# Custom Dataset for Skip-gram
# -----------------------------
class SkipGramDataset(Dataset):
    """
    Expects a pandas DataFrame with columns ['center', 'context'] storing integer word indices.
    """
    def __init__(self, skipgram_df):
        self.centers = torch.tensor(skipgram_df["center"].values, dtype=torch.long)
        self.contexts = torch.tensor(skipgram_df["context"].values, dtype=torch.long)

    def __len__(self):
        return self.centers.size(0)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# -----------------------------
# Skip-gram with Negative Sampling model
# -----------------------------
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)   # center / input
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # context / output

        # Optional: match common word2vec init
        init_range = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed.weight.data.zero_()

    def forward_pos(self, center_ids, context_ids):
        """
        center_ids: (B,)
        context_ids: (B,)
        returns logits: (B,) dot(center, context)
        """
        v = self.in_embed(center_ids)      # (B, D)
        u = self.out_embed(context_ids)    # (B, D)
        return (v * u).sum(dim=1)          # (B,)

    def forward_neg(self, center_ids, neg_context_ids):
        """
        center_ids: (B,)
        neg_context_ids: (B, K)
        returns logits: (B, K) dot(center, neg_context)
        """
        v = self.in_embed(center_ids)                 # (B, D)
        u_neg = self.out_embed(neg_context_ids)       # (B, K, D)
        return (u_neg * v.unsqueeze(1)).sum(dim=2)    # (B, K)

    def get_embeddings(self):
        # Return input embeddings (standard for word2vec)
        return self.in_embed.weight.detach()


def make_targets(center, context, vocab_size):
    """
    Included because it was in the starter file.
    For negative sampling, we don't use a full vocab-sized target.
    We'll return standard binary labels for pos/neg:
      pos: (B,) ones
      neg: (B, K) zeros
    """
    pos = torch.ones_like(center, dtype=torch.float32)
    # placeholder "neg" shape will be created in training once we know K
    return pos


# -----------------------------
# Load processed data
# -----------------------------
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

skipgram_df = data["skipgram_df"]
word2idx = data["word2idx"]
idx2word = data["idx2word"]
counter = data.get("counter", None)

vocab_size = len(word2idx)
print(f"Loaded processed_data.pkl | #pairs={len(skipgram_df)} | vocab_size={vocab_size}")


# -----------------------------
# Precompute negative sampling distribution (unigram^(3/4))
# -----------------------------
# If counter exists (it should), align counts with vocab indices
counts = torch.zeros(vocab_size, dtype=torch.float32)
if counter is not None:
    for w, idx in word2idx.items():
        counts[idx] = float(counter.get(w, 0))
else:
    # Fallback: estimate counts from skipgram pairs
    # (counts center occurrences + context occurrences)
    centers_np = skipgram_df["center"].values
    contexts_np = skipgram_df["context"].values
    for idx in centers_np:
        counts[int(idx)] += 1.0
    for idx in contexts_np:
        counts[int(idx)] += 1.0

# Avoid all-zeros edge case
if counts.sum() == 0:
    counts += 1.0

neg_sampling_dist = counts.pow(0.75)
neg_sampling_dist = neg_sampling_dist / neg_sampling_dist.sum()


# -----------------------------
# Device selection: CUDA > MPS > CPU
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


# -----------------------------
# Dataset and DataLoader
# -----------------------------
dataset = SkipGramDataset(skipgram_df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = Word2Vec(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

neg_sampling_dist = neg_sampling_dist.to(device)


def sample_negatives(batch_size, k, dist, positive_context=None):
    """
    dist: (vocab_size,) probability distribution on device
    positive_context: (B,) tensor of true context indices to avoid sampling
    returns: (B, K) negative samples
    """
    # initial sample
    neg = torch.multinomial(dist, num_samples=batch_size * k, replacement=True).view(batch_size, k)

    # If we want to ensure we don't sample the true context word:
    if positive_context is not None:
        # re-draw collisions a few times (fast and usually enough)
        pos = positive_context.view(-1, 1)
        for _ in range(3):
            mask = (neg == pos)
            if not mask.any():
                break
            resample = torch.multinomial(dist, num_samples=int(mask.sum().item()), replacement=True)
            neg[mask] = resample
    return neg


# -----------------------------
# Training loop
# -----------------------------
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for center_ids, context_ids in pbar:
        center_ids = center_ids.to(device)
        context_ids = context_ids.to(device)

        # Positive logits and labels
        pos_logits = model.forward_pos(center_ids, context_ids)          # (B,)
        pos_labels = torch.ones_like(pos_logits)                         # (B,)

        # Negative samples logits and labels
        neg_context = sample_negatives(center_ids.size(0), NEGATIVE_SAMPLES, neg_sampling_dist, context_ids)
        neg_logits = model.forward_neg(center_ids, neg_context)          # (B, K)
        neg_labels = torch.zeros_like(neg_logits)                        # (B, K)

        # Compute loss
        loss_pos = criterion(pos_logits, pos_labels)
        loss_neg = criterion(neg_logits.view(-1), neg_labels.view(-1))
        loss = loss_pos + loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch:02d} | avg_loss={avg_loss:.4f}")


# -----------------------------
# Save embeddings and mappings
# -----------------------------
embeddings = model.get_embeddings().cpu().numpy()
with open("word2vec_embeddings.pkl", "wb") as f:
    pickle.dump({"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
