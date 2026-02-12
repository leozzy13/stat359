#!/usr/bin/env python
# coding: utf-8
"""
Train an LSTM sentiment classifier with padded FastText word vectors.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import datasets
import gensim.downloader as api
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
MAX_EPOCHS = 45
MIN_EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 64
MAX_LEN = 32
EMB_DIM = 300
MODEL_NAME = "fasttext-wiki-news-subwords-300"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text.lower())


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = len(y) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def safe_get_vector(ft_model, token: str) -> np.ndarray:
    for candidate in (token, token.lower(), token.capitalize(), token.upper()):
        if candidate in ft_model.key_to_index:
            return ft_model.get_vector(candidate)
    return np.zeros(EMB_DIM, dtype=np.float32)


def sentence_to_padded_sequence(sentence: str, ft_model, max_len: int = MAX_LEN) -> np.ndarray:
    tokens = tokenize(sentence)
    tokens = tokens[:max_len]

    seq = np.zeros((max_len, EMB_DIM), dtype=np.float32)
    for idx, tok in enumerate(tokens):
        seq[idx] = safe_get_vector(ft_model, tok)
    return seq


def build_sequence_embeddings(sentences: pd.Series, ft_model, max_len: int = MAX_LEN) -> np.ndarray:
    sequences = [
        sentence_to_padded_sequence(sentence, ft_model, max_len=max_len)
        for sentence in tqdm(sentences, desc="Building padded FastText sequences")
    ]
    return np.stack(sequences, axis=0)


class SequenceDataset(Dataset):
    def __init__(self, x_seq: np.ndarray, y: np.ndarray):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x_seq[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        # h_n shape for bidirectional LSTM: (num_layers * 2, batch, hidden_dim)
        final_forward = h_n[-2]
        final_backward = h_n[-1]
        final_hidden = torch.cat((final_forward, final_backward), dim=1)
        return self.head(final_hidden)


def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(is_train):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y_batch.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1, np.array(all_labels), np.array(all_preds)


def plot_learning_curves(history: dict[str, list[float]], output_dir: Path, prefix: str) -> None:
    plt.figure(figsize=(12, 15))
    plt.subplot(3, 1, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(history["train_f1"], label="Train F1")
    plt.plot(history["val_f1"], label="Val F1")
    plt.title("F1 Macro Score Curve")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_f1_learning_curves.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_accuracy_learning_curve.png")
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    class_names: list[str],
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    set_seed(SEED)
    device = get_device()
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== Loading Dataset ==========")
    dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    data = pd.DataFrame(dataset["train"])
    print(f"DataFrame shape: {data.shape}")

    print("\n========== Loading FastText Model ==========")
    print(f"Using Gensim model: {MODEL_NAME}")
    ft_model = api.load(MODEL_NAME)
    print("FastText model loaded.")

    print("\n========== Building Padded FastText Sequences ==========")
    x_seq = build_sequence_embeddings(data["sentence"], ft_model, max_len=MAX_LEN)
    y = data["label"].to_numpy()
    print(f"X_seq shape: {x_seq.shape}, y shape: {y.shape}")

    print("\n========== Stratified Split ==========")
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x_seq, y, test_size=0.15, stratify=y, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
    )
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    num_classes = int(np.max(y) + 1)
    class_weights = compute_class_weights(y_train, num_classes).to(device)

    model = LSTMClassifier(
        input_dim=EMB_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    epochs_since_improve = 0
    best_model_path = output_dir / "best_lstm_model.pth"
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    print("\n========== Training ==========")
    print(f"Device: {device} | Min epochs: {MIN_EPOCHS} | Max epochs: {MAX_EPOCHS}")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        scheduler.step(val_f1)
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            epochs_since_improve = 0
            print(f">>> Saved new best LSTM model (Val F1: {best_val_f1:.4f})")
        else:
            epochs_since_improve += 1

        if epoch >= MIN_EPOCHS and epochs_since_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val F1 improvement for {PATIENCE} epochs).")
            break

    plot_learning_curves(history, output_dir, prefix="lstm")

    print("\n========== Test Evaluation ==========")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_f1_macro, y_true, y_pred = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
    )
    test_f1_weighted = f1_score(y_true, y_pred, average="weighted")
    class_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]

    print("\n" + "=" * 50)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
    print("=" * 50 + "\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm_path = output_dir / "lstm_confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, cm_path, class_names)
    print(f"Confusion matrix saved as '{cm_path}'.")

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    print("\nPer-class F1 Scores:")
    for idx, name in enumerate(class_names):
        print(f"{name}: {per_class_f1[idx]:.4f}")

    metrics = {
        "model": "LSTM",
        "seed": SEED,
        "epochs_ran": len(history["train_loss"]),
        "best_val_f1": float(best_val_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "per_class_f1": {
            "negative": float(per_class_f1[0]),
            "neutral": float(per_class_f1[1]),
            "positive": float(per_class_f1[2]),
        },
    }
    with (output_dir / "lstm_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "lstm_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n========== Script Complete ==========")


if __name__ == "__main__":
    main()
