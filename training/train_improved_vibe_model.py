#!/usr/bin/env python3
"""
IMPROVED VIBE MODEL TRAINER
===========================
Trains on the new multi-dimensional dataset with proper coverage.

Key improvements:
1. Uses high-quality multi-dimensional training data
2. Validates learning across ALL dimensions
3. Includes proper test cases during training
4. Saves checkpoints based on actual performance, not just loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DIMENSIONS = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']

class VibeDataset(Dataset):
    """Dataset for vibe prediction."""
    def __init__(self, texts, labels, embedder):
        self.texts = texts
        self.labels = labels
        self.embedder = embedder
        print(f"Encoding {len(texts)} examples...")
        self.embeddings = self.embedder.encode(texts, show_progress_bar=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.embeddings[idx]),
            torch.FloatTensor(self.labels[idx])
        )

class ImprovedVibeModel(nn.Module):
    """Improved architecture with better capacity."""
    def __init__(self, embedding_dim=384, vibe_dim=8):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )

        # Per-dimension heads (allows specialization)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(vibe_dim)
        ])

    def forward(self, x):
        shared = self.backbone(x)
        outputs = [head(shared) for head in self.heads]
        return torch.cat(outputs, dim=1)

def load_training_data(csv_path: str):
    """Load and prepare training data."""
    df = pd.read_csv(csv_path)

    print(f"\nLoaded {len(df)} examples from {csv_path}")

    # Analyze distribution
    print("\nData distribution:")
    for dim in DIMENSIONS:
        if dim in df.columns:
            values = df[dim].values
            neutral = np.sum((values > 0.45) & (values < 0.55))
            print(f"  {dim}: {neutral/len(values)*100:.1f}% neutral")

    texts = df['text'].tolist()
    labels = df[DIMENSIONS].values

    return texts, labels

def create_test_cases():
    """Create canonical test cases to validate learning."""
    test_cases = [
        # Temperature tests
        ("very hot blazing heat", {"warmth": (0.8, 1.0)}),
        ("freezing cold ice", {"warmth": (0.0, 0.2)}),
        ("warm cozy comfortable", {"warmth": (0.65, 0.85)}),
        ("cool refreshing breeze", {"warmth": (0.2, 0.4)}),

        # Brightness tests
        ("bright sunny daylight", {"brightness": (0.75, 1.0)}),
        ("pitch black darkness", {"brightness": (0.0, 0.15)}),
        ("dim shadowy twilight", {"brightness": (0.15, 0.35)}),

        # Valence tests
        ("happy joyful wonderful", {"valence": (0.8, 1.0)}),
        ("sad depressing miserable", {"valence": (0.0, 0.2)}),
        ("devastating tragedy grief", {"valence": (0.0, 0.15)}),

        # Arousal tests
        ("exciting thrilling energetic", {"arousal": (0.75, 1.0)}),
        ("calm peaceful serene", {"arousal": (0.0, 0.25)}),
        ("sleeping dormant unconscious", {"arousal": (0.0, 0.15)}),

        # Intensity tests
        ("explosive powerful overwhelming", {"intensity": (0.8, 1.0)}),
        ("subtle gentle mild", {"intensity": (0.0, 0.25)}),

        # Texture tests
        ("smooth polished silky", {"texture": (0.0, 0.2)}),
        ("rough jagged coarse", {"texture": (0.75, 1.0)}),

        # Geometry tests
        ("geometric angular precise", {"geometry": (0.75, 1.0)}),
        ("organic flowing chaotic", {"geometry": (0.0, 0.25)}),

        # Color temperature tests
        ("warm golden orange tones", {"color_temperature": (0.7, 1.0)}),
        ("cool blue icy tones", {"color_temperature": (0.0, 0.3)}),

        # Multi-dimensional tests
        ("hot bright happy summer day", {"warmth": (0.7, 1.0), "brightness": (0.7, 1.0), "valence": (0.7, 1.0)}),
        ("cold dark sad winter night", {"warmth": (0.0, 0.25), "brightness": (0.0, 0.2), "valence": (0.0, 0.3)}),
        ("rough cold stone in darkness", {"texture": (0.7, 1.0), "warmth": (0.0, 0.3), "brightness": (0.0, 0.25)}),
        ("smooth warm polished wood", {"texture": (0.0, 0.25), "warmth": (0.6, 0.85)}),
    ]
    return test_cases

def evaluate_test_cases(model, embedder, test_cases):
    """Evaluate model on canonical test cases."""
    model.eval()
    passed = 0
    failed = 0
    results = []

    with torch.no_grad():
        for text, expected in test_cases:
            embedding = torch.FloatTensor(embedder.encode([text]))
            pred = model(embedding).numpy()[0]

            all_pass = True
            test_result = {"text": text, "dims": {}}

            for dim, (low, high) in expected.items():
                idx = DIMENSIONS.index(dim)
                val = pred[idx]
                dim_pass = low <= val <= high
                test_result["dims"][dim] = {
                    "value": float(val),
                    "expected": (low, high),
                    "pass": dim_pass
                }
                if not dim_pass:
                    all_pass = False

            results.append(test_result)
            if all_pass:
                passed += 1
            else:
                failed += 1

    accuracy = passed / len(test_cases) * 100
    return accuracy, results

def train_model(csv_path: str, epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
    """Train the improved vibe model."""

    print("=" * 60)
    print("IMPROVED VIBE MODEL TRAINING")
    print("=" * 60)

    # Load data
    texts, labels = load_training_data(csv_path)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42
    )

    print(f"\nTraining: {len(X_train)} | Validation: {len(X_val)}")

    # Load embedder
    print("\nLoading sentence transformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Create datasets
    train_dataset = VibeDataset(X_train, y_train, embedder)
    val_dataset = VibeDataset(X_val, y_val, embedder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = ImprovedVibeModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Test cases for validation
    test_cases = create_test_cases()

    # Training loop
    best_test_acc = 0
    best_val_loss = float('inf')

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for embeddings, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for embeddings, targets in val_loader:
                outputs = model(embeddings)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Test case evaluation
        test_acc, test_results = evaluate_test_cases(model, embedder, test_cases)

        # Print progress
        if (epoch + 1) % 5 == 0 or test_acc > best_test_acc:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_acc:.1f}%")

        # Save best model (based on test accuracy, not just loss!)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_val_loss = val_loss

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"improved_vibe_model_{timestamp}.pth"
            torch.save(model.state_dict(), model_path)

            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "epochs_trained": epoch + 1,
                "test_accuracy": test_acc,
                "val_loss": val_loss,
                "train_examples": len(X_train),
                "dimensions": DIMENSIONS,
            }
            with open(f"improved_vibe_model_{timestamp}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  → New best! Saved: {model_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_acc, final_results = evaluate_test_cases(model, embedder, test_cases)

    print(f"\nTest Case Results: {final_acc:.1f}% accuracy")
    print("\nDetailed results:")

    for result in final_results:
        status = "✅" if all(d["pass"] for d in result["dims"].values()) else "❌"
        print(f"\n{status} '{result['text']}'")
        for dim, data in result["dims"].items():
            pass_str = "✓" if data["pass"] else "✗"
            print(f"   {dim}: {data['value']:.3f} (expected {data['expected'][0]:.2f}-{data['expected'][1]:.2f}) {pass_str}")

    print(f"\n{'='*60}")
    print(f"Best Test Accuracy: {best_test_acc:.1f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    return model, best_test_acc

if __name__ == "__main__":
    import sys
    import glob

    # Find the most recent multi-dimensional dataset
    datasets = glob.glob("multidim_vibe_dataset_*.csv")
    if not datasets:
        print("No multi-dimensional dataset found. Run build_multidim_vibe_dataset.py first.")
        sys.exit(1)

    latest_dataset = max(datasets)
    print(f"Using dataset: {latest_dataset}")

    # Train
    model, accuracy = train_model(latest_dataset, epochs=100)

    print(f"\n✅ Training complete! Final accuracy: {accuracy:.1f}%")
