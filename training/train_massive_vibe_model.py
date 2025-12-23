#!/usr/bin/env python3
"""
MASSIVE VIBE MODEL TRAINER
==========================
Trains an attention-based multi-task model on 20k+ examples.
Target: >99.99% accuracy on comprehensive test suite.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm

# Configuration
EMBEDDING_DIM = 384
HIDDEN_DIM = 1024
NUM_HEADS = 8
NUM_LAYERS = 3
VIBE_DIM = 8
DROPOUT = 0.15
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 15  # Early stopping

DIMENSIONS = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']


class AttentionBlock(nn.Module):
    """Self-attention block for feature refinement."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, dim) or (batch, dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x.squeeze(1)


class VibeModelV4(nn.Module):
    """
    Vibe Engine v4 - Attention-based multi-task architecture.

    Architecture:
    1. Embedding projection (384 → 1024)
    2. Stack of attention blocks for feature refinement
    3. Separate prediction heads for each dimension
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                 vibe_dim=VIBE_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # Attention blocks
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Dimension-specific heads with cross-attention
        self.dimension_embeddings = nn.Parameter(torch.randn(vibe_dim, hidden_dim) * 0.02)

        # Prediction heads (one per dimension)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            ) for _ in range(vibe_dim)
        ])

        # Shared final layer for dimension interactions
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + vibe_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        # Project input
        x = self.input_proj(x)

        # Pass through attention layers
        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        # Initial predictions from each head
        initial_preds = torch.cat([head(x) for head in self.heads], dim=1)

        # Fuse with initial predictions for refinement
        fused = self.fusion(torch.cat([x, initial_preds], dim=1))

        # Final predictions with refined features
        final_preds = torch.cat([head(fused) for head in self.heads], dim=1)

        return final_preds


class VibeTrainer:
    """Training manager with early stopping and comprehensive evaluation."""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                all_preds.append(outputs.cpu())
                all_targets.append(batch_y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Calculate per-dimension accuracy (within 0.15 tolerance)
        correct = (torch.abs(all_preds - all_targets) < 0.15).float()
        dim_accuracy = correct.mean(dim=0)
        overall_accuracy = correct.mean()

        return total_loss / len(dataloader), overall_accuracy.item(), dim_accuracy.tolist()


def load_and_prepare_data(csv_path, embedder, max_examples=None):
    """Load data and compute embeddings."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if max_examples:
        df = df.sample(n=min(max_examples, len(df)), random_state=42)

    print(f"Loaded {len(df)} examples")

    # Get texts and targets
    texts = df['text'].tolist()
    targets = df[DIMENSIONS].values.astype(np.float32)

    # Compute embeddings
    print("Computing embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=128)
    embeddings = embeddings.astype(np.float32)

    return embeddings, targets


def create_test_suite():
    """Create comprehensive test suite for evaluation."""
    return [
        # TEMPERATURE (warmth)
        ("freezing cold winter blizzard", [0.05, None, None, None, None, None, None, 0.05]),
        ("blazing hot summer heat", [0.95, None, None, None, None, None, None, 0.95]),
        ("lukewarm indifference", [0.50, None, None, 0.40, None, None, None, 0.50]),
        ("arctic tundra", [0.08, None, None, None, None, None, None, 0.08]),
        ("tropical paradise", [0.90, 0.85, None, 0.90, None, None, None, 0.90]),

        # BRIGHTNESS
        ("pitch black darkness", [None, 0.05, None, None, None, None, None, None]),
        ("blinding bright light", [None, 0.95, None, None, None, None, None, None]),
        ("dim candlelit room", [0.70, 0.20, None, 0.70, 0.25, None, None, 0.75]),
        ("neon fluorescent glare", [None, 0.90, None, None, 0.70, None, 0.80, None]),
        ("soft twilight glow", [0.55, 0.40, None, 0.70, 0.25, 0.45, None, 0.55]),

        # TEXTURE
        ("silky smooth fabric", [None, None, 0.08, 0.75, None, None, None, None]),
        ("rough jagged rocks", [None, None, 0.92, 0.35, None, None, 0.85, None]),
        ("velvety soft touch", [0.70, None, 0.10, 0.80, None, None, None, 0.65]),
        ("coarse sandpaper", [None, None, 0.90, 0.30, None, None, None, None]),
        ("polished marble surface", [None, 0.70, 0.08, 0.65, None, None, 0.75, None]),

        # VALENCE (emotion)
        ("pure unbridled joy", [0.85, 0.85, None, 0.98, 0.85, None, None, 0.80]),
        ("crushing devastating grief", [0.25, 0.15, None, 0.02, 0.40, 0.95, None, 0.25]),
        ("peaceful contentment", [0.65, 0.60, 0.30, 0.85, 0.15, 0.35, None, 0.60]),
        ("anxious dread", [0.35, 0.40, 0.60, 0.15, 0.85, 0.80, None, 0.35]),
        ("bittersweet nostalgia", [0.55, 0.45, None, 0.45, 0.25, 0.70, None, 0.55]),

        # AROUSAL
        ("heart-pounding excitement", [None, None, None, 0.85, 0.98, 0.90, None, None]),
        ("deep meditative calm", [0.55, None, None, 0.75, 0.05, 0.20, None, 0.50]),
        ("sleepy drowsy afternoon", [0.60, 0.50, None, 0.60, 0.10, 0.20, None, 0.55]),
        ("manic frenzied energy", [None, 0.75, None, 0.60, 0.98, 0.95, None, None]),
        ("serene tranquility", [0.55, 0.60, 0.25, 0.80, 0.08, 0.25, None, 0.50]),

        # INTENSITY
        ("overwhelming powerful force", [None, None, 0.70, 0.50, 0.85, 0.98, None, None]),
        ("gentle subtle whisper", [0.60, None, 0.20, 0.65, 0.15, 0.15, None, None]),
        ("mild pleasant breeze", [0.55, 0.65, 0.30, 0.70, 0.30, 0.25, None, None]),
        ("catastrophic devastation", [None, None, 0.80, 0.05, 0.90, 0.99, None, None]),
        ("delicate fragile beauty", [0.60, 0.60, 0.15, 0.75, 0.25, 0.30, 0.30, None]),

        # GEOMETRY
        ("sharp angular edges", [None, None, 0.75, None, None, None, 0.95, None]),
        ("soft flowing curves", [0.60, None, 0.20, 0.70, None, None, 0.10, None]),
        ("rigid geometric patterns", [0.40, None, 0.55, 0.50, None, None, 0.92, None]),
        ("organic natural forms", [0.60, None, 0.45, 0.70, None, None, 0.12, None]),
        ("brutalist concrete blocks", [0.30, 0.45, 0.75, 0.30, None, 0.65, 0.90, 0.30]),

        # COLOR TEMPERATURE
        ("warm golden sunset", [0.90, 0.75, None, 0.85, None, 0.60, None, 0.95]),
        ("cool blue moonlight", [0.25, 0.50, None, 0.55, None, None, None, 0.10]),
        ("fiery orange flames", [0.95, 0.80, None, 0.60, 0.85, 0.90, None, 0.98]),
        ("icy silver frost", [0.10, 0.70, 0.50, 0.45, None, None, None, 0.08]),
        ("neutral grey overcast", [0.45, 0.45, None, 0.45, 0.30, 0.35, None, 0.40]),

        # COMPLEX MULTI-DIMENSIONAL
        ("grandmother's cozy kitchen with warm bread baking", [0.92, 0.60, 0.35, 0.95, 0.35, 0.55, 0.30, 0.88]),
        ("abandoned factory rusting in the rain", [0.30, 0.30, 0.85, 0.15, 0.25, 0.60, 0.75, 0.35]),
        ("neon-lit cyberpunk city at night", [0.45, 0.80, 0.60, 0.55, 0.85, 0.80, 0.85, 0.55]),
        ("ancient forest cathedral with filtered sunlight", [0.55, 0.55, 0.50, 0.80, 0.30, 0.70, 0.60, 0.50]),
        ("children laughing in summer sunshine", [0.85, 0.90, 0.35, 0.95, 0.75, 0.65, 0.25, 0.85]),
        ("hospital waiting room at 3am", [0.35, 0.85, 0.45, 0.25, 0.55, 0.60, 0.80, 0.30]),
        ("velvet jazz club at midnight", [0.65, 0.20, 0.15, 0.70, 0.40, 0.55, 0.40, 0.65]),
        ("thunderstorm approaching over mountains", [0.40, 0.30, 0.65, 0.35, 0.85, 0.90, 0.50, 0.40]),
        ("spa retreat with lavender and soft music", [0.70, 0.55, 0.15, 0.85, 0.10, 0.30, 0.25, 0.65]),
        ("punk rock concert mosh pit", [0.55, 0.65, 0.85, 0.70, 0.98, 0.98, 0.70, 0.55]),
    ]


def evaluate_test_suite(model, embedder, device='cpu'):
    """Evaluate model on comprehensive test suite."""
    model.eval()
    test_cases = create_test_suite()

    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for text, expected in test_cases:
            # Get prediction
            embedding = torch.FloatTensor(embedder.encode([text])).to(device)
            pred = model(embedding).cpu().numpy()[0]

            # Check each dimension
            case_correct = True
            case_results = {'text': text, 'dims': {}}

            for i, (dim_name, expected_val) in enumerate(zip(DIMENSIONS, expected)):
                if expected_val is not None:
                    total += 1
                    error = abs(pred[i] - expected_val)
                    is_correct = error < 0.15  # 0.15 tolerance

                    if is_correct:
                        correct += 1
                    else:
                        case_correct = False

                    case_results['dims'][dim_name] = {
                        'expected': expected_val,
                        'predicted': round(pred[i], 3),
                        'error': round(error, 3),
                        'correct': is_correct
                    }

            results.append(case_results)

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def main():
    print("=" * 70)
    print("MASSIVE VIBE MODEL TRAINER")
    print("Target: >99.99% accuracy")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load embedder
    print("\nLoading sentence transformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Load data
    embeddings, targets = load_and_prepare_data(
        '/Users/erinsaintgull/P/comprehensive_vibe_training_data.csv',
        embedder
    )

    # Split data (90% train, 10% validation)
    n = len(embeddings)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.9 * n)]
    val_idx = indices[int(0.9 * n):]

    train_X = torch.FloatTensor(embeddings[train_idx])
    train_Y = torch.FloatTensor(targets[train_idx])
    val_X = torch.FloatTensor(embeddings[val_idx])
    val_Y = torch.FloatTensor(targets[val_idx])

    print(f"\nTrain: {len(train_X)} | Validation: {len(val_X)}")

    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(train_X, train_Y),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_Y),
        batch_size=BATCH_SIZE
    )

    # Create model
    print("\nCreating model...")
    model = VibeModelV4()
    trainer = VibeTrainer(model, device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print(f"\nTraining for up to {EPOCHS} epochs...")
    print("-" * 70)

    best_accuracy = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)

        # Evaluate
        val_loss, val_accuracy, dim_accuracy = trainer.evaluate(val_loader, criterion)

        # Update scheduler
        scheduler.step(val_loss)

        # Test suite evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_accuracy, _ = evaluate_test_suite(model, embedder, device)
            test_str = f" | Test: {test_accuracy:.2%}"
        else:
            test_str = ""

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%}{test_str}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), '/Users/erinsaintgull/P/best_vibe_model_v4.pth')
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1

        # Early stopping
        if trainer.patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("-" * 70)
    print(f"Best validation accuracy: {best_accuracy:.2%} at epoch {best_epoch}")

    # Load best model and final evaluation
    model.load_state_dict(torch.load('/Users/erinsaintgull/P/best_vibe_model_v4.pth', weights_only=True))
    test_accuracy, test_results = evaluate_test_suite(model, embedder, device)

    print(f"\n{'=' * 70}")
    print(f"FINAL TEST SUITE ACCURACY: {test_accuracy:.4%}")
    print(f"{'=' * 70}")

    # Show failures
    failures = [r for r in test_results if any(not d['correct'] for d in r['dims'].values())]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures[:10]:
            print(f"\n  '{f['text']}'")
            for dim, info in f['dims'].items():
                if not info['correct']:
                    print(f"    {dim}: expected {info['expected']:.2f}, got {info['predicted']:.2f} (error: {info['error']:.2f})")

    # Save metadata
    metadata = {
        'model': 'VibeModelV4',
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'training_examples': len(train_X),
        'best_epoch': best_epoch,
        'best_val_accuracy': float(best_accuracy),
        'test_accuracy': float(test_accuracy),
        'timestamp': datetime.now().isoformat(),
        'dimensions': DIMENSIONS
    }

    with open('/Users/erinsaintgull/P/best_vibe_model_v4_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: best_vibe_model_v4.pth")
    print(f"Metadata saved to: best_vibe_model_v4_metadata.json")

    return test_accuracy


if __name__ == "__main__":
    main()
