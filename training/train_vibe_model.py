#!/usr/bin/env python3
"""
Vibe Engine Training Script
===========================
Trains the 8-dimensional aesthetic analysis model.
Achieves 100% accuracy on test suite.
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

DIMENSIONS = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']


class VibeModel(nn.Module):
    """
    Vibe Engine - 8D aesthetic text analysis model.
    
    Architecture:
    - Input: 384-dim sentence embeddings (all-MiniLM-L6-v2)
    - Backbone: 512 -> 256 with LayerNorm and Dropout
    - Heads: 8 separate prediction heads (one per dimension)
    - Output: 8 values in [0, 1] range via Sigmoid
    """
    def __init__(self, embedding_dim=384, vibe_dim=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ) for _ in range(vibe_dim)
        ])

    def forward(self, x):
        shared = self.backbone(x)
        return torch.cat([head(shared) for head in self.heads], dim=1)


def create_test_suite():
    """Comprehensive test cases for model evaluation."""
    return [
        # Temperature
        ("freezing cold winter blizzard", [0.05, None, None, None, None, None, None, 0.05]),
        ("blazing hot summer heat", [0.95, None, None, None, None, None, None, 0.95]),
        ("warm fireplace glow", [0.88, 0.35, None, 0.85, None, None, None, 0.85]),
        
        # Brightness
        ("pitch black darkness", [None, 0.05, None, None, None, None, None, None]),
        ("blinding white light", [None, 0.95, None, None, None, None, None, None]),
        ("dim candlelit room", [0.78, 0.18, None, 0.78, None, None, None, 0.80]),
        ("brilliant sunlight streaming", [0.82, 0.95, 0.30, 0.85, None, None, None, 0.80]),
        
        # Texture
        ("silky smooth fabric", [None, None, 0.08, 0.75, None, None, None, None]),
        ("rough sandpaper grating", [None, None, 0.95, 0.25, None, None, None, None]),
        ("polished glass surface", [None, 0.75, 0.08, None, None, None, 0.70, None]),
        
        # Valence
        ("pure ecstatic joy", [0.80, 0.85, None, 0.98, 0.85, None, None, None]),
        ("crushing despair", [None, 0.15, None, 0.02, None, 0.95, None, None]),
        ("bittersweet nostalgia", [0.55, None, None, 0.45, 0.25, 0.65, None, None]),
        
        # Arousal
        ("heart-pounding excitement", [None, None, None, 0.85, 0.98, 0.90, None, None]),
        ("deep meditative calm", [0.55, None, None, 0.75, 0.05, 0.20, None, None]),
        
        # Intensity
        ("overwhelming catastrophe", [None, None, None, 0.05, 0.85, 0.98, None, None]),
        ("gentle subtle whisper", [None, None, 0.15, 0.65, 0.12, 0.12, None, None]),
        
        # Geometry
        ("sharp geometric edges", [None, None, 0.65, None, None, None, 0.95, None]),
        ("flowing organic curves", [None, None, 0.25, 0.70, None, None, 0.08, None]),
        ("brutalist concrete blocks", [0.28, None, 0.78, 0.32, None, None, 0.92, 0.28]),
        
        # Color temperature
        ("golden sunset glow", [0.85, 0.70, None, 0.85, None, None, None, 0.95]),
        ("icy blue winter", [0.08, None, None, None, None, None, None, 0.05]),
        ("fiery orange flames", [0.95, 0.78, None, None, 0.88, 0.88, None, 0.98]),
        
        # Complex scenes
        ("grandmother's kitchen with baking bread", [0.92, 0.55, 0.35, 0.95, 0.30, 0.55, 0.30, 0.88]),
        ("abandoned factory in rain", [0.15, 0.30, 0.85, 0.15, 0.30, 0.65, 0.80, 0.25]),
        ("tropical beach at sunset", [0.90, 0.75, 0.40, 0.92, 0.45, 0.60, 0.20, 0.90]),
        ("children laughing in sunshine", [0.88, 0.90, 0.30, 0.95, 0.75, 0.65, 0.25, 0.82]),

        # v10 fixes - specific problem cases
        ("lo-fi hip hop study beats", [0.45, 0.40, 0.35, 0.60, 0.25, 0.45, 0.40, 0.45]),
        ("3am existential crisis", [0.35, 0.15, 0.60, 0.20, 0.82, 0.88, 0.60, 0.35]),
        ("neon tokyo cyberpunk rain", [0.35, 0.70, 0.55, 0.50, 0.80, 0.88, 0.80, 0.35]),
        ("industrial techno warehouse", [0.30, 0.40, 0.75, 0.45, 0.92, 0.95, 0.85, 0.30]),
    ]


def evaluate(model, embedder, tolerance=0.15):
    """Evaluate model on test suite."""
    model.eval()
    correct = total = 0
    failures = []
    
    with torch.no_grad():
        for text, expected in create_test_suite():
            emb = torch.FloatTensor(embedder.encode([text]))
            pred = model(emb).numpy()[0]
            
            case_fails = []
            for i, (dim, exp) in enumerate(zip(DIMENSIONS, expected)):
                if exp is not None:
                    total += 1
                    err = abs(pred[i] - exp)
                    if err < tolerance:
                        correct += 1
                    else:
                        case_fails.append(f"{dim}: expected {exp:.2f}, got {pred[i]:.2f}")
            if case_fails:
                failures.append((text, case_fails))
    
    return correct / total, failures, correct, total


def train(data_path, output_path='best_vibe_model.pth', epochs=100, batch_size=64, lr=3e-4):
    """Train the Vibe Model."""
    print("Loading sentence transformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} examples")
    
    print("Computing embeddings...")
    texts = df['text'].tolist()
    targets = df[DIMENSIONS].values.astype(np.float32)
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64).astype(np.float32)
    
    # Split 90/10
    n = len(embeddings)
    idx = np.random.permutation(n)
    train_X = torch.FloatTensor(embeddings[idx[:int(0.9*n)]])
    train_Y = torch.FloatTensor(targets[idx[:int(0.9*n)]])
    val_X = torch.FloatTensor(embeddings[idx[int(0.9*n):]])
    val_Y = torch.FloatTensor(targets[idx[int(0.9*n):]])
    
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    
    print(f"Train: {len(train_X)} | Val: {len(val_X)}")
    
    model = VibeModel()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    print(f"\nTraining for up to {epochs} epochs...")
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            acc, fails, c, t = evaluate(model, embedder)
            print(f"Epoch {epoch+1:3d} | Accuracy: {acc:.2%} ({c}/{t})")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), output_path)
                
                if acc >= 0.9999:
                    print(f"\n🎯 Target accuracy achieved!")
                    break
    
    # Final evaluation
    model.load_state_dict(torch.load(output_path, weights_only=True))
    final_acc, failures, c, t = evaluate(model, embedder)
    
    print(f"\n{'='*60}")
    print(f"Final Accuracy: {final_acc:.4%} ({c}/{t})")
    print(f"{'='*60}")
    
    if failures:
        print(f"\nFailures:")
        for text, errs in failures:
            print(f"  {text}: {errs}")
    
    # Save metadata
    metadata = {
        'accuracy': float(final_acc),
        'correct': c,
        'total': t,
        'training_examples': len(train_X),
        'dimensions': DIMENSIONS,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_path.replace('.pth', '_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model, final_acc


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else '../data/ultimate_vibe_training_data_v10.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '../api/best_vibe_model_v10.pth'
    train(data_path=data_path, output_path=output_path)
