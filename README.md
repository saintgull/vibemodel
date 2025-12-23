# Vibe Engine

An 8-dimensional aesthetic analysis model that quantifies the "vibe" of text into structured vectors.

**100% test accuracy achieved** on comprehensive test suite (104/104 cases).

## Dimensions

| Dimension | Low (0.0) | High (1.0) |
|-----------|-----------|------------|
| **warmth** | cold, icy | hot, warm |
| **brightness** | dark, dim | bright, luminous |
| **texture** | smooth, silky | rough, jagged |
| **valence** | negative, sad | positive, joyful |
| **arousal** | calm, peaceful | excited, energetic |
| **intensity** | subtle, gentle | overwhelming, powerful |
| **geometry** | organic, curved | angular, geometric |
| **color_temperature** | cool blue | warm orange |

## Live API

**Production URL:** https://vibe-engine-production.up.railway.app

### Endpoints

```bash
# Health check
curl https://vibe-engine-production.up.railway.app/health

# Analyze single text
curl -X POST https://vibe-engine-production.up.railway.app/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "warm sunny beach day"}'

# Batch analysis (up to 100 texts)
curl -X POST https://vibe-engine-production.up.railway.app/api/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["cold winter night", "tropical paradise", "dark gothic cathedral"]}'
```

### Response Format

```json
{
  "text": "warm sunny beach day",
  "prediction": [0.89, 0.82, 0.35, 0.90, 0.42, 0.50, 0.30, 0.88],
  "dimensions": {
    "warmth": 0.89,
    "brightness": 0.82,
    "texture": 0.35,
    "valence": 0.90,
    "arousal": 0.42,
    "intensity": 0.50,
    "geometry": 0.30,
    "color_temperature": 0.88
  },
  "model": "vibe-engine-v7"
}
```

## Project Structure

```
vibemodel/
├── models/
│   ├── best_vibe_model_v7.pth           # 100% accuracy model
│   └── best_vibe_model_v7_metadata.json # Training metadata
├── training/
│   └── train_vibe_model.py              # Training script with test suite
├── data/
│   └── ultimate_vibe_training_data_v3.csv  # 5,476 training examples
├── api/
│   ├── simple_vibe_backend.py           # Flask API server
│   ├── best_vibe_model_v7.pth           # Model copy for deployment
│   └── requirements.txt
└── docs/
    ├── VIBE_ENGINE_MASTERDOC.md
    └── vibe-engine-offering.md
```

## Model Architecture

```
Text → all-MiniLM-L6-v2 (384-dim) → Backbone (512→256) → 8 Heads → Sigmoid
```

```python
class VibeModel(nn.Module):
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
```

Key design choices:
- **LayerNorm** (not BatchNorm) - works with single-sample inference
- **Separate heads** per dimension - allows independent learning
- **GELU activation** - smooth gradients for aesthetic data
- **Sigmoid output** - bounded [0, 1] range

## Training Data

5,476 curated examples covering:

1. **Physical Sensations** - Temperature, texture, brightness
2. **Emotional States** - Joy, grief, nostalgia, anxiety
3. **Scenes & Environments** - Kitchens, forests, cities
4. **Art & Music** - Jazz, punk, ambient, classical
5. **Food & Beverages** - Coffee, wine, spices
6. **Fashion & Interiors** - Fabrics, furniture, lighting
7. **Weather & Nature** - Storms, sunsets, seasons
8. **Architecture** - Brutalist, organic, minimalist

Multi-dimensional examples where each training sample exercises 4+ dimensions simultaneously.

## Performance

| Version | Test Accuracy | Training Examples | Notes |
|---------|--------------|-------------------|-------|
| **v7** | **100%** | 5,476 | Current production model |
| v6 | 97.3% | 4,532 | 4 edge case failures |
| v5 | 96.6% | 5,000 | First attention-based |

### Test Suite

The model passes 104 test cases across:
- Temperature extremes ("freezing cold" → "blazing hot")
- Brightness range ("pitch black" → "blinding white")
- Texture spectrum ("silky smooth" → "rough sandpaper")
- Emotional valence ("crushing despair" → "pure ecstatic joy")
- Arousal levels ("deep meditative calm" → "heart-pounding excitement")
- Intensity ("gentle whisper" → "overwhelming catastrophe")
- Geometric shapes ("flowing organic curves" → "sharp geometric edges")
- Complex scenes ("grandmother's kitchen" → "abandoned factory in rain")

## Local Development

```bash
# Install dependencies
pip install torch sentence-transformers flask flask-cors pandas numpy

# Run API locally
cd api
python simple_vibe_backend.py

# Test
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "peaceful garden at sunrise"}'
```

## Train Your Own

```bash
cd training
python train_vibe_model.py
```

Edit `train_vibe_model.py` to point to your training data:
```python
train(
    data_path='../data/ultimate_vibe_training_data_v3.csv',
    output_path='../models/best_vibe_model_v7.pth'
)
```

## Deploy to Railway

```bash
# In api/ directory
railway login
railway init
railway up
```

The API automatically uses the `PORT` environment variable.

## Citation

```
@software{vibeengine2025,
  title = {Vibe Engine: 8-Dimensional Aesthetic Text Analysis},
  author = {Erin Saint Gull},
  year = {2025},
  url = {https://github.com/saintgull/vibemodel}
}
```

## License

MIT License - See LICENSE file for details.

## Author

**Erin Saint Gull**
- Website: [synthesis.baby](https://synthesis.baby)
- Email: erin@curate.beauty
