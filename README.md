# Vibe Engine

An 8-dimensional aesthetic analysis model that quantifies the "vibe" of text into structured vectors.

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
  "prediction": [0.80, 0.70, 0.35, 0.81, 0.33, 0.45, 0.38, 0.78],
  "dimensions": {
    "warmth": 0.80,
    "brightness": 0.70,
    "texture": 0.35,
    "valence": 0.81,
    "arousal": 0.33,
    "intensity": 0.45,
    "geometry": 0.38,
    "color_temperature": 0.78
  },
  "model": "vibe-engine-v2"
}
```

## Project Structure

```
vibemodel/
├── models/           # Trained PyTorch models
│   ├── best_vibe_model_v5.pth          # Latest (96.6% accuracy)
│   ├── best_vibe_model_v3.pth          # Production (94.7% accuracy)
│   └── best_vibe_model_v2.pth          # Deployed version
├── training/         # Training scripts
│   ├── build_ultimate_vibe_dataset.py  # Dataset generation
│   ├── train_massive_vibe_model.py     # Attention-based trainer
│   ├── acquire_academic_vibe_data.py   # Academic dataset acquisition
│   └── expand_vibe_domains.py          # Domain expansion
├── data/            # Training datasets
│   ├── ultimate_vibe_training_data.csv # 5000 curated examples
│   └── final_enhanced_vibe_data.csv    # 1050 high-quality examples
├── api/             # Flask API server
│   ├── simple_vibe_backend.py
│   └── requirements.txt
└── docs/            # Documentation
```

## Model Architecture

The Vibe Engine uses a multi-head neural network built on sentence-transformer embeddings:

```
Text → all-MiniLM-L6-v2 (384-dim) → Shared Backbone → 8 Dimension Heads → Sigmoid
```

- **Embedder**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Backbone**: 512 → 256 with LayerNorm and Dropout
- **Heads**: Separate prediction head per dimension with Sigmoid activation
- **Output**: 8 values in [0, 1] range

## Training Data

The model is trained on ~5000 curated examples covering:

1. **Physical Sensations** - Temperature, texture, brightness
2. **Emotional States** - Joy, grief, nostalgia, anxiety
3. **Scenes & Environments** - Kitchens, forests, cities, hospitals
4. **Art & Music** - Jazz, punk, ambient, classical
5. **Food & Beverages** - Coffee, wine, spices
6. **Fashion & Interiors** - Fabrics, furniture, lighting
7. **Weather & Nature** - Storms, sunsets, seasons

## Performance

| Version | Test Accuracy | Training Examples | Notes |
|---------|--------------|-------------------|-------|
| v5 | 96.6% | 5,000 | Attention-based architecture |
| v3 | 94.7% | 1,135 | Production-deployed |
| v2 | 87.5% | 1,050 | First multi-dimensional model |

### Known Strengths
- Temperature (warmth/cold) - excellent
- Brightness/darkness - excellent
- Valence (positive/negative emotion) - very good
- Arousal (calm/energetic) - very good

### Areas for Improvement
- Complex emotional states (bittersweet, anxious anticipation)
- Abstract intensity ("the weight of unspoken words")
- Context-dependent brightness (2am bar should be dim)

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

## Training Your Own Model

```python
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

# Load training data
df = pd.read_csv('data/ultimate_vibe_training_data.csv')

# Load embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(df['text'].tolist())

# Define model (see training/train_massive_vibe_model.py)
# Train and evaluate
```

## Citation

If you use this in research, please cite:

```
@software{vibeengine2024,
  title = {Vibe Engine: 8-Dimensional Aesthetic Text Analysis},
  author = {Erin St. Gull},
  year = {2024},
  url = {https://github.com/saintgull/vibemodel}
}
```

## License

MIT License - See LICENSE file for details.
