#!/usr/bin/env python3
"""
Vibe Engine v2 API Backend
Serves the improved 8-dimensional vibe model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:*',
    'https://*.netlify.app',
    'https://synthesis.baby',
    'https://www.synthesis.baby',
    'https://*.railway.app'
])

DIMENSIONS = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']

class VibeModelV2(nn.Module):
    """Vibe Engine v2 - 8D multi-domain model"""
    def __init__(self, embedding_dim=384, vibe_dim=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
        )
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
        return torch.cat([head(shared) for head in self.heads], dim=1)

# Load models at startup
print("Loading Vibe Engine v2...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = VibeModelV2()

model_path = os.path.join(os.path.dirname(__file__), 'best_vibe_model_v2.pth')
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()
print("Vibe Engine v2 loaded successfully!")

def predict_vibes(text: str) -> dict:
    """Predict 8D vibe vector for text"""
    with torch.no_grad():
        embedding = torch.FloatTensor(embedder.encode([text]))
        prediction = model(embedding).numpy()[0]

    return {
        'vector': prediction.tolist(),
        'dimensions': {dim: float(prediction[i]) for i, dim in enumerate(DIMENSIONS)}
    }

@app.route('/')
def index():
    return jsonify({
        'name': 'Vibe Engine v2 API',
        'version': '2.0',
        'dimensions': DIMENSIONS,
        'endpoints': {
            '/api/analyze': 'POST - Analyze text vibes',
            '/api/batch': 'POST - Analyze multiple texts',
            '/health': 'GET - Health check'
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = predict_vibes(text)

        return jsonify({
            'text': text,
            'prediction': result['vector'],
            'dimensions': result['dimensions'],
            'model': 'vibe-engine-v2'
        })

    except Exception as e:
        print(f"Error analyzing text: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        texts = data.get('texts', [])

        if not texts:
            return jsonify({'error': 'No texts provided'}), 400

        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts per batch'}), 400

        results = []
        for text in texts:
            if text.strip():
                result = predict_vibes(text.strip())
                results.append({
                    'text': text,
                    'prediction': result['vector'],
                    'dimensions': result['dimensions']
                })

        return jsonify({
            'results': results,
            'count': len(results),
            'model': 'vibe-engine-v2'
        })

    except Exception as e:
        print(f"Error in batch analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'vibe-engine-v2',
        'dimensions': len(DIMENSIONS)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
