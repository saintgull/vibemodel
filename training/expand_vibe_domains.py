#!/usr/bin/env python3
"""
EXPANDED VIBE DOMAIN DATASET
============================
Explores underexplored domains for vibe analysis:

1. FOOD & CULINARY - texture, warmth, intensity
2. WINE & BEVERAGES - complex tasting vocabulary
3. PERFUME & FRAGRANCE - cross-modal synesthesia
4. FASHION & TEXTILE - structure, texture, mood
5. INTERIOR DESIGN - comprehensive vibe space
6. FILM & CINEMA - mood, tone, atmosphere
7. VIDEO GAMES - aesthetic vocabulary
8. ART MOVEMENTS - style descriptions
9. TRAVEL & HOTELS - atmosphere marketing
10. COCKTAILS - tasting notes
11. SPA & WELLNESS - relaxation vibes
12. COFFEE - roast profiles
13. SKINCARE & BEAUTY - texture focus
14. REAL ESTATE - property vibes
15. NATURE & WILDLIFE - environmental descriptions
16. SOUND DESIGN - audio texture vocabulary
17. POETRY & LITERATURE - dense imagery
18. ASTROLOGY & SPIRITUALITY - abstract vibes

Each domain provides unique vocabulary and dimension combinations.
"""

import csv
import random
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

DIMENSIONS = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']

@dataclass
class VibeExample:
    text: str
    warmth: float = 0.5
    brightness: float = 0.5
    texture: float = 0.5
    valence: float = 0.5
    arousal: float = 0.5
    intensity: float = 0.5
    geometry: float = 0.5
    color_temperature: float = 0.5

# ============================================================
# DOMAIN 1: FOOD & CULINARY
# ============================================================
FOOD_EXAMPLES = [
    # Hot/warm foods
    {"text": "Steaming hot bowl of ramen with rich golden broth",
     "warmth": 0.92, "brightness": 0.55, "texture": 0.45, "valence": 0.82, "arousal": 0.45, "intensity": 0.65, "color_temperature": 0.75},
    {"text": "Crispy golden fried chicken with crunchy coating",
     "warmth": 0.78, "brightness": 0.68, "texture": 0.85, "valence": 0.85, "arousal": 0.55, "intensity": 0.72, "color_temperature": 0.78},
    {"text": "Warm gooey chocolate lava cake oozing dark chocolate",
     "warmth": 0.85, "brightness": 0.22, "texture": 0.25, "valence": 0.92, "arousal": 0.42, "intensity": 0.78, "color_temperature": 0.65},
    {"text": "Sizzling hot fajitas with caramelized onions and peppers",
     "warmth": 0.88, "brightness": 0.62, "texture": 0.65, "valence": 0.82, "arousal": 0.72, "intensity": 0.78, "color_temperature": 0.82},
    {"text": "Fresh baked bread with warm soft interior and crusty exterior",
     "warmth": 0.82, "brightness": 0.55, "texture": 0.72, "valence": 0.88, "arousal": 0.32, "intensity": 0.52, "color_temperature": 0.72},

    # Cold foods
    {"text": "Cold creamy vanilla ice cream melting slowly",
     "warmth": 0.12, "brightness": 0.82, "texture": 0.15, "valence": 0.85, "arousal": 0.25, "intensity": 0.42, "color_temperature": 0.55},
    {"text": "Refreshing chilled gazpacho with crisp vegetables",
     "warmth": 0.15, "brightness": 0.55, "texture": 0.45, "valence": 0.72, "arousal": 0.35, "intensity": 0.42, "color_temperature": 0.45},
    {"text": "Icy cold raw oysters on a bed of crushed ice",
     "warmth": 0.08, "brightness": 0.65, "texture": 0.35, "valence": 0.62, "arousal": 0.28, "intensity": 0.55, "color_temperature": 0.25},
    {"text": "Fresh cool cucumber salad with light vinaigrette",
     "warmth": 0.22, "brightness": 0.68, "texture": 0.35, "valence": 0.72, "arousal": 0.25, "intensity": 0.28, "color_temperature": 0.42},

    # Spicy/intense
    {"text": "Fiery hot ghost pepper curry burning with intensity",
     "warmth": 0.95, "brightness": 0.45, "texture": 0.52, "valence": 0.55, "arousal": 0.88, "intensity": 0.98, "color_temperature": 0.92},
    {"text": "Pungent fermented kimchi with sharp tangy bite",
     "warmth": 0.35, "brightness": 0.45, "texture": 0.65, "valence": 0.58, "arousal": 0.62, "intensity": 0.78, "color_temperature": 0.72},
    {"text": "Rich umami miso paste deep and complex",
     "warmth": 0.65, "brightness": 0.28, "texture": 0.35, "valence": 0.72, "arousal": 0.35, "intensity": 0.72, "color_temperature": 0.58},

    # Textures
    {"text": "Silky smooth panna cotta with delicate wobble",
     "warmth": 0.42, "brightness": 0.78, "texture": 0.08, "valence": 0.82, "arousal": 0.18, "intensity": 0.35, "color_temperature": 0.55},
    {"text": "Crunchy granola clusters with toasted nuts",
     "warmth": 0.55, "brightness": 0.62, "texture": 0.92, "valence": 0.75, "arousal": 0.45, "intensity": 0.55, "color_temperature": 0.68},
    {"text": "Tender slow-cooked pulled pork falling apart",
     "warmth": 0.82, "brightness": 0.45, "texture": 0.22, "valence": 0.85, "arousal": 0.28, "intensity": 0.65, "color_temperature": 0.72},
]

# ============================================================
# DOMAIN 2: WINE & BEVERAGES
# ============================================================
WINE_EXAMPLES = [
    # Red wines
    {"text": "Full-bodied Cabernet with velvety tannins and dark fruit",
     "warmth": 0.72, "brightness": 0.22, "texture": 0.28, "valence": 0.75, "arousal": 0.35, "intensity": 0.78, "color_temperature": 0.72},
    {"text": "Light crisp Pinot Noir with bright cherry notes",
     "warmth": 0.52, "brightness": 0.58, "texture": 0.25, "valence": 0.78, "arousal": 0.42, "intensity": 0.48, "color_temperature": 0.65},
    {"text": "Bold intense Shiraz with peppery spice and smoke",
     "warmth": 0.78, "brightness": 0.25, "texture": 0.45, "valence": 0.72, "arousal": 0.52, "intensity": 0.85, "color_temperature": 0.78},
    {"text": "Earthy rustic Chianti with rough tannins",
     "warmth": 0.62, "brightness": 0.35, "texture": 0.72, "valence": 0.65, "arousal": 0.38, "intensity": 0.62, "color_temperature": 0.68},

    # White wines
    {"text": "Crisp refreshing Sauvignon Blanc with citrus zing",
     "warmth": 0.28, "brightness": 0.82, "texture": 0.18, "valence": 0.78, "arousal": 0.52, "intensity": 0.55, "color_temperature": 0.35},
    {"text": "Rich buttery Chardonnay with oak and vanilla",
     "warmth": 0.68, "brightness": 0.62, "texture": 0.32, "valence": 0.75, "arousal": 0.28, "intensity": 0.62, "color_temperature": 0.72},
    {"text": "Light delicate Riesling with floral sweetness",
     "warmth": 0.45, "brightness": 0.78, "texture": 0.12, "valence": 0.82, "arousal": 0.25, "intensity": 0.38, "color_temperature": 0.52},
    {"text": "Bone dry Muscadet with mineral crispness",
     "warmth": 0.22, "brightness": 0.75, "texture": 0.35, "valence": 0.65, "arousal": 0.32, "intensity": 0.42, "color_temperature": 0.32},

    # Other beverages
    {"text": "Smooth aged whiskey with warm caramel and oak",
     "warmth": 0.85, "brightness": 0.42, "texture": 0.22, "valence": 0.78, "arousal": 0.35, "intensity": 0.72, "color_temperature": 0.82},
    {"text": "Bitter sharp espresso with dark roasted intensity",
     "warmth": 0.72, "brightness": 0.15, "texture": 0.35, "valence": 0.62, "arousal": 0.78, "intensity": 0.85, "color_temperature": 0.58},
    {"text": "Refreshing sparkling prosecco with light bubbles",
     "warmth": 0.35, "brightness": 0.82, "texture": 0.28, "valence": 0.82, "arousal": 0.55, "intensity": 0.42, "color_temperature": 0.48},
]

# ============================================================
# DOMAIN 3: PERFUME & FRAGRANCE
# ============================================================
PERFUME_EXAMPLES = [
    # Warm fragrances
    {"text": "Warm amber base notes with vanilla and musk",
     "warmth": 0.88, "brightness": 0.42, "texture": 0.32, "valence": 0.78, "arousal": 0.28, "intensity": 0.72, "color_temperature": 0.82},
    {"text": "Spicy oriental perfume with cinnamon and cardamom",
     "warmth": 0.85, "brightness": 0.35, "texture": 0.55, "valence": 0.72, "arousal": 0.52, "intensity": 0.78, "color_temperature": 0.85},
    {"text": "Rich oud wood dark and mysterious",
     "warmth": 0.72, "brightness": 0.15, "texture": 0.65, "valence": 0.62, "arousal": 0.35, "intensity": 0.82, "color_temperature": 0.68},
    {"text": "Sweet gourmand fragrance with caramel and praline",
     "warmth": 0.78, "brightness": 0.52, "texture": 0.25, "valence": 0.85, "arousal": 0.32, "intensity": 0.65, "color_temperature": 0.78},

    # Fresh fragrances
    {"text": "Cool fresh aquatic top notes like ocean breeze",
     "warmth": 0.22, "brightness": 0.75, "texture": 0.18, "valence": 0.78, "arousal": 0.42, "intensity": 0.45, "color_temperature": 0.28},
    {"text": "Crisp citrus burst of bergamot and lemon",
     "warmth": 0.35, "brightness": 0.88, "texture": 0.25, "valence": 0.82, "arousal": 0.62, "intensity": 0.58, "color_temperature": 0.55},
    {"text": "Clean ozonic scent like fresh mountain air",
     "warmth": 0.25, "brightness": 0.82, "texture": 0.12, "valence": 0.75, "arousal": 0.35, "intensity": 0.38, "color_temperature": 0.32},
    {"text": "Green grassy notes with cucumber and mint",
     "warmth": 0.32, "brightness": 0.72, "texture": 0.28, "valence": 0.72, "arousal": 0.38, "intensity": 0.42, "color_temperature": 0.42},

    # Floral
    {"text": "Soft powdery rose petals delicate and romantic",
     "warmth": 0.55, "brightness": 0.72, "texture": 0.15, "valence": 0.85, "arousal": 0.22, "intensity": 0.45, "color_temperature": 0.68},
    {"text": "Heady intoxicating jasmine heavy and sweet",
     "warmth": 0.65, "brightness": 0.55, "texture": 0.28, "valence": 0.78, "arousal": 0.55, "intensity": 0.78, "color_temperature": 0.62},
    {"text": "Sharp green violet leaves crisp and vegetal",
     "warmth": 0.35, "brightness": 0.62, "texture": 0.42, "valence": 0.65, "arousal": 0.42, "intensity": 0.52, "color_temperature": 0.45},
]

# ============================================================
# DOMAIN 4: FASHION & TEXTILE
# ============================================================
FASHION_EXAMPLES = [
    # Structured/formal
    {"text": "Sharp tailored suit with precise angular lines",
     "warmth": 0.38, "brightness": 0.55, "texture": 0.35, "valence": 0.68, "arousal": 0.42, "intensity": 0.65, "geometry": 0.92},
    {"text": "Crisp white cotton shirt perfectly pressed",
     "warmth": 0.42, "brightness": 0.92, "texture": 0.28, "valence": 0.72, "arousal": 0.32, "intensity": 0.45, "geometry": 0.85},
    {"text": "Sleek minimalist black dress clean silhouette",
     "warmth": 0.35, "brightness": 0.15, "texture": 0.22, "valence": 0.72, "arousal": 0.35, "intensity": 0.55, "geometry": 0.88},

    # Soft/flowing
    {"text": "Flowing bohemian maxi dress with soft drape",
     "warmth": 0.62, "brightness": 0.65, "texture": 0.18, "valence": 0.78, "arousal": 0.28, "intensity": 0.35, "geometry": 0.15},
    {"text": "Cozy oversized cashmere sweater soft and warm",
     "warmth": 0.82, "brightness": 0.55, "texture": 0.12, "valence": 0.88, "arousal": 0.15, "intensity": 0.32, "geometry": 0.22},
    {"text": "Delicate silk chiffon blouse ethereal and light",
     "warmth": 0.48, "brightness": 0.78, "texture": 0.08, "valence": 0.82, "arousal": 0.22, "intensity": 0.28, "geometry": 0.18},

    # Edgy/rough
    {"text": "Distressed leather jacket worn and rugged",
     "warmth": 0.55, "brightness": 0.32, "texture": 0.82, "valence": 0.65, "arousal": 0.55, "intensity": 0.72, "geometry": 0.45},
    {"text": "Ripped raw denim jeans with frayed edges",
     "warmth": 0.48, "brightness": 0.52, "texture": 0.85, "valence": 0.62, "arousal": 0.48, "intensity": 0.58, "geometry": 0.35},
    {"text": "Heavy chunky knit cable sweater textured wool",
     "warmth": 0.78, "brightness": 0.45, "texture": 0.78, "valence": 0.75, "arousal": 0.22, "intensity": 0.55, "geometry": 0.55},

    # Colors/materials
    {"text": "Vibrant coral summer dress bright and cheerful",
     "warmth": 0.68, "brightness": 0.82, "texture": 0.28, "valence": 0.88, "arousal": 0.55, "intensity": 0.62, "color_temperature": 0.78},
    {"text": "Deep navy velvet gown rich and luxurious",
     "warmth": 0.45, "brightness": 0.22, "texture": 0.25, "valence": 0.78, "arousal": 0.28, "intensity": 0.72, "color_temperature": 0.25},
    {"text": "Shimmering gold sequin top glamorous sparkle",
     "warmth": 0.72, "brightness": 0.92, "texture": 0.55, "valence": 0.85, "arousal": 0.72, "intensity": 0.78, "color_temperature": 0.82},
]

# ============================================================
# DOMAIN 5: INTERIOR DESIGN
# ============================================================
INTERIOR_EXAMPLES = [
    # Cozy/warm
    {"text": "Cozy hygge living room with soft throws and candles",
     "warmth": 0.88, "brightness": 0.35, "texture": 0.55, "valence": 0.92, "arousal": 0.15, "intensity": 0.42, "geometry": 0.28, "color_temperature": 0.78},
    {"text": "Rustic farmhouse kitchen with wooden beams and copper pots",
     "warmth": 0.78, "brightness": 0.55, "texture": 0.72, "valence": 0.82, "arousal": 0.28, "intensity": 0.52, "geometry": 0.35, "color_temperature": 0.72},
    {"text": "Warm Mediterranean villa with terracotta tiles and ochre walls",
     "warmth": 0.85, "brightness": 0.72, "texture": 0.65, "valence": 0.85, "arousal": 0.35, "intensity": 0.58, "geometry": 0.42, "color_temperature": 0.82},

    # Modern/minimal
    {"text": "Sleek minimalist apartment with white walls and clean lines",
     "warmth": 0.35, "brightness": 0.92, "texture": 0.12, "valence": 0.68, "arousal": 0.22, "intensity": 0.38, "geometry": 0.95},
    {"text": "Modern Scandinavian bedroom light wood and neutral tones",
     "warmth": 0.55, "brightness": 0.82, "texture": 0.28, "valence": 0.78, "arousal": 0.18, "intensity": 0.35, "geometry": 0.78, "color_temperature": 0.55},
    {"text": "Contemporary open concept loft with concrete and glass",
     "warmth": 0.32, "brightness": 0.75, "texture": 0.45, "valence": 0.65, "arousal": 0.35, "intensity": 0.52, "geometry": 0.88},

    # Industrial/raw
    {"text": "Raw industrial warehouse with exposed brick and steel",
     "warmth": 0.42, "brightness": 0.45, "texture": 0.85, "valence": 0.55, "arousal": 0.42, "intensity": 0.68, "geometry": 0.72, "color_temperature": 0.55},
    {"text": "Stark brutalist interior with bare concrete surfaces",
     "warmth": 0.25, "brightness": 0.52, "texture": 0.78, "valence": 0.35, "arousal": 0.32, "intensity": 0.72, "geometry": 0.88},

    # Luxurious
    {"text": "Opulent baroque parlor with gilded mirrors and velvet",
     "warmth": 0.72, "brightness": 0.55, "texture": 0.65, "valence": 0.78, "arousal": 0.42, "intensity": 0.82, "geometry": 0.45, "color_temperature": 0.78},
    {"text": "Glamorous Art Deco lounge with gold and black accents",
     "warmth": 0.65, "brightness": 0.58, "texture": 0.42, "valence": 0.82, "arousal": 0.52, "intensity": 0.75, "geometry": 0.85, "color_temperature": 0.78},
]

# ============================================================
# DOMAIN 6: FILM & CINEMA
# ============================================================
FILM_EXAMPLES = [
    # Dark/noir
    {"text": "Classic film noir with shadowy cinematography and rain",
     "warmth": 0.28, "brightness": 0.15, "texture": 0.55, "valence": 0.32, "arousal": 0.55, "intensity": 0.72, "geometry": 0.65, "color_temperature": 0.35},
    {"text": "Gritty crime thriller dark alleyways and neon",
     "warmth": 0.38, "brightness": 0.28, "texture": 0.68, "valence": 0.28, "arousal": 0.72, "intensity": 0.82, "geometry": 0.55},
    {"text": "Atmospheric horror with creeping dread and shadows",
     "warmth": 0.22, "brightness": 0.12, "texture": 0.62, "valence": 0.12, "arousal": 0.68, "intensity": 0.78, "color_temperature": 0.32},

    # Bright/colorful
    {"text": "Vibrant Wes Anderson palette pastel symmetry",
     "warmth": 0.58, "brightness": 0.78, "texture": 0.25, "valence": 0.75, "arousal": 0.42, "intensity": 0.55, "geometry": 0.92, "color_temperature": 0.68},
    {"text": "Colorful Bollywood musical bright costumes and dancing",
     "warmth": 0.75, "brightness": 0.88, "texture": 0.45, "valence": 0.92, "arousal": 0.88, "intensity": 0.85, "color_temperature": 0.78},
    {"text": "Saturated neon-soaked cyberpunk aesthetic",
     "warmth": 0.48, "brightness": 0.72, "texture": 0.52, "valence": 0.55, "arousal": 0.72, "intensity": 0.82, "geometry": 0.78, "color_temperature": 0.55},

    # Naturalistic
    {"text": "Soft handheld documentary intimate and raw",
     "warmth": 0.55, "brightness": 0.52, "texture": 0.58, "valence": 0.55, "arousal": 0.35, "intensity": 0.48, "geometry": 0.22},
    {"text": "Golden hour magic realism warm nostalgic glow",
     "warmth": 0.82, "brightness": 0.68, "texture": 0.28, "valence": 0.85, "arousal": 0.28, "intensity": 0.55, "color_temperature": 0.85},
    {"text": "Bleak social realism grey urban landscapes",
     "warmth": 0.32, "brightness": 0.42, "texture": 0.62, "valence": 0.25, "arousal": 0.32, "intensity": 0.55, "geometry": 0.55, "color_temperature": 0.42},
]

# ============================================================
# DOMAIN 7: VIDEO GAMES
# ============================================================
GAME_EXAMPLES = [
    # Dark/atmospheric
    {"text": "Dark Souls oppressive gothic ruins and despair",
     "warmth": 0.25, "brightness": 0.18, "texture": 0.82, "valence": 0.15, "arousal": 0.58, "intensity": 0.85, "geometry": 0.55, "color_temperature": 0.35},
    {"text": "Silent Hill foggy psychological horror distorted",
     "warmth": 0.32, "brightness": 0.22, "texture": 0.72, "valence": 0.08, "arousal": 0.72, "intensity": 0.82, "geometry": 0.28},
    {"text": "Limbo stark black and white silhouette platformer",
     "warmth": 0.28, "brightness": 0.35, "texture": 0.42, "valence": 0.32, "arousal": 0.42, "intensity": 0.58, "geometry": 0.55, "color_temperature": 0.45},

    # Bright/colorful
    {"text": "Nintendo vibrant cel-shaded cartoon world",
     "warmth": 0.68, "brightness": 0.92, "texture": 0.25, "valence": 0.92, "arousal": 0.65, "intensity": 0.55, "geometry": 0.45, "color_temperature": 0.68},
    {"text": "Colorful indie puzzle game cheerful and whimsical",
     "warmth": 0.62, "brightness": 0.85, "texture": 0.22, "valence": 0.88, "arousal": 0.45, "intensity": 0.42, "geometry": 0.55, "color_temperature": 0.62},
    {"text": "Sunny open world adventure lush green landscapes",
     "warmth": 0.65, "brightness": 0.82, "texture": 0.45, "valence": 0.85, "arousal": 0.52, "intensity": 0.48, "geometry": 0.28, "color_temperature": 0.58},

    # Realistic/gritty
    {"text": "Photorealistic military shooter gritty and tactical",
     "warmth": 0.42, "brightness": 0.48, "texture": 0.72, "valence": 0.38, "arousal": 0.78, "intensity": 0.85, "geometry": 0.65},
    {"text": "Post-apocalyptic wasteland brown and desolate",
     "warmth": 0.45, "brightness": 0.42, "texture": 0.78, "valence": 0.22, "arousal": 0.42, "intensity": 0.65, "geometry": 0.35, "color_temperature": 0.55},
    {"text": "Cyberpunk dystopia neon rain and chrome",
     "warmth": 0.42, "brightness": 0.55, "texture": 0.58, "valence": 0.42, "arousal": 0.68, "intensity": 0.78, "geometry": 0.78, "color_temperature": 0.52},
]

# ============================================================
# DOMAIN 8: ART MOVEMENTS
# ============================================================
ART_EXAMPLES = [
    # Classical/traditional
    {"text": "Renaissance oil painting rich golden chiaroscuro",
     "warmth": 0.72, "brightness": 0.45, "texture": 0.55, "valence": 0.72, "arousal": 0.32, "intensity": 0.68, "geometry": 0.55, "color_temperature": 0.75},
    {"text": "Baroque dramatic lighting intense shadows and gold",
     "warmth": 0.68, "brightness": 0.38, "texture": 0.62, "valence": 0.65, "arousal": 0.55, "intensity": 0.82, "geometry": 0.48, "color_temperature": 0.72},
    {"text": "Dutch masters still life warm candlelit realism",
     "warmth": 0.75, "brightness": 0.35, "texture": 0.58, "valence": 0.68, "arousal": 0.22, "intensity": 0.55, "color_temperature": 0.72},

    # Impressionist/soft
    {"text": "Impressionist soft dappled light garden scene",
     "warmth": 0.62, "brightness": 0.78, "texture": 0.35, "valence": 0.82, "arousal": 0.28, "intensity": 0.42, "geometry": 0.18, "color_temperature": 0.65},
    {"text": "Monet water lilies dreamy atmospheric blur",
     "warmth": 0.55, "brightness": 0.72, "texture": 0.22, "valence": 0.85, "arousal": 0.15, "intensity": 0.35, "geometry": 0.12, "color_temperature": 0.55},
    {"text": "Pointillism tiny dots creating luminous color",
     "warmth": 0.58, "brightness": 0.82, "texture": 0.65, "valence": 0.78, "arousal": 0.32, "intensity": 0.52, "geometry": 0.55, "color_temperature": 0.62},

    # Modern/abstract
    {"text": "Cubist fractured geometric planes sharp angles",
     "warmth": 0.45, "brightness": 0.52, "texture": 0.68, "valence": 0.52, "arousal": 0.55, "intensity": 0.72, "geometry": 0.95},
    {"text": "Abstract expressionist bold gestural brushstrokes",
     "warmth": 0.58, "brightness": 0.55, "texture": 0.85, "valence": 0.62, "arousal": 0.75, "intensity": 0.82, "geometry": 0.15},
    {"text": "Minimalist white canvas single black line",
     "warmth": 0.35, "brightness": 0.95, "texture": 0.08, "valence": 0.55, "arousal": 0.08, "intensity": 0.25, "geometry": 0.92},
    {"text": "Pop art bright bold commercial colors flat",
     "warmth": 0.58, "brightness": 0.92, "texture": 0.22, "valence": 0.75, "arousal": 0.65, "intensity": 0.72, "geometry": 0.78, "color_temperature": 0.65},

    # Dark/intense
    {"text": "Expressionist distorted anguished emotional",
     "warmth": 0.42, "brightness": 0.32, "texture": 0.75, "valence": 0.18, "arousal": 0.82, "intensity": 0.88, "geometry": 0.35},
    {"text": "Surrealist dreamscape melting clocks bizarre",
     "warmth": 0.48, "brightness": 0.55, "texture": 0.42, "valence": 0.45, "arousal": 0.52, "intensity": 0.68, "geometry": 0.22},
]

# ============================================================
# DOMAIN 9: TRAVEL & HOTELS
# ============================================================
TRAVEL_EXAMPLES = [
    # Luxury
    {"text": "Five star resort spa infinity pool overlooking ocean",
     "warmth": 0.72, "brightness": 0.85, "texture": 0.22, "valence": 0.95, "arousal": 0.32, "intensity": 0.62, "geometry": 0.55, "color_temperature": 0.55},
    {"text": "Boutique hotel with designer furniture and curated art",
     "warmth": 0.58, "brightness": 0.68, "texture": 0.35, "valence": 0.82, "arousal": 0.35, "intensity": 0.58, "geometry": 0.72},
    {"text": "Glamorous rooftop bar with city lights twinkling",
     "warmth": 0.62, "brightness": 0.55, "texture": 0.28, "valence": 0.88, "arousal": 0.68, "intensity": 0.72, "geometry": 0.75, "color_temperature": 0.72},

    # Rustic/natural
    {"text": "Cozy mountain lodge with crackling fireplace and pine",
     "warmth": 0.88, "brightness": 0.38, "texture": 0.68, "valence": 0.88, "arousal": 0.18, "intensity": 0.52, "geometry": 0.32, "color_temperature": 0.75},
    {"text": "Remote eco-lodge in rainforest surrounded by wildlife",
     "warmth": 0.72, "brightness": 0.42, "texture": 0.75, "valence": 0.82, "arousal": 0.45, "intensity": 0.62, "geometry": 0.15, "color_temperature": 0.55},
    {"text": "Beachside bungalow with thatched roof and hammock",
     "warmth": 0.78, "brightness": 0.78, "texture": 0.58, "valence": 0.92, "arousal": 0.22, "intensity": 0.38, "geometry": 0.22, "color_temperature": 0.68},

    # Urban/modern
    {"text": "Sleek urban hotel with floor to ceiling windows",
     "warmth": 0.42, "brightness": 0.82, "texture": 0.15, "valence": 0.72, "arousal": 0.52, "intensity": 0.58, "geometry": 0.92},
    {"text": "Bustling hostel common room backpacker energy",
     "warmth": 0.62, "brightness": 0.65, "texture": 0.55, "valence": 0.78, "arousal": 0.82, "intensity": 0.68, "geometry": 0.35},
]

# ============================================================
# DOMAIN 10: SPA & WELLNESS
# ============================================================
SPA_EXAMPLES = [
    # Relaxation
    {"text": "Tranquil zen spa with soft lighting and bamboo",
     "warmth": 0.62, "brightness": 0.35, "texture": 0.42, "valence": 0.88, "arousal": 0.08, "intensity": 0.22, "geometry": 0.55, "color_temperature": 0.55},
    {"text": "Warm aromatherapy massage with lavender oil",
     "warmth": 0.78, "brightness": 0.28, "texture": 0.18, "valence": 0.92, "arousal": 0.05, "intensity": 0.32, "color_temperature": 0.62},
    {"text": "Peaceful meditation garden with flowing water",
     "warmth": 0.55, "brightness": 0.58, "texture": 0.38, "valence": 0.85, "arousal": 0.08, "intensity": 0.22, "geometry": 0.32},
    {"text": "Gentle floating in sensory deprivation tank",
     "warmth": 0.55, "brightness": 0.02, "texture": 0.05, "valence": 0.72, "arousal": 0.02, "intensity": 0.08, "geometry": 0.15},

    # Invigorating
    {"text": "Invigorating cold plunge pool shocking and bracing",
     "warmth": 0.05, "brightness": 0.72, "texture": 0.28, "valence": 0.55, "arousal": 0.92, "intensity": 0.85, "color_temperature": 0.15},
    {"text": "Eucalyptus steam room hot and cleansing",
     "warmth": 0.92, "brightness": 0.25, "texture": 0.15, "valence": 0.75, "arousal": 0.42, "intensity": 0.65},
    {"text": "Deep tissue massage intense pressure and release",
     "warmth": 0.72, "brightness": 0.32, "texture": 0.62, "valence": 0.65, "arousal": 0.55, "intensity": 0.82},
]

# ============================================================
# DOMAIN 11: COFFEE
# ============================================================
COFFEE_EXAMPLES = [
    # Light roasts
    {"text": "Bright fruity Ethiopian coffee with blueberry notes",
     "warmth": 0.55, "brightness": 0.82, "texture": 0.28, "valence": 0.82, "arousal": 0.58, "intensity": 0.52, "color_temperature": 0.55},
    {"text": "Delicate floral Kenyan single origin citrus acidity",
     "warmth": 0.48, "brightness": 0.78, "texture": 0.22, "valence": 0.78, "arousal": 0.55, "intensity": 0.48, "color_temperature": 0.52},
    {"text": "Light crisp cold brew smooth and refreshing",
     "warmth": 0.22, "brightness": 0.72, "texture": 0.18, "valence": 0.78, "arousal": 0.52, "intensity": 0.42, "color_temperature": 0.42},

    # Dark roasts
    {"text": "Dark rich Italian espresso roast bitter and bold",
     "warmth": 0.72, "brightness": 0.15, "texture": 0.42, "valence": 0.68, "arousal": 0.78, "intensity": 0.88, "color_temperature": 0.62},
    {"text": "Smoky French roast deep and charred",
     "warmth": 0.68, "brightness": 0.12, "texture": 0.55, "valence": 0.62, "arousal": 0.68, "intensity": 0.82, "color_temperature": 0.58},
    {"text": "Velvety dark chocolate notes with caramel finish",
     "warmth": 0.72, "brightness": 0.25, "texture": 0.22, "valence": 0.82, "arousal": 0.45, "intensity": 0.68, "color_temperature": 0.68},

    # Medium
    {"text": "Balanced medium roast nutty and smooth",
     "warmth": 0.62, "brightness": 0.52, "texture": 0.28, "valence": 0.75, "arousal": 0.48, "intensity": 0.52, "color_temperature": 0.58},
]

# ============================================================
# DOMAIN 12: NATURE & WILDLIFE
# ============================================================
NATURE_EXAMPLES = [
    # Tropical
    {"text": "Dense humid rainforest canopy alive with sounds",
     "warmth": 0.82, "brightness": 0.28, "texture": 0.78, "valence": 0.72, "arousal": 0.65, "intensity": 0.72, "geometry": 0.12, "color_temperature": 0.55},
    {"text": "Colorful coral reef teeming with tropical fish",
     "warmth": 0.68, "brightness": 0.72, "texture": 0.65, "valence": 0.88, "arousal": 0.55, "intensity": 0.68, "geometry": 0.15, "color_temperature": 0.45},
    {"text": "Steamy volcanic hot springs with mineral pools",
     "warmth": 0.92, "brightness": 0.45, "texture": 0.42, "valence": 0.72, "arousal": 0.38, "intensity": 0.68, "color_temperature": 0.65},

    # Cold/harsh
    {"text": "Barren windswept arctic tundra endless white",
     "warmth": 0.05, "brightness": 0.88, "texture": 0.35, "valence": 0.35, "arousal": 0.42, "intensity": 0.68, "geometry": 0.25, "color_temperature": 0.15},
    {"text": "Towering glacial ice formations ancient and blue",
     "warmth": 0.08, "brightness": 0.75, "texture": 0.55, "valence": 0.55, "arousal": 0.32, "intensity": 0.72, "geometry": 0.65, "color_temperature": 0.12},
    {"text": "Desolate desert dunes scorching and silent",
     "warmth": 0.95, "brightness": 0.92, "texture": 0.42, "valence": 0.38, "arousal": 0.22, "intensity": 0.72, "geometry": 0.35, "color_temperature": 0.82},

    # Temperate
    {"text": "Gentle rolling hills with wildflower meadows",
     "warmth": 0.62, "brightness": 0.78, "texture": 0.45, "valence": 0.88, "arousal": 0.28, "intensity": 0.35, "geometry": 0.25, "color_temperature": 0.58},
    {"text": "Misty Scottish highlands moody and windswept",
     "warmth": 0.35, "brightness": 0.42, "texture": 0.55, "valence": 0.55, "arousal": 0.38, "intensity": 0.52, "geometry": 0.28, "color_temperature": 0.42},
    {"text": "Pristine alpine lake crystal clear and cold",
     "warmth": 0.18, "brightness": 0.85, "texture": 0.12, "valence": 0.82, "arousal": 0.25, "intensity": 0.45, "geometry": 0.48, "color_temperature": 0.28},
]

# ============================================================
# DOMAIN 13: SOUND DESIGN
# ============================================================
SOUND_EXAMPLES = [
    # Warm sounds
    {"text": "Warm analog synthesizer with rich harmonics",
     "warmth": 0.82, "brightness": 0.45, "texture": 0.35, "valence": 0.72, "arousal": 0.35, "intensity": 0.55, "geometry": 0.42},
    {"text": "Vintage tube saturation soft and creamy",
     "warmth": 0.85, "brightness": 0.38, "texture": 0.28, "valence": 0.75, "arousal": 0.28, "intensity": 0.52, "geometry": 0.35},
    {"text": "Deep warm bass rumbling and resonant",
     "warmth": 0.78, "brightness": 0.18, "texture": 0.42, "valence": 0.68, "arousal": 0.42, "intensity": 0.72},

    # Bright sounds
    {"text": "Crisp clean high frequencies sparkling and clear",
     "warmth": 0.28, "brightness": 0.95, "texture": 0.22, "valence": 0.75, "arousal": 0.55, "intensity": 0.52, "geometry": 0.72},
    {"text": "Bright shimmering reverb ethereal and airy",
     "warmth": 0.35, "brightness": 0.88, "texture": 0.15, "valence": 0.78, "arousal": 0.32, "intensity": 0.42, "geometry": 0.28},
    {"text": "Sharp transient attack punchy and precise",
     "warmth": 0.38, "brightness": 0.72, "texture": 0.65, "valence": 0.62, "arousal": 0.72, "intensity": 0.78, "geometry": 0.85},

    # Dark sounds
    {"text": "Dark rumbling sub bass ominous and heavy",
     "warmth": 0.55, "brightness": 0.08, "texture": 0.48, "valence": 0.35, "arousal": 0.48, "intensity": 0.82},
    {"text": "Muffled lo-fi texture dusty and degraded",
     "warmth": 0.58, "brightness": 0.28, "texture": 0.78, "valence": 0.52, "arousal": 0.25, "intensity": 0.45, "geometry": 0.32},
    {"text": "Harsh distortion clipping and aggressive",
     "warmth": 0.48, "brightness": 0.55, "texture": 0.95, "valence": 0.28, "arousal": 0.88, "intensity": 0.95, "geometry": 0.58},
]

# ============================================================
# DOMAIN 14: POETRY & LITERATURE
# ============================================================
LITERATURE_EXAMPLES = [
    # Gothic/dark
    {"text": "Gothic castle on stormy moor lightning and ravens",
     "warmth": 0.22, "brightness": 0.15, "texture": 0.75, "valence": 0.18, "arousal": 0.68, "intensity": 0.82, "geometry": 0.58, "color_temperature": 0.35},
    {"text": "Decaying Victorian mansion dust and cobwebs",
     "warmth": 0.32, "brightness": 0.22, "texture": 0.82, "valence": 0.22, "arousal": 0.38, "intensity": 0.62, "geometry": 0.55},
    {"text": "Dark romantic poetry moonlight and longing",
     "warmth": 0.38, "brightness": 0.22, "texture": 0.35, "valence": 0.42, "arousal": 0.45, "intensity": 0.68, "color_temperature": 0.35},

    # Nature poetry
    {"text": "Wordsworth daffodils dancing golden in breeze",
     "warmth": 0.65, "brightness": 0.82, "texture": 0.28, "valence": 0.92, "arousal": 0.45, "intensity": 0.52, "geometry": 0.18, "color_temperature": 0.78},
    {"text": "Haiku cherry blossoms falling gentle spring rain",
     "warmth": 0.52, "brightness": 0.68, "texture": 0.22, "valence": 0.78, "arousal": 0.18, "intensity": 0.32, "geometry": 0.22, "color_temperature": 0.65},
    {"text": "Transcendentalist woods quiet solitude nature",
     "warmth": 0.55, "brightness": 0.55, "texture": 0.52, "valence": 0.75, "arousal": 0.12, "intensity": 0.38, "geometry": 0.22},

    # Modernist
    {"text": "Fragmentary modernist poem sharp angles urban",
     "warmth": 0.38, "brightness": 0.52, "texture": 0.72, "valence": 0.42, "arousal": 0.55, "intensity": 0.65, "geometry": 0.78},
    {"text": "Stream of consciousness flowing thoughts unending",
     "warmth": 0.52, "brightness": 0.48, "texture": 0.35, "valence": 0.52, "arousal": 0.58, "intensity": 0.55, "geometry": 0.12},
]

# ============================================================
# DOMAIN 15: ASTROLOGY & SPIRITUALITY
# ============================================================
ASTROLOGY_EXAMPLES = [
    # Fire signs
    {"text": "Fiery passionate Aries energy bold and impulsive",
     "warmth": 0.92, "brightness": 0.78, "texture": 0.55, "valence": 0.72, "arousal": 0.92, "intensity": 0.88, "color_temperature": 0.92},
    {"text": "Dramatic Leo energy golden confident radiant",
     "warmth": 0.85, "brightness": 0.88, "texture": 0.38, "valence": 0.82, "arousal": 0.78, "intensity": 0.82, "color_temperature": 0.85},
    {"text": "Adventurous Sagittarius energy expansive free",
     "warmth": 0.75, "brightness": 0.78, "texture": 0.42, "valence": 0.85, "arousal": 0.82, "intensity": 0.72, "geometry": 0.22, "color_temperature": 0.78},

    # Earth signs
    {"text": "Grounded Taurus energy stable sensual earthy",
     "warmth": 0.65, "brightness": 0.52, "texture": 0.68, "valence": 0.72, "arousal": 0.25, "intensity": 0.58, "geometry": 0.55, "color_temperature": 0.62},
    {"text": "Practical Virgo energy analytical cool precise",
     "warmth": 0.42, "brightness": 0.68, "texture": 0.32, "valence": 0.62, "arousal": 0.38, "intensity": 0.52, "geometry": 0.88},
    {"text": "Ambitious Capricorn energy structured determined",
     "warmth": 0.38, "brightness": 0.48, "texture": 0.55, "valence": 0.58, "arousal": 0.55, "intensity": 0.72, "geometry": 0.85},

    # Water signs
    {"text": "Deep emotional Cancer energy nurturing protective",
     "warmth": 0.72, "brightness": 0.42, "texture": 0.32, "valence": 0.65, "arousal": 0.35, "intensity": 0.68, "color_temperature": 0.58},
    {"text": "Intense Scorpio energy mysterious transformative",
     "warmth": 0.55, "brightness": 0.18, "texture": 0.62, "valence": 0.45, "arousal": 0.72, "intensity": 0.92, "color_temperature": 0.45},
    {"text": "Dreamy Pisces energy flowing intuitive ethereal",
     "warmth": 0.52, "brightness": 0.55, "texture": 0.18, "valence": 0.72, "arousal": 0.22, "intensity": 0.48, "geometry": 0.12, "color_temperature": 0.45},

    # Air signs
    {"text": "Intellectual Aquarius energy innovative detached",
     "warmth": 0.32, "brightness": 0.72, "texture": 0.28, "valence": 0.65, "arousal": 0.55, "intensity": 0.58, "geometry": 0.78, "color_temperature": 0.35},
    {"text": "Harmonious Libra energy balanced aesthetic",
     "warmth": 0.55, "brightness": 0.72, "texture": 0.22, "valence": 0.78, "arousal": 0.38, "intensity": 0.45, "geometry": 0.75, "color_temperature": 0.55},
    {"text": "Curious Gemini energy quick changeable bright",
     "warmth": 0.52, "brightness": 0.82, "texture": 0.28, "valence": 0.75, "arousal": 0.78, "intensity": 0.55, "geometry": 0.42, "color_temperature": 0.55},
]

# ============================================================
# COMPILE ALL DOMAINS
# ============================================================
ALL_DOMAIN_EXAMPLES = (
    FOOD_EXAMPLES +
    WINE_EXAMPLES +
    PERFUME_EXAMPLES +
    FASHION_EXAMPLES +
    INTERIOR_EXAMPLES +
    FILM_EXAMPLES +
    GAME_EXAMPLES +
    ART_EXAMPLES +
    TRAVEL_EXAMPLES +
    SPA_EXAMPLES +
    COFFEE_EXAMPLES +
    NATURE_EXAMPLES +
    SOUND_EXAMPLES +
    LITERATURE_EXAMPLES +
    ASTROLOGY_EXAMPLES
)

def dict_to_example(d: Dict) -> VibeExample:
    """Convert dict to VibeExample."""
    ex = VibeExample(text=d["text"])
    for dim in DIMENSIONS:
        if dim in d:
            setattr(ex, dim, d[dim])
    return ex

def expand_with_variations(examples: List[Dict]) -> List[VibeExample]:
    """Create variations of examples."""
    result = []

    prefixes = [
        "", "The feeling of ", "Like ", "Experience of ",
        "Reminds me of ", "The essence of ", "Pure ",
    ]

    for ex_dict in examples:
        # Original
        result.append(dict_to_example(ex_dict))

        # Add 2-3 variations
        for _ in range(random.randint(2, 3)):
            prefix = random.choice(prefixes[1:])
            new_dict = ex_dict.copy()
            new_dict["text"] = prefix + ex_dict["text"].lower()

            # Slight value variation
            for dim in DIMENSIONS:
                if dim in new_dict and dim != "text":
                    new_dict[dim] = max(0.02, min(0.98, new_dict[dim] + random.uniform(-0.04, 0.04)))

            result.append(dict_to_example(new_dict))

    return result

def analyze_domain_coverage(examples: List[VibeExample]):
    """Analyze coverage by domain and dimension."""
    print("\n" + "=" * 70)
    print("DOMAIN EXPANSION ANALYSIS")
    print("=" * 70)

    print(f"\nTotal new examples: {len(examples)}")

    # Per-dimension stats
    for dim in DIMENSIONS:
        values = [getattr(ex, dim) for ex in examples]
        neutral = sum(1 for v in values if 0.45 <= v <= 0.55)
        low = sum(1 for v in values if v < 0.3)
        high = sum(1 for v in values if v > 0.7)

        print(f"\n{dim}:")
        print(f"  Neutral: {neutral:4d} ({neutral/len(values)*100:5.1f}%)")
        print(f"  Low:     {low:4d} ({low/len(values)*100:5.1f}%)")
        print(f"  High:    {high:4d} ({high/len(values)*100:5.1f}%)")

    # Multi-dim stats
    multi = [0] * 9
    for ex in examples:
        active = sum(1 for d in DIMENSIONS if getattr(ex, d) < 0.4 or getattr(ex, d) > 0.6)
        multi[min(active, 8)] += 1

    print(f"\n{'='*70}")
    print("ACTIVE DIMENSIONS PER EXAMPLE:")
    for i, count in enumerate(multi):
        if count > 0:
            print(f"  {i} dims: {count:4d} ({count/len(examples)*100:5.1f}%)")

def save_examples(examples: List[VibeExample], filename: str):
    """Save to CSV."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text'] + DIMENSIONS)
        for ex in examples:
            writer.writerow([ex.text] + [round(getattr(ex, d), 3) for d in DIMENSIONS])
    print(f"\nSaved {len(examples)} examples to {filename}")

if __name__ == "__main__":
    print("EXPANDING VIBE DOMAINS")
    print("=" * 70)

    print(f"\nDomains covered:")
    print("  1. Food & Culinary")
    print("  2. Wine & Beverages")
    print("  3. Perfume & Fragrance")
    print("  4. Fashion & Textile")
    print("  5. Interior Design")
    print("  6. Film & Cinema")
    print("  7. Video Games")
    print("  8. Art Movements")
    print("  9. Travel & Hotels")
    print("  10. Spa & Wellness")
    print("  11. Coffee")
    print("  12. Nature & Wildlife")
    print("  13. Sound Design")
    print("  14. Poetry & Literature")
    print("  15. Astrology & Spirituality")

    print(f"\nBase examples: {len(ALL_DOMAIN_EXAMPLES)}")

    # Expand with variations
    expanded = expand_with_variations(ALL_DOMAIN_EXAMPLES)

    # Analyze
    analyze_domain_coverage(expanded)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"domain_expanded_vibe_data_{timestamp}.csv"
    save_examples(expanded, filename)

    print(f"\n✅ Domain expansion complete: {filename}")
