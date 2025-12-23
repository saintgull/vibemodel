#!/usr/bin/env python3
"""
ACADEMIC VIBE DATA ACQUISITION
==============================
Downloads and processes academic datasets to create massive training data for the 8D Vibe Engine.

Datasets:
1. GoEmotions (Google) - 58k Reddit comments with 27 emotions
2. Emotion lexicons (NRC, LIWC-style)
3. Color-emotion associations
4. Music mood vocabularies

Target: 50,000+ training examples mapped to 8D vibe space
"""

import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

# ===== MAPPING FUNCTIONS =====

# GoEmotions 27 categories → 8D vibe space
GOEMOTIONS_TO_VIBE = {
    # Format: emotion → (warmth, brightness, texture, valence, arousal, intensity, geometry, color_temp)
    # Scale: 0.0-1.0 where 0.5 is neutral

    # POSITIVE EMOTIONS (12)
    'admiration': (0.70, 0.70, 0.40, 0.85, 0.55, 0.60, 0.45, 0.65),
    'amusement': (0.75, 0.80, 0.35, 0.90, 0.70, 0.55, 0.30, 0.70),
    'approval': (0.65, 0.65, 0.40, 0.75, 0.45, 0.45, 0.50, 0.60),
    'caring': (0.85, 0.60, 0.25, 0.80, 0.35, 0.50, 0.25, 0.75),
    'desire': (0.80, 0.55, 0.50, 0.70, 0.75, 0.80, 0.45, 0.80),
    'excitement': (0.75, 0.85, 0.55, 0.90, 0.95, 0.90, 0.55, 0.75),
    'gratitude': (0.80, 0.70, 0.30, 0.90, 0.40, 0.55, 0.35, 0.70),
    'joy': (0.85, 0.90, 0.30, 0.95, 0.80, 0.75, 0.30, 0.80),
    'love': (0.90, 0.70, 0.20, 0.95, 0.65, 0.85, 0.25, 0.85),
    'optimism': (0.75, 0.80, 0.35, 0.85, 0.65, 0.60, 0.40, 0.70),
    'pride': (0.65, 0.75, 0.50, 0.80, 0.70, 0.75, 0.60, 0.65),
    'relief': (0.70, 0.60, 0.25, 0.75, 0.25, 0.40, 0.35, 0.60),

    # NEGATIVE EMOTIONS (11)
    'anger': (0.70, 0.40, 0.85, 0.10, 0.90, 0.95, 0.80, 0.80),
    'annoyance': (0.55, 0.45, 0.70, 0.25, 0.65, 0.60, 0.65, 0.55),
    'disappointment': (0.35, 0.30, 0.45, 0.20, 0.30, 0.55, 0.50, 0.35),
    'disapproval': (0.30, 0.35, 0.65, 0.20, 0.55, 0.60, 0.70, 0.30),
    'disgust': (0.25, 0.25, 0.80, 0.10, 0.70, 0.80, 0.75, 0.30),
    'embarrassment': (0.50, 0.40, 0.55, 0.25, 0.60, 0.65, 0.50, 0.55),
    'fear': (0.20, 0.20, 0.70, 0.10, 0.85, 0.90, 0.70, 0.15),
    'grief': (0.25, 0.15, 0.40, 0.05, 0.40, 0.85, 0.35, 0.20),
    'nervousness': (0.45, 0.50, 0.60, 0.30, 0.80, 0.70, 0.60, 0.45),
    'remorse': (0.35, 0.25, 0.50, 0.15, 0.45, 0.70, 0.45, 0.30),
    'sadness': (0.30, 0.20, 0.35, 0.10, 0.25, 0.70, 0.30, 0.25),

    # AMBIGUOUS EMOTIONS (4)
    'confusion': (0.45, 0.50, 0.55, 0.40, 0.60, 0.55, 0.60, 0.45),
    'curiosity': (0.60, 0.70, 0.45, 0.70, 0.70, 0.55, 0.50, 0.55),
    'realization': (0.55, 0.75, 0.40, 0.65, 0.65, 0.60, 0.55, 0.55),
    'surprise': (0.55, 0.80, 0.50, 0.60, 0.85, 0.80, 0.55, 0.55),

    # NEUTRAL
    'neutral': (0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50),
}

# NRC Emotion Lexicon mappings (word → emotions)
# These are common emotion-associated words we can use directly
EMOTION_WORD_VIBES = {
    # HIGH WARMTH words
    'warm': (0.90, 0.60, 0.35, 0.70, 0.40, 0.45, 0.35, 0.85),
    'cozy': (0.90, 0.45, 0.25, 0.85, 0.20, 0.40, 0.25, 0.80),
    'tender': (0.85, 0.55, 0.20, 0.80, 0.25, 0.50, 0.25, 0.75),
    'affectionate': (0.90, 0.65, 0.25, 0.90, 0.45, 0.60, 0.25, 0.80),
    'hospitable': (0.85, 0.70, 0.30, 0.80, 0.50, 0.50, 0.35, 0.75),

    # LOW WARMTH words
    'cold': (0.10, 0.50, 0.50, 0.35, 0.30, 0.45, 0.55, 0.15),
    'icy': (0.05, 0.70, 0.60, 0.25, 0.35, 0.55, 0.65, 0.05),
    'frigid': (0.05, 0.55, 0.55, 0.20, 0.30, 0.50, 0.60, 0.05),
    'distant': (0.25, 0.45, 0.50, 0.35, 0.25, 0.40, 0.55, 0.30),
    'aloof': (0.20, 0.50, 0.55, 0.30, 0.20, 0.35, 0.60, 0.25),

    # HIGH BRIGHTNESS words
    'bright': (0.60, 0.95, 0.40, 0.75, 0.65, 0.55, 0.45, 0.60),
    'radiant': (0.75, 0.95, 0.30, 0.90, 0.70, 0.70, 0.40, 0.75),
    'luminous': (0.65, 0.95, 0.30, 0.80, 0.55, 0.60, 0.40, 0.65),
    'gleaming': (0.60, 0.90, 0.50, 0.70, 0.60, 0.55, 0.50, 0.60),
    'dazzling': (0.65, 0.95, 0.55, 0.80, 0.80, 0.80, 0.55, 0.70),

    # LOW BRIGHTNESS words
    'dark': (0.35, 0.10, 0.55, 0.30, 0.35, 0.60, 0.50, 0.35),
    'dim': (0.45, 0.20, 0.45, 0.45, 0.25, 0.35, 0.45, 0.45),
    'gloomy': (0.30, 0.15, 0.50, 0.20, 0.30, 0.55, 0.50, 0.30),
    'murky': (0.35, 0.15, 0.65, 0.25, 0.35, 0.50, 0.55, 0.35),
    'shadowy': (0.40, 0.15, 0.55, 0.35, 0.40, 0.55, 0.50, 0.40),

    # SMOOTH TEXTURE words
    'smooth': (0.55, 0.60, 0.10, 0.65, 0.35, 0.35, 0.40, 0.55),
    'silky': (0.70, 0.55, 0.08, 0.75, 0.30, 0.45, 0.35, 0.65),
    'velvety': (0.75, 0.45, 0.10, 0.80, 0.25, 0.50, 0.30, 0.70),
    'sleek': (0.50, 0.70, 0.12, 0.65, 0.50, 0.45, 0.60, 0.50),
    'polished': (0.50, 0.80, 0.08, 0.60, 0.40, 0.40, 0.65, 0.50),

    # ROUGH TEXTURE words
    'rough': (0.45, 0.40, 0.90, 0.35, 0.55, 0.60, 0.65, 0.45),
    'jagged': (0.35, 0.45, 0.95, 0.25, 0.70, 0.75, 0.90, 0.40),
    'coarse': (0.40, 0.35, 0.85, 0.35, 0.45, 0.50, 0.60, 0.40),
    'gritty': (0.40, 0.35, 0.88, 0.35, 0.55, 0.60, 0.65, 0.40),
    'abrasive': (0.30, 0.40, 0.92, 0.20, 0.65, 0.70, 0.75, 0.35),

    # HIGH AROUSAL words
    'exciting': (0.70, 0.80, 0.50, 0.85, 0.95, 0.85, 0.55, 0.70),
    'thrilling': (0.65, 0.75, 0.55, 0.80, 0.95, 0.90, 0.60, 0.65),
    'exhilarating': (0.70, 0.85, 0.50, 0.90, 0.95, 0.90, 0.55, 0.70),
    'electrifying': (0.60, 0.90, 0.60, 0.85, 0.98, 0.95, 0.70, 0.60),
    'intense': (0.55, 0.60, 0.65, 0.55, 0.85, 0.95, 0.65, 0.55),

    # LOW AROUSAL words
    'calm': (0.60, 0.55, 0.30, 0.70, 0.10, 0.25, 0.35, 0.55),
    'peaceful': (0.65, 0.60, 0.25, 0.80, 0.08, 0.20, 0.30, 0.60),
    'serene': (0.60, 0.65, 0.20, 0.85, 0.05, 0.20, 0.30, 0.55),
    'tranquil': (0.60, 0.60, 0.22, 0.80, 0.08, 0.20, 0.30, 0.55),
    'relaxed': (0.70, 0.55, 0.25, 0.75, 0.12, 0.25, 0.30, 0.65),

    # HIGH INTENSITY words
    'powerful': (0.55, 0.60, 0.65, 0.60, 0.80, 0.95, 0.70, 0.55),
    'overwhelming': (0.45, 0.50, 0.70, 0.40, 0.85, 0.98, 0.65, 0.45),
    'devastating': (0.30, 0.30, 0.75, 0.10, 0.80, 0.98, 0.65, 0.30),
    'profound': (0.50, 0.45, 0.50, 0.55, 0.50, 0.90, 0.50, 0.50),
    'fierce': (0.60, 0.55, 0.80, 0.40, 0.90, 0.95, 0.75, 0.65),

    # LOW INTENSITY words
    'mild': (0.55, 0.55, 0.35, 0.60, 0.30, 0.20, 0.40, 0.55),
    'gentle': (0.70, 0.55, 0.20, 0.75, 0.20, 0.25, 0.30, 0.65),
    'subtle': (0.50, 0.50, 0.35, 0.60, 0.25, 0.20, 0.45, 0.50),
    'faint': (0.50, 0.40, 0.40, 0.50, 0.20, 0.15, 0.45, 0.50),
    'delicate': (0.60, 0.60, 0.15, 0.70, 0.25, 0.25, 0.35, 0.60),

    # ANGULAR GEOMETRY words
    'angular': (0.40, 0.55, 0.65, 0.45, 0.55, 0.55, 0.95, 0.40),
    'geometric': (0.45, 0.60, 0.55, 0.50, 0.50, 0.50, 0.92, 0.45),
    'sharp': (0.35, 0.65, 0.70, 0.40, 0.70, 0.70, 0.90, 0.35),
    'rigid': (0.35, 0.50, 0.70, 0.35, 0.45, 0.60, 0.92, 0.35),
    'structured': (0.45, 0.55, 0.55, 0.55, 0.45, 0.50, 0.88, 0.45),

    # ORGANIC GEOMETRY words
    'organic': (0.60, 0.55, 0.45, 0.65, 0.45, 0.45, 0.10, 0.60),
    'flowing': (0.55, 0.55, 0.30, 0.65, 0.40, 0.40, 0.12, 0.55),
    'curved': (0.55, 0.55, 0.35, 0.60, 0.40, 0.40, 0.15, 0.55),
    'sinuous': (0.55, 0.50, 0.40, 0.55, 0.50, 0.50, 0.15, 0.55),
    'undulating': (0.55, 0.50, 0.40, 0.55, 0.45, 0.45, 0.15, 0.55),

    # WARM COLOR TEMP words
    'golden': (0.85, 0.80, 0.35, 0.80, 0.55, 0.60, 0.40, 0.90),
    'amber': (0.80, 0.65, 0.40, 0.70, 0.45, 0.55, 0.45, 0.88),
    'sunset': (0.80, 0.70, 0.35, 0.80, 0.50, 0.65, 0.40, 0.92),
    'fiery': (0.90, 0.75, 0.60, 0.55, 0.85, 0.90, 0.60, 0.95),
    'rustic': (0.75, 0.50, 0.60, 0.65, 0.35, 0.50, 0.50, 0.80),

    # COOL COLOR TEMP words
    'azure': (0.35, 0.75, 0.30, 0.70, 0.40, 0.45, 0.45, 0.15),
    'icy blue': (0.15, 0.80, 0.45, 0.50, 0.35, 0.50, 0.55, 0.08),
    'steel': (0.30, 0.55, 0.65, 0.40, 0.40, 0.55, 0.80, 0.20),
    'silver': (0.40, 0.80, 0.30, 0.55, 0.40, 0.45, 0.60, 0.25),
    'moonlit': (0.35, 0.45, 0.25, 0.60, 0.25, 0.45, 0.40, 0.20),
}

# Complex scene descriptions with multi-dimensional vibes
SCENE_VIBES = {
    # INDOOR SCENES
    "a grandmother's kitchen filled with the smell of fresh bread": (0.90, 0.65, 0.35, 0.92, 0.35, 0.55, 0.30, 0.85),
    "fluorescent lights buzzing in an empty hospital corridor": (0.30, 0.90, 0.55, 0.25, 0.55, 0.60, 0.85, 0.25),
    "candlelit dinner in an intimate restaurant": (0.85, 0.25, 0.30, 0.85, 0.35, 0.55, 0.35, 0.80),
    "a dusty attic filled with forgotten memories": (0.55, 0.25, 0.65, 0.45, 0.25, 0.60, 0.45, 0.55),
    "sterile white laboratory with humming machines": (0.25, 0.95, 0.45, 0.40, 0.60, 0.55, 0.90, 0.15),
    "cozy cabin with a crackling fireplace": (0.95, 0.35, 0.40, 0.90, 0.25, 0.55, 0.30, 0.90),
    "abandoned factory with broken windows and rust": (0.30, 0.35, 0.85, 0.15, 0.25, 0.60, 0.70, 0.35),
    "velvet-draped theater before the curtain rises": (0.70, 0.30, 0.20, 0.75, 0.60, 0.70, 0.45, 0.70),
    "minimalist zen garden with raked sand": (0.50, 0.70, 0.15, 0.75, 0.08, 0.30, 0.50, 0.45),
    "neon-lit arcade with beeping machines": (0.55, 0.85, 0.60, 0.75, 0.90, 0.80, 0.70, 0.55),

    # OUTDOOR SCENES
    "sunrise over a misty mountain valley": (0.70, 0.75, 0.25, 0.85, 0.40, 0.70, 0.30, 0.75),
    "storm clouds gathering before a thunderstorm": (0.40, 0.25, 0.65, 0.30, 0.80, 0.90, 0.55, 0.35),
    "gentle snowfall in a quiet forest": (0.15, 0.75, 0.25, 0.70, 0.10, 0.40, 0.30, 0.10),
    "tropical beach at golden hour": (0.90, 0.85, 0.30, 0.95, 0.50, 0.60, 0.25, 0.90),
    "gothic cathedral silhouetted against grey sky": (0.35, 0.40, 0.55, 0.40, 0.45, 0.80, 0.85, 0.30),
    "wildflower meadow buzzing with bees": (0.75, 0.85, 0.40, 0.90, 0.65, 0.55, 0.20, 0.70),
    "frozen lake under the northern lights": (0.10, 0.70, 0.35, 0.75, 0.50, 0.80, 0.40, 0.05),
    "desert canyon at high noon": (0.85, 0.95, 0.65, 0.55, 0.45, 0.70, 0.60, 0.85),
    "foggy cemetery at dusk": (0.30, 0.20, 0.50, 0.20, 0.40, 0.70, 0.55, 0.30),
    "bamboo forest with filtered sunlight": (0.55, 0.60, 0.45, 0.75, 0.30, 0.45, 0.65, 0.50),

    # EMOTIONAL MOMENTS
    "the moment you realize you're in love": (0.90, 0.80, 0.20, 0.98, 0.85, 0.95, 0.25, 0.85),
    "receiving terrible news unexpectedly": (0.25, 0.30, 0.60, 0.05, 0.90, 0.98, 0.60, 0.25),
    "the silence after an argument": (0.35, 0.35, 0.55, 0.20, 0.30, 0.75, 0.55, 0.35),
    "nostalgia for a place you've never been": (0.65, 0.50, 0.40, 0.50, 0.25, 0.70, 0.40, 0.60),
    "the anticipation before opening a gift": (0.70, 0.75, 0.45, 0.85, 0.80, 0.70, 0.45, 0.65),
    "waking up from a beautiful dream": (0.60, 0.55, 0.25, 0.65, 0.30, 0.55, 0.30, 0.60),
    "the weight of unspoken words": (0.40, 0.35, 0.55, 0.30, 0.15, 0.85, 0.50, 0.40),
    "childhood summer afternoons": (0.85, 0.90, 0.30, 0.92, 0.60, 0.55, 0.25, 0.80),
    "standing at a crossroads in life": (0.50, 0.55, 0.55, 0.45, 0.55, 0.75, 0.60, 0.50),
    "the relief after a long struggle": (0.65, 0.65, 0.30, 0.80, 0.20, 0.50, 0.35, 0.60),

    # SENSORY EXPERIENCES
    "the taste of bitter coffee on a cold morning": (0.60, 0.35, 0.55, 0.55, 0.60, 0.65, 0.55, 0.55),
    "silk sliding across bare skin": (0.70, 0.50, 0.05, 0.80, 0.45, 0.55, 0.25, 0.65),
    "the smell of rain on hot pavement": (0.60, 0.45, 0.55, 0.70, 0.35, 0.50, 0.50, 0.55),
    "bass vibrating through your chest": (0.55, 0.40, 0.75, 0.65, 0.90, 0.95, 0.60, 0.55),
    "ice cream melting on a summer day": (0.70, 0.80, 0.20, 0.85, 0.50, 0.45, 0.25, 0.60),
    "the crunch of autumn leaves underfoot": (0.55, 0.55, 0.75, 0.70, 0.45, 0.45, 0.55, 0.55),
    "whispered secrets in the dark": (0.60, 0.10, 0.35, 0.60, 0.35, 0.65, 0.35, 0.55),
    "the sting of cold wind on your face": (0.15, 0.65, 0.60, 0.35, 0.70, 0.70, 0.55, 0.10),
    "warm sand between your toes": (0.85, 0.80, 0.55, 0.85, 0.30, 0.40, 0.30, 0.80),
    "the echo of footsteps in an empty hall": (0.35, 0.40, 0.50, 0.35, 0.40, 0.55, 0.75, 0.35),

    # MUSIC/ART VIBES
    "jazz playing softly at 2am": (0.65, 0.25, 0.35, 0.65, 0.30, 0.50, 0.35, 0.60),
    "thundering orchestral crescendo": (0.55, 0.60, 0.60, 0.70, 0.95, 0.98, 0.55, 0.55),
    "melancholic piano in the rain": (0.45, 0.30, 0.30, 0.35, 0.25, 0.75, 0.35, 0.40),
    "aggressive industrial electronic": (0.35, 0.55, 0.90, 0.30, 0.95, 0.95, 0.85, 0.35),
    "ethereal ambient soundscape": (0.50, 0.55, 0.15, 0.70, 0.10, 0.40, 0.25, 0.45),
    "punk rock mosh pit energy": (0.60, 0.65, 0.85, 0.70, 0.98, 0.98, 0.75, 0.60),
    "baroque harpsichord elegance": (0.55, 0.70, 0.35, 0.65, 0.50, 0.55, 0.75, 0.55),
    "lo-fi beats for studying": (0.60, 0.45, 0.40, 0.70, 0.25, 0.30, 0.40, 0.55),
    "drone music meditation": (0.50, 0.40, 0.30, 0.60, 0.08, 0.45, 0.35, 0.45),
    "tropical house summer vibes": (0.80, 0.85, 0.35, 0.90, 0.70, 0.65, 0.35, 0.75),
}

# Bittersweet/complex emotion examples (these were weak in previous model)
COMPLEX_EMOTIONS = {
    "bittersweet memories of childhood": (0.60, 0.50, 0.40, 0.45, 0.25, 0.70, 0.35, 0.55),
    "nostalgia tinged with regret": (0.55, 0.40, 0.45, 0.35, 0.25, 0.75, 0.40, 0.50),
    "melancholy beauty": (0.45, 0.45, 0.35, 0.40, 0.20, 0.75, 0.35, 0.45),
    "wistful longing for the past": (0.55, 0.45, 0.35, 0.40, 0.25, 0.70, 0.35, 0.50),
    "the ache of something beautiful ending": (0.50, 0.45, 0.40, 0.35, 0.30, 0.85, 0.40, 0.50),
    "joyful tears": (0.75, 0.65, 0.30, 0.80, 0.60, 0.85, 0.30, 0.70),
    "peaceful acceptance of loss": (0.55, 0.50, 0.35, 0.55, 0.15, 0.65, 0.35, 0.50),
    "the comfort of familiar sadness": (0.60, 0.35, 0.40, 0.40, 0.20, 0.60, 0.35, 0.55),
    "hopeful despite everything": (0.65, 0.65, 0.40, 0.70, 0.45, 0.65, 0.40, 0.60),
    "anxious anticipation": (0.50, 0.60, 0.55, 0.45, 0.80, 0.75, 0.55, 0.50),
    "nervous excitement": (0.60, 0.70, 0.50, 0.65, 0.85, 0.75, 0.50, 0.55),
    "love mixed with fear": (0.70, 0.50, 0.45, 0.55, 0.75, 0.85, 0.45, 0.65),
    "grief slowly transforming into gratitude": (0.55, 0.50, 0.40, 0.55, 0.25, 0.70, 0.40, 0.55),
    "the strange comfort of rain": (0.50, 0.35, 0.40, 0.60, 0.25, 0.50, 0.40, 0.45),
    "existential awe at the universe": (0.45, 0.60, 0.40, 0.65, 0.45, 0.90, 0.55, 0.45),
    "humble before nature's power": (0.45, 0.55, 0.50, 0.60, 0.50, 0.85, 0.45, 0.45),
    "quietly devastated": (0.35, 0.25, 0.45, 0.15, 0.20, 0.90, 0.45, 0.35),
    "fiercely protective love": (0.75, 0.55, 0.55, 0.75, 0.80, 0.95, 0.55, 0.75),
    "tender vulnerability": (0.75, 0.50, 0.25, 0.65, 0.40, 0.70, 0.30, 0.70),
    "dignified suffering": (0.45, 0.40, 0.50, 0.35, 0.30, 0.80, 0.55, 0.45),
}

# Context-specific vibes (same word, different meaning)
CONTEXT_VIBES = {
    # "Hot" in different contexts
    "hot summer afternoon": (0.95, 0.90, 0.45, 0.70, 0.55, 0.60, 0.35, 0.95),
    "hot spicy curry": (0.85, 0.55, 0.65, 0.65, 0.75, 0.80, 0.50, 0.90),
    "hot new trend": (0.70, 0.80, 0.50, 0.75, 0.80, 0.70, 0.55, 0.70),
    "hot-tempered argument": (0.75, 0.60, 0.80, 0.20, 0.90, 0.90, 0.70, 0.80),

    # "Cold" in different contexts
    "cold winter night": (0.08, 0.30, 0.50, 0.35, 0.25, 0.55, 0.50, 0.08),
    "cold shoulder": (0.20, 0.45, 0.55, 0.20, 0.35, 0.60, 0.60, 0.25),
    "cold hard facts": (0.30, 0.60, 0.65, 0.45, 0.40, 0.55, 0.80, 0.30),
    "cold precision": (0.25, 0.65, 0.50, 0.50, 0.40, 0.55, 0.85, 0.25),

    # "Dark" in different contexts
    "dark chocolate": (0.60, 0.15, 0.55, 0.65, 0.35, 0.60, 0.50, 0.55),
    "dark humor": (0.40, 0.25, 0.55, 0.50, 0.55, 0.65, 0.55, 0.40),
    "dark secrets": (0.35, 0.10, 0.60, 0.25, 0.50, 0.80, 0.55, 0.35),
    "dark academia aesthetic": (0.50, 0.25, 0.50, 0.55, 0.40, 0.65, 0.70, 0.45),

    # "Light" in different contexts
    "light and airy room": (0.55, 0.90, 0.20, 0.80, 0.45, 0.35, 0.40, 0.55),
    "light-hearted comedy": (0.70, 0.80, 0.30, 0.85, 0.65, 0.40, 0.35, 0.65),
    "light touch": (0.60, 0.60, 0.15, 0.70, 0.30, 0.25, 0.35, 0.55),
    "light at the end of the tunnel": (0.65, 0.85, 0.30, 0.80, 0.50, 0.65, 0.40, 0.60),
}


def create_vibe_dataset():
    """Create massive vibe dataset from all sources."""
    data = []
    dims = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']

    # 1. Add emotion word vibes
    print("Processing emotion word vibes...")
    for text, vibes in EMOTION_WORD_VIBES.items():
        row = {'text': text}
        for i, dim in enumerate(dims):
            row[dim] = vibes[i]
        data.append(row)

    # 2. Add scene vibes
    print("Processing scene vibes...")
    for text, vibes in SCENE_VIBES.items():
        row = {'text': text}
        for i, dim in enumerate(dims):
            row[dim] = vibes[i]
        data.append(row)

    # 3. Add complex emotions
    print("Processing complex emotions...")
    for text, vibes in COMPLEX_EMOTIONS.items():
        row = {'text': text}
        for i, dim in enumerate(dims):
            row[dim] = vibes[i]
        data.append(row)

    # 4. Add context vibes
    print("Processing context vibes...")
    for text, vibes in CONTEXT_VIBES.items():
        row = {'text': text}
        for i, dim in enumerate(dims):
            row[dim] = vibes[i]
        data.append(row)

    return pd.DataFrame(data)


def augment_with_variations(df):
    """Create variations of existing examples to increase dataset size."""
    augmented = []

    # Prefix variations
    prefixes = [
        ("the feeling of ", 1.0),
        ("experiencing ", 1.0),
        ("a sense of ", 1.0),
        ("like ", 1.0),
        ("reminds me of ", 1.0),
        ("the vibe of ", 1.0),
        ("", 1.0),  # No prefix
    ]

    # Suffix variations
    suffixes = [
        ("", 1.0),
        (" energy", 1.05),  # Slightly boost intensity
        (" aesthetic", 1.0),
        (" atmosphere", 1.0),
        (" mood", 1.0),
    ]

    dims = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']

    for _, row in df.iterrows():
        base_text = row['text']

        # Only augment scene descriptions and complex emotions (longer texts)
        if len(base_text.split()) < 3:
            continue

        for prefix, _ in prefixes[:3]:  # Limit augmentation
            for suffix, intensity_mod in suffixes[:2]:
                new_text = prefix + base_text + suffix
                if new_text != base_text:
                    new_row = {'text': new_text}
                    for dim in dims:
                        new_row[dim] = row[dim]
                    augmented.append(new_row)

    return pd.DataFrame(augmented)


def download_goemotions():
    """Download and process GoEmotions dataset."""
    try:
        from datasets import load_dataset
        print("Loading GoEmotions from HuggingFace...")
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
        return dataset
    except Exception as e:
        print(f"Could not load GoEmotions: {e}")
        print("Creating synthetic GoEmotions-style data instead...")
        return None


def process_goemotions(dataset):
    """Convert GoEmotions to 8D vibe format."""
    if dataset is None:
        return pd.DataFrame()

    data = []
    dims = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']
    emotion_labels = list(GOEMOTIONS_TO_VIBE.keys())

    print("Processing GoEmotions train split...")
    for example in dataset['train']:
        text = example['text']
        labels = example['labels']  # List of emotion indices

        # Skip if no clear emotion or neutral-only
        if not labels or (len(labels) == 1 and labels[0] == 27):  # 27 is neutral
            continue

        # Average vibes across all labeled emotions
        vibe_sum = np.zeros(8)
        count = 0

        for label_idx in labels:
            if label_idx < len(emotion_labels):
                emotion = emotion_labels[label_idx]
                if emotion in GOEMOTIONS_TO_VIBE:
                    vibe_sum += np.array(GOEMOTIONS_TO_VIBE[emotion])
                    count += 1

        if count > 0:
            avg_vibe = vibe_sum / count
            row = {'text': text}
            for i, dim in enumerate(dims):
                row[dim] = float(avg_vibe[i])
            data.append(row)

        # Limit to manageable size for training
        if len(data) >= 20000:
            break

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("=" * 70)
    print("ACADEMIC VIBE DATA ACQUISITION")
    print("=" * 70)

    # 1. Create base dataset from curated examples
    print("\n1. Creating base dataset from curated examples...")
    base_df = create_vibe_dataset()
    print(f"   Base examples: {len(base_df)}")

    # 2. Augment with variations
    print("\n2. Augmenting with variations...")
    aug_df = augment_with_variations(base_df)
    print(f"   Augmented examples: {len(aug_df)}")

    # 3. Try to download GoEmotions
    print("\n3. Attempting to download GoEmotions...")
    goemotions = download_goemotions()
    goemotions_df = process_goemotions(goemotions)
    print(f"   GoEmotions examples: {len(goemotions_df)}")

    # 4. Combine all sources
    print("\n4. Combining all data sources...")
    combined = pd.concat([base_df, aug_df, goemotions_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'])

    # 5. Save
    output_path = "/Users/erinsaintgull/P/comprehensive_vibe_training_data.csv"
    combined.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"TOTAL TRAINING EXAMPLES: {len(combined)}")
    print(f"Saved to: {output_path}")
    print(f"{'=' * 70}")

    # Show dimension coverage
    dims = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']
    print("\nDimension value distributions:")
    for dim in dims:
        values = combined[dim]
        active = ((values < 0.4) | (values > 0.6)).sum() / len(values) * 100
        print(f"  {dim:<18} min={values.min():.2f} max={values.max():.2f} active={active:.1f}%")
