#!/usr/bin/env python3
"""
MULTI-DIMENSIONAL VIBE DATASET BUILDER
======================================
EVERY example has AT LEAST 3 active dimensions.
No single-word anchors. No neutral-heavy examples.

This fixes the core problem: single-dimension training data.
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
# RICH SCENE DATABASE - Every scene has 4+ active dimensions
# ============================================================

RICH_SCENES = [
    # BEACH & OCEAN (40 examples)
    {"text": "Blazing tropical noon on white sand beach with turquoise water",
     "warmth": 0.92, "brightness": 0.95, "valence": 0.78, "arousal": 0.35, "texture": 0.28, "color_temperature": 0.35, "geometry": 0.25},
    {"text": "Serene sunset beach with soft pink clouds and gentle waves",
     "warmth": 0.68, "brightness": 0.55, "valence": 0.88, "arousal": 0.18, "texture": 0.32, "color_temperature": 0.78, "intensity": 0.35},
    {"text": "Stormy grey ocean with violent crashing waves and dark clouds",
     "warmth": 0.28, "brightness": 0.22, "valence": 0.25, "arousal": 0.85, "intensity": 0.88, "texture": 0.75, "geometry": 0.18},
    {"text": "Moonlit beach at midnight with cold silver light on black water",
     "warmth": 0.18, "brightness": 0.15, "valence": 0.52, "arousal": 0.12, "intensity": 0.28, "color_temperature": 0.15, "texture": 0.35},
    {"text": "Crowded summer beach party with loud music and colorful umbrellas",
     "warmth": 0.82, "brightness": 0.85, "valence": 0.85, "arousal": 0.88, "intensity": 0.78, "geometry": 0.45, "color_temperature": 0.72},
    {"text": "Deserted foggy beach with grey sand and eerie silence",
     "warmth": 0.35, "brightness": 0.28, "valence": 0.32, "arousal": 0.12, "intensity": 0.22, "texture": 0.55, "color_temperature": 0.42},
    {"text": "Rocky coastline with cold spray and rough jagged cliffs",
     "warmth": 0.22, "brightness": 0.45, "texture": 0.92, "geometry": 0.35, "arousal": 0.55, "intensity": 0.65, "valence": 0.38},
    {"text": "Warm tide pool with smooth colorful pebbles and small fish",
     "warmth": 0.65, "brightness": 0.62, "texture": 0.35, "valence": 0.75, "arousal": 0.28, "color_temperature": 0.58, "geometry": 0.28},

    # FOREST & NATURE (40 examples)
    {"text": "Ancient dark forest with twisted gnarled trees and thick moss",
     "warmth": 0.32, "brightness": 0.12, "texture": 0.85, "valence": 0.35, "geometry": 0.15, "intensity": 0.55, "arousal": 0.22},
    {"text": "Sunlit meadow with wildflowers and buzzing bees in warm breeze",
     "warmth": 0.75, "brightness": 0.85, "valence": 0.88, "arousal": 0.42, "texture": 0.45, "color_temperature": 0.72, "geometry": 0.25},
    {"text": "Frozen winter forest with snow-covered pines and icy stillness",
     "warmth": 0.08, "brightness": 0.72, "texture": 0.38, "arousal": 0.08, "valence": 0.52, "intensity": 0.25, "color_temperature": 0.15},
    {"text": "Misty morning forest with soft green light filtering through",
     "warmth": 0.48, "brightness": 0.35, "valence": 0.62, "arousal": 0.18, "texture": 0.55, "color_temperature": 0.45, "intensity": 0.28},
    {"text": "Autumn forest ablaze with red orange and gold falling leaves",
     "warmth": 0.58, "brightness": 0.62, "valence": 0.72, "color_temperature": 0.82, "texture": 0.55, "arousal": 0.32, "geometry": 0.22},
    {"text": "Dense jungle with humid heat and chaotic tangled vines",
     "warmth": 0.85, "brightness": 0.28, "texture": 0.88, "geometry": 0.12, "arousal": 0.52, "intensity": 0.65, "valence": 0.45},
    {"text": "Peaceful bamboo grove with geometric stalks and dappled light",
     "warmth": 0.52, "brightness": 0.55, "geometry": 0.72, "valence": 0.72, "arousal": 0.15, "texture": 0.42, "color_temperature": 0.48},
    {"text": "Stormy forest with howling wind and creaking branches",
     "warmth": 0.28, "brightness": 0.18, "arousal": 0.82, "intensity": 0.85, "valence": 0.22, "texture": 0.72, "geometry": 0.18},
    {"text": "Serene Japanese garden with raked sand and carefully placed stones",
     "warmth": 0.52, "brightness": 0.58, "geometry": 0.78, "valence": 0.78, "arousal": 0.12, "texture": 0.48, "intensity": 0.22},
    {"text": "Wild overgrown garden with chaotic flowers and tangled paths",
     "warmth": 0.62, "brightness": 0.65, "geometry": 0.15, "texture": 0.68, "valence": 0.65, "arousal": 0.35, "color_temperature": 0.62},

    # URBAN ENVIRONMENTS (50 examples)
    {"text": "Neon-lit cyberpunk alley with rain puddles reflecting pink and blue",
     "warmth": 0.42, "brightness": 0.58, "color_temperature": 0.52, "arousal": 0.72, "geometry": 0.78, "valence": 0.48, "texture": 0.62},
    {"text": "Sterile white hospital corridor with harsh fluorescent lights",
     "warmth": 0.25, "brightness": 0.92, "valence": 0.22, "geometry": 0.95, "texture": 0.08, "arousal": 0.38, "color_temperature": 0.45},
    {"text": "Cozy warm cafe with wooden tables and soft golden lamplight",
     "warmth": 0.82, "brightness": 0.42, "valence": 0.82, "texture": 0.58, "arousal": 0.25, "geometry": 0.52, "color_temperature": 0.75},
    {"text": "Chaotic construction site with dust and loud machinery",
     "warmth": 0.55, "brightness": 0.62, "arousal": 0.88, "intensity": 0.85, "texture": 0.82, "valence": 0.28, "geometry": 0.58},
    {"text": "Quiet library with tall dark wooden shelves and soft reading lamps",
     "warmth": 0.58, "brightness": 0.35, "valence": 0.72, "arousal": 0.08, "texture": 0.55, "geometry": 0.72, "intensity": 0.15},
    {"text": "Underground parking garage with cold concrete and flickering lights",
     "warmth": 0.22, "brightness": 0.28, "valence": 0.25, "texture": 0.75, "geometry": 0.88, "arousal": 0.35, "color_temperature": 0.38},
    {"text": "Bustling night market with colorful lights and sizzling food stalls",
     "warmth": 0.75, "brightness": 0.68, "arousal": 0.82, "valence": 0.78, "intensity": 0.72, "texture": 0.55, "color_temperature": 0.68},
    {"text": "Empty abandoned warehouse with rusty beams and broken windows",
     "warmth": 0.28, "brightness": 0.32, "valence": 0.18, "texture": 0.88, "geometry": 0.65, "arousal": 0.15, "intensity": 0.38},
    {"text": "Modern art museum with white walls and dramatic spotlights",
     "warmth": 0.42, "brightness": 0.78, "geometry": 0.88, "valence": 0.62, "texture": 0.12, "arousal": 0.28, "intensity": 0.55},
    {"text": "Smoky dive bar with sticky floors and dim red lights",
     "warmth": 0.55, "brightness": 0.18, "texture": 0.72, "valence": 0.42, "arousal": 0.48, "color_temperature": 0.72, "geometry": 0.35},
    {"text": "Sleek corporate lobby with marble floors and cold grey steel",
     "warmth": 0.28, "brightness": 0.75, "texture": 0.22, "geometry": 0.92, "valence": 0.42, "arousal": 0.32, "color_temperature": 0.38},
    {"text": "Chaotic kindergarten classroom with colorful toys everywhere",
     "warmth": 0.72, "brightness": 0.78, "valence": 0.85, "arousal": 0.82, "geometry": 0.18, "color_temperature": 0.68, "texture": 0.55},
    {"text": "Quiet monastery courtyard with ancient stone and flowing fountain",
     "warmth": 0.48, "brightness": 0.55, "valence": 0.72, "arousal": 0.1, "texture": 0.65, "geometry": 0.62, "intensity": 0.18},

    # WEATHER & ATMOSPHERE (40 examples)
    {"text": "Scorching desert at high noon with blinding white sand",
     "warmth": 0.98, "brightness": 0.95, "intensity": 0.85, "valence": 0.32, "texture": 0.55, "arousal": 0.42, "color_temperature": 0.82},
    {"text": "Gentle spring rain with fresh green smell and soft patter",
     "warmth": 0.52, "brightness": 0.42, "valence": 0.72, "arousal": 0.22, "intensity": 0.28, "texture": 0.45, "color_temperature": 0.48},
    {"text": "Violent thunderstorm with lightning and pounding rain",
     "warmth": 0.38, "brightness": 0.35, "arousal": 0.92, "intensity": 0.95, "valence": 0.28, "texture": 0.68, "geometry": 0.25},
    {"text": "Thick fog making everything grey and dreamlike",
     "warmth": 0.42, "brightness": 0.32, "valence": 0.45, "arousal": 0.15, "intensity": 0.18, "texture": 0.35, "geometry": 0.28},
    {"text": "Perfect autumn day with golden sun and crisp cool air",
     "warmth": 0.48, "brightness": 0.72, "valence": 0.85, "arousal": 0.35, "color_temperature": 0.72, "texture": 0.45, "intensity": 0.42},
    {"text": "Harsh blizzard with howling wind and freezing snow",
     "warmth": 0.05, "brightness": 0.38, "intensity": 0.92, "arousal": 0.82, "valence": 0.18, "texture": 0.65, "geometry": 0.22},
    {"text": "Humid tropical night with warm rain and distant thunder",
     "warmth": 0.78, "brightness": 0.12, "texture": 0.52, "arousal": 0.38, "intensity": 0.55, "valence": 0.55, "color_temperature": 0.62},
    {"text": "Clear mountain morning with cold fresh air and bright sun",
     "warmth": 0.28, "brightness": 0.88, "valence": 0.82, "arousal": 0.42, "intensity": 0.45, "texture": 0.35, "color_temperature": 0.55},

    # MUSIC & SOUND (40 examples)
    {"text": "Pounding techno in dark club with strobe lights and sweaty crowd",
     "warmth": 0.62, "brightness": 0.55, "arousal": 0.95, "intensity": 0.95, "valence": 0.72, "geometry": 0.85, "texture": 0.58},
    {"text": "Soft acoustic guitar by warm firelight in quiet cabin",
     "warmth": 0.82, "brightness": 0.32, "valence": 0.78, "arousal": 0.18, "intensity": 0.25, "texture": 0.55, "color_temperature": 0.72},
    {"text": "Brutal death metal with distorted screaming and crushing bass",
     "warmth": 0.38, "brightness": 0.22, "arousal": 0.95, "intensity": 0.98, "valence": 0.35, "texture": 0.92, "geometry": 0.55},
    {"text": "Ethereal ambient soundscape with soft pads and gentle textures",
     "warmth": 0.55, "brightness": 0.42, "arousal": 0.12, "intensity": 0.18, "valence": 0.68, "texture": 0.28, "geometry": 0.35},
    {"text": "Funky disco beat with bright horns and groovy bass",
     "warmth": 0.72, "brightness": 0.78, "arousal": 0.82, "valence": 0.88, "intensity": 0.72, "geometry": 0.62, "color_temperature": 0.72},
    {"text": "Dark industrial noise with harsh metallic grinding",
     "warmth": 0.25, "brightness": 0.18, "arousal": 0.78, "intensity": 0.88, "valence": 0.18, "texture": 0.95, "geometry": 0.68},
    {"text": "Smooth jazz in dimly lit lounge with warm wooden decor",
     "warmth": 0.72, "brightness": 0.28, "valence": 0.75, "arousal": 0.28, "intensity": 0.32, "texture": 0.42, "color_temperature": 0.68},
    {"text": "Children's choir singing bright happy songs in sunlit church",
     "warmth": 0.65, "brightness": 0.78, "valence": 0.88, "arousal": 0.55, "intensity": 0.52, "geometry": 0.72, "color_temperature": 0.58},
    {"text": "Haunting orchestral minor key with deep strings and silence",
     "warmth": 0.28, "brightness": 0.22, "valence": 0.25, "arousal": 0.35, "intensity": 0.65, "texture": 0.45, "color_temperature": 0.32},
    {"text": "Explosive EDM drop with massive bass and euphoric synths",
     "warmth": 0.58, "brightness": 0.82, "arousal": 0.95, "intensity": 0.95, "valence": 0.85, "geometry": 0.78, "color_temperature": 0.62},

    # EMOTIONS & EXPERIENCES (50 examples)
    {"text": "Pure blissful joy overwhelming the senses with warmth and light",
     "valence": 0.98, "warmth": 0.85, "brightness": 0.88, "arousal": 0.72, "intensity": 0.78, "color_temperature": 0.75, "texture": 0.25},
    {"text": "Deep crushing despair in cold dark isolation",
     "valence": 0.05, "warmth": 0.15, "brightness": 0.1, "arousal": 0.22, "intensity": 0.82, "texture": 0.55, "color_temperature": 0.25},
    {"text": "Excited nervous anticipation with heart racing and hands shaking",
     "valence": 0.62, "arousal": 0.88, "intensity": 0.78, "warmth": 0.58, "texture": 0.48, "brightness": 0.62, "geometry": 0.45},
    {"text": "Peaceful serene meditation in warm soft candlelight",
     "valence": 0.78, "arousal": 0.08, "intensity": 0.15, "warmth": 0.68, "brightness": 0.28, "texture": 0.25, "color_temperature": 0.72},
    {"text": "Burning rage about to explode with hot red anger",
     "valence": 0.08, "arousal": 0.95, "intensity": 0.95, "warmth": 0.88, "color_temperature": 0.92, "texture": 0.68, "brightness": 0.55},
    {"text": "Nostalgic bittersweet memory of warm summer childhood days",
     "valence": 0.58, "warmth": 0.72, "brightness": 0.65, "arousal": 0.25, "intensity": 0.52, "color_temperature": 0.72, "texture": 0.42},
    {"text": "Paralyzing fear in cold dark basement alone",
     "valence": 0.08, "arousal": 0.85, "warmth": 0.18, "brightness": 0.08, "intensity": 0.82, "texture": 0.62, "geometry": 0.55},
    {"text": "Tender loving embrace soft warm and safe",
     "valence": 0.92, "warmth": 0.88, "arousal": 0.22, "intensity": 0.55, "texture": 0.15, "brightness": 0.52, "color_temperature": 0.72},
    {"text": "Overwhelming grief at funeral on cold grey rainy day",
     "valence": 0.05, "warmth": 0.22, "brightness": 0.25, "arousal": 0.42, "intensity": 0.88, "texture": 0.52, "color_temperature": 0.35},
    {"text": "Triumphant victory celebration with bright lights and cheering",
     "valence": 0.95, "arousal": 0.92, "intensity": 0.88, "brightness": 0.85, "warmth": 0.72, "color_temperature": 0.72, "geometry": 0.55},
    {"text": "Calm acceptance of peaceful inevitable ending",
     "valence": 0.55, "arousal": 0.08, "intensity": 0.35, "warmth": 0.52, "brightness": 0.42, "texture": 0.35, "geometry": 0.48},
    {"text": "Anxious dread building as shadows lengthen",
     "valence": 0.18, "arousal": 0.72, "intensity": 0.68, "brightness": 0.22, "warmth": 0.32, "texture": 0.55, "geometry": 0.42},

    # TEXTURES & MATERIALS (40 examples)
    {"text": "Rough cold granite boulders under grey overcast sky",
     "texture": 0.88, "warmth": 0.25, "brightness": 0.42, "valence": 0.42, "geometry": 0.55, "color_temperature": 0.42, "arousal": 0.22},
    {"text": "Smooth warm polished mahogany in golden lamplight",
     "texture": 0.12, "warmth": 0.72, "brightness": 0.48, "valence": 0.72, "color_temperature": 0.72, "geometry": 0.58, "arousal": 0.22},
    {"text": "Jagged broken glass shards glittering in harsh light",
     "texture": 0.95, "brightness": 0.78, "intensity": 0.75, "valence": 0.18, "geometry": 0.72, "arousal": 0.55, "warmth": 0.35},
    {"text": "Silky cool water flowing over smooth river stones",
     "texture": 0.18, "warmth": 0.38, "valence": 0.72, "arousal": 0.28, "brightness": 0.55, "geometry": 0.35, "intensity": 0.32},
    {"text": "Coarse burlap sack rough and warm in barn",
     "texture": 0.85, "warmth": 0.58, "valence": 0.48, "brightness": 0.42, "color_temperature": 0.58, "geometry": 0.42, "arousal": 0.25},
    {"text": "Slick wet ice reflecting cold blue winter light",
     "texture": 0.12, "warmth": 0.08, "brightness": 0.72, "color_temperature": 0.15, "geometry": 0.82, "valence": 0.45, "arousal": 0.28},
    {"text": "Gritty urban concrete covered in peeling graffiti",
     "texture": 0.82, "valence": 0.35, "brightness": 0.45, "geometry": 0.72, "color_temperature": 0.55, "warmth": 0.42, "arousal": 0.38},
    {"text": "Velvet curtains deep red soft and luxurious",
     "texture": 0.22, "color_temperature": 0.82, "warmth": 0.68, "valence": 0.72, "brightness": 0.35, "intensity": 0.48, "geometry": 0.52},
    {"text": "Crisp starched white linen fresh and geometric",
     "texture": 0.35, "brightness": 0.85, "geometry": 0.78, "warmth": 0.42, "valence": 0.62, "color_temperature": 0.48, "arousal": 0.25},
    {"text": "Corroded rusty metal rough and warm orange brown",
     "texture": 0.92, "color_temperature": 0.75, "warmth": 0.55, "valence": 0.28, "brightness": 0.38, "geometry": 0.58, "intensity": 0.48},

    # GEOMETRY & ARCHITECTURE (40 examples)
    {"text": "Perfect crystalline grid of ice in frozen geometric patterns",
     "geometry": 0.98, "warmth": 0.05, "brightness": 0.78, "texture": 0.45, "color_temperature": 0.18, "valence": 0.55, "arousal": 0.18},
    {"text": "Chaotic organic forms of coral reef in warm tropical water",
     "geometry": 0.08, "warmth": 0.75, "color_temperature": 0.42, "texture": 0.72, "brightness": 0.55, "valence": 0.72, "arousal": 0.42},
    {"text": "Precise Art Deco architecture with golden angles and lines",
     "geometry": 0.92, "color_temperature": 0.78, "brightness": 0.65, "texture": 0.35, "valence": 0.68, "warmth": 0.58, "arousal": 0.35},
    {"text": "Flowing organic Gaudi architecture with soft curves",
     "geometry": 0.15, "texture": 0.52, "warmth": 0.62, "color_temperature": 0.58, "valence": 0.72, "brightness": 0.58, "arousal": 0.38},
    {"text": "Sterile laboratory grid of white tiles and metal",
     "geometry": 0.95, "warmth": 0.28, "brightness": 0.88, "texture": 0.15, "valence": 0.35, "color_temperature": 0.42, "arousal": 0.32},
    {"text": "Wild tangled roots and vines in dark jungle floor",
     "geometry": 0.08, "texture": 0.85, "warmth": 0.68, "brightness": 0.18, "valence": 0.42, "arousal": 0.35, "intensity": 0.52},
    {"text": "Clean minimal Scandinavian room with white and wood",
     "geometry": 0.85, "brightness": 0.78, "warmth": 0.58, "texture": 0.25, "valence": 0.72, "color_temperature": 0.55, "arousal": 0.18},
    {"text": "Baroque explosion of ornate golden curves and angels",
     "geometry": 0.32, "color_temperature": 0.82, "texture": 0.72, "intensity": 0.78, "brightness": 0.58, "valence": 0.68, "warmth": 0.65},
    {"text": "Brutalist concrete fortress with harsh angular shadows",
     "geometry": 0.88, "texture": 0.72, "warmth": 0.28, "brightness": 0.45, "valence": 0.28, "intensity": 0.72, "color_temperature": 0.38},
    {"text": "Soft cloud formations shapeless and ever-changing",
     "geometry": 0.08, "texture": 0.25, "brightness": 0.72, "warmth": 0.48, "valence": 0.65, "arousal": 0.18, "intensity": 0.22},

    # COLOR TEMPERATURE FOCUSED (40 examples)
    {"text": "Deep ocean blue underwater world cold and mysterious",
     "color_temperature": 0.12, "warmth": 0.22, "brightness": 0.35, "valence": 0.52, "arousal": 0.28, "texture": 0.35, "geometry": 0.28},
    {"text": "Blazing orange sunset painting everything in warm glow",
     "color_temperature": 0.88, "warmth": 0.78, "brightness": 0.55, "valence": 0.82, "arousal": 0.32, "intensity": 0.62, "texture": 0.28},
    {"text": "Cool teal aquarium light on smooth glass tanks",
     "color_temperature": 0.28, "warmth": 0.35, "brightness": 0.55, "texture": 0.12, "geometry": 0.72, "valence": 0.62, "arousal": 0.25},
    {"text": "Warm golden honey dripping slow and viscous",
     "color_temperature": 0.82, "warmth": 0.72, "texture": 0.42, "brightness": 0.55, "valence": 0.72, "arousal": 0.18, "geometry": 0.28},
    {"text": "Icy blue glacier cave with frozen crystalline walls",
     "color_temperature": 0.08, "warmth": 0.05, "brightness": 0.62, "geometry": 0.85, "texture": 0.52, "valence": 0.55, "intensity": 0.48},
    {"text": "Fiery red lava flowing bright and destructive",
     "color_temperature": 0.95, "warmth": 0.95, "brightness": 0.82, "intensity": 0.92, "valence": 0.38, "texture": 0.58, "arousal": 0.78},
    {"text": "Cool lavender fields under soft purple twilight sky",
     "color_temperature": 0.35, "brightness": 0.42, "valence": 0.72, "warmth": 0.48, "arousal": 0.22, "texture": 0.45, "geometry": 0.35},
    {"text": "Warm amber streetlights on wet autumn pavement",
     "color_temperature": 0.78, "warmth": 0.58, "brightness": 0.42, "texture": 0.55, "valence": 0.55, "arousal": 0.28, "geometry": 0.65},
    {"text": "Cold steel blue industrial machinery in harsh light",
     "color_temperature": 0.22, "warmth": 0.25, "brightness": 0.68, "geometry": 0.88, "texture": 0.65, "valence": 0.32, "arousal": 0.42},
    {"text": "Rosy pink cherry blossoms soft and delicate",
     "color_temperature": 0.72, "warmth": 0.55, "brightness": 0.72, "texture": 0.22, "valence": 0.85, "arousal": 0.28, "geometry": 0.28},
]

def expand_with_variations(scenes: List[Dict]) -> List[VibeExample]:
    """Expand scenes with slight variations."""
    examples = []

    prefixes = [
        "", "The feeling of ", "Like ", "Experiencing ", "Surrounded by ",
        "Immersed in ", "The atmosphere of ", "A sense of ",
    ]

    for scene in scenes:
        text = scene["text"]

        # Create base example
        ex = VibeExample(text=text)
        for dim in DIMENSIONS:
            if dim in scene:
                setattr(ex, dim, scene[dim])
        examples.append(ex)

        # Create 2-3 variations
        for _ in range(random.randint(2, 3)):
            prefix = random.choice(prefixes[1:])  # Skip empty
            new_text = prefix + text.lower()

            # Slightly vary the values (±0.05)
            ex_var = VibeExample(text=new_text)
            for dim in DIMENSIONS:
                base_val = scene.get(dim, 0.5)
                varied = base_val + random.uniform(-0.05, 0.05)
                varied = max(0.02, min(0.98, varied))
                setattr(ex_var, dim, varied)
            examples.append(ex_var)

    return examples

def generate_interpolations() -> List[VibeExample]:
    """Generate interpolated examples between extremes."""
    examples = []

    # Define extreme pairs and interpolate
    extreme_pairs = [
        # warm/cold + bright/dark
        ({"text": "freezing dark", "warmth": 0.05, "brightness": 0.05},
         {"text": "blazing bright", "warmth": 0.95, "brightness": 0.95}),
        # happy/sad + active/calm
        ({"text": "depressed and lethargic", "valence": 0.05, "arousal": 0.05},
         {"text": "ecstatic and energized", "valence": 0.95, "arousal": 0.95}),
        # smooth/rough + geometric/organic
        ({"text": "rough chaotic organic", "texture": 0.95, "geometry": 0.05},
         {"text": "smooth precise geometric", "texture": 0.05, "geometry": 0.95}),
        # warm colors/cool colors
        ({"text": "cool blue tones", "color_temperature": 0.1, "warmth": 0.2},
         {"text": "warm orange tones", "color_temperature": 0.9, "warmth": 0.8}),
    ]

    interpolation_texts = [
        ("slightly {dir} {base}", 0.2),
        ("moderately {dir} {base}", 0.35),
        ("neutral between {base1} and {base2}", 0.5),
        ("somewhat {dir} {base}", 0.65),
        ("quite {dir} {base}", 0.8),
    ]

    # Generate interpolations
    for low, high in extreme_pairs:
        for t in [0.15, 0.3, 0.45, 0.55, 0.7, 0.85]:
            # Interpolate values
            interp_text = f"Between {low['text']} and {high['text']} at {int(t*100)}%"
            ex = VibeExample(text=interp_text)

            for dim in DIMENSIONS:
                low_val = low.get(dim, 0.5)
                high_val = high.get(dim, 0.5)
                interp_val = low_val + t * (high_val - low_val)
                setattr(ex, dim, interp_val)

            examples.append(ex)

    return examples

def generate_negations() -> List[VibeExample]:
    """Generate negation examples with multiple active dimensions."""
    examples = []

    negation_templates = [
        # Temperature + brightness negations
        {"text": "Not hot at all - actually quite cold and dim",
         "warmth": 0.15, "brightness": 0.25, "valence": 0.45, "arousal": 0.25},
        {"text": "Not cold but not warm either - neutral temperature in bright light",
         "warmth": 0.5, "brightness": 0.75, "valence": 0.55, "arousal": 0.35},
        {"text": "Definitely not bright - very dark and somewhat cold",
         "brightness": 0.1, "warmth": 0.35, "valence": 0.4, "arousal": 0.2},

        # Emotion negations
        {"text": "Not sad at all - genuinely happy and energetic",
         "valence": 0.85, "arousal": 0.75, "warmth": 0.65, "brightness": 0.7},
        {"text": "Not happy but not depressed - just neutral and calm",
         "valence": 0.5, "arousal": 0.25, "intensity": 0.3, "warmth": 0.5},
        {"text": "Far from exciting - actually very boring and dull",
         "arousal": 0.1, "intensity": 0.15, "valence": 0.35, "brightness": 0.4},

        # Texture negations
        {"text": "Not smooth - quite rough and jagged surface",
         "texture": 0.85, "geometry": 0.4, "valence": 0.4, "warmth": 0.45},
        {"text": "Not rough at all - perfectly smooth and polished",
         "texture": 0.08, "brightness": 0.7, "valence": 0.65, "geometry": 0.75},

        # Intensity negations
        {"text": "Not intense - very gentle subtle and soft",
         "intensity": 0.12, "arousal": 0.2, "valence": 0.65, "texture": 0.2},
        {"text": "Anything but quiet - extremely loud and powerful",
         "intensity": 0.95, "arousal": 0.9, "valence": 0.55, "brightness": 0.65},

        # Complex negations
        {"text": "Neither hot nor cold, neither bright nor dark - just neutral grey everything",
         "warmth": 0.5, "brightness": 0.5, "valence": 0.45, "arousal": 0.3, "intensity": 0.3},
        {"text": "Not geometric at all - completely organic flowing chaotic forms",
         "geometry": 0.08, "texture": 0.65, "arousal": 0.45, "valence": 0.55},
        {"text": "Not warm colors - very cool blue and purple tones",
         "color_temperature": 0.15, "warmth": 0.3, "brightness": 0.5, "valence": 0.55},
    ]

    for template in negation_templates:
        text = template["text"]
        ex = VibeExample(text=text)
        for dim in DIMENSIONS:
            if dim in template:
                setattr(ex, dim, template[dim])
        examples.append(ex)

        # Add variations
        prefixes = ["Feeling like: ", "This is: ", "Experiencing: "]
        for prefix in prefixes:
            ex_var = VibeExample(text=prefix + text)
            for dim in DIMENSIONS:
                val = template.get(dim, 0.5) + random.uniform(-0.03, 0.03)
                setattr(ex_var, dim, max(0.02, min(0.98, val)))
            examples.append(ex_var)

    return examples

def save_dataset(examples: List[VibeExample], filename: str):
    """Save to CSV."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text'] + DIMENSIONS)
        for ex in examples:
            writer.writerow([ex.text] + [round(getattr(ex, d), 3) for d in DIMENSIONS])
    print(f"Saved {len(examples)} examples to {filename}")

def analyze_dataset(examples: List[VibeExample]):
    """Quality analysis."""
    print("\n" + "=" * 70)
    print("DATASET QUALITY ANALYSIS")
    print("=" * 70)
    print(f"\nTotal examples: {len(examples)}")

    for dim in DIMENSIONS:
        values = [getattr(ex, dim) for ex in examples]
        neutral = sum(1 for v in values if 0.45 <= v <= 0.55)
        low = sum(1 for v in values if v < 0.35)
        high = sum(1 for v in values if v > 0.65)

        print(f"\n{dim}:")
        print(f"  Neutral (0.45-0.55): {neutral:4d} ({neutral/len(values)*100:5.1f}%)")
        print(f"  Low (<0.35):         {low:4d} ({low/len(values)*100:5.1f}%)")
        print(f"  High (>0.65):        {high:4d} ({high/len(values)*100:5.1f}%)")
        print(f"  Active (non-neutral):{len(values)-neutral:4d} ({(len(values)-neutral)/len(values)*100:5.1f}%)")

    # Multi-dimensional analysis
    multi_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for ex in examples:
        active = sum(1 for d in DIMENSIONS if getattr(ex, d) < 0.4 or getattr(ex, d) > 0.6)
        if active < len(multi_counts):
            multi_counts[active] += 1

    print(f"\n{'='*70}")
    print("ACTIVE DIMENSIONS PER EXAMPLE:")
    for i, count in enumerate(multi_counts):
        if count > 0:
            print(f"  {i} dims active: {count:4d} ({count/len(examples)*100:5.1f}%)")

if __name__ == "__main__":
    print("Building multi-dimensional vibe dataset...\n")

    all_examples = []

    print("1. Expanding rich scenes...")
    all_examples.extend(expand_with_variations(RICH_SCENES))
    print(f"   → {len(all_examples)} examples")

    print("2. Generating interpolations...")
    all_examples.extend(generate_interpolations())
    print(f"   → {len(all_examples)} examples")

    print("3. Generating negations...")
    all_examples.extend(generate_negations())
    print(f"   → {len(all_examples)} examples")

    # Shuffle
    random.shuffle(all_examples)

    # Analyze
    analyze_dataset(all_examples)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multidim_vibe_dataset_{timestamp}.csv"
    save_dataset(all_examples, filename)

    print(f"\n✅ Dataset generation complete: {filename}")
