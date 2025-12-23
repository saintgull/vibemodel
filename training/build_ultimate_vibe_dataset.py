#!/usr/bin/env python3
"""
ULTIMATE VIBE DATASET BUILDER
=============================
Creates the highest quality training data for >99.99% accuracy.

Strategy: Direct, unambiguous examples for each dimension.
Each text must clearly evoke specific dimensional values.
"""

import pandas as pd
import numpy as np
import random

DIMENSIONS = ['warmth', 'brightness', 'texture', 'valence', 'arousal', 'intensity', 'geometry', 'color_temperature']

# ===== DIMENSION-SPECIFIC EXAMPLES =====
# Each example is (text, dict of dimension->value)
# Only specify dimensions that are clearly active

WARMTH_EXAMPLES = [
    # HIGH WARMTH
    ("scorching hot desert sun", {'warmth': 0.98, 'brightness': 0.95, 'intensity': 0.80, 'color_temperature': 0.95}),
    ("blazing summer heat wave", {'warmth': 0.95, 'brightness': 0.85, 'arousal': 0.65, 'color_temperature': 0.90}),
    ("steaming hot bath", {'warmth': 0.92, 'texture': 0.15, 'arousal': 0.20, 'valence': 0.80}),
    ("warm fireplace glow", {'warmth': 0.88, 'brightness': 0.35, 'valence': 0.85, 'color_temperature': 0.85}),
    ("tropical heat", {'warmth': 0.95, 'brightness': 0.80, 'valence': 0.70, 'color_temperature': 0.90}),
    ("hot summer afternoon", {'warmth': 0.90, 'brightness': 0.90, 'valence': 0.70, 'color_temperature': 0.85}),
    ("warm embrace", {'warmth': 0.85, 'texture': 0.20, 'valence': 0.90, 'intensity': 0.60}),
    ("cozy warm blanket", {'warmth': 0.85, 'texture': 0.20, 'valence': 0.85, 'arousal': 0.15}),
    ("hot cup of tea", {'warmth': 0.85, 'valence': 0.80, 'arousal': 0.25, 'intensity': 0.40}),
    ("balmy tropical evening", {'warmth': 0.82, 'brightness': 0.30, 'valence': 0.80, 'arousal': 0.30}),

    # LOW WARMTH
    ("freezing arctic tundra", {'warmth': 0.02, 'brightness': 0.70, 'intensity': 0.70, 'color_temperature': 0.05}),
    ("ice cold glacier", {'warmth': 0.05, 'brightness': 0.75, 'texture': 0.50, 'color_temperature': 0.05}),
    ("bitter winter cold", {'warmth': 0.08, 'valence': 0.30, 'intensity': 0.65, 'color_temperature': 0.10}),
    ("frigid winter night", {'warmth': 0.05, 'brightness': 0.15, 'arousal': 0.20, 'color_temperature': 0.08}),
    ("frozen lake surface", {'warmth': 0.08, 'brightness': 0.65, 'texture': 0.40, 'color_temperature': 0.10}),
    ("cold metal railing", {'warmth': 0.15, 'texture': 0.50, 'geometry': 0.75, 'color_temperature': 0.20}),
    ("chilly morning fog", {'warmth': 0.20, 'brightness': 0.45, 'texture': 0.15, 'color_temperature': 0.25}),
    ("icy wind cutting through", {'warmth': 0.10, 'texture': 0.70, 'intensity': 0.75, 'arousal': 0.60}),
    ("cold stone floor", {'warmth': 0.18, 'texture': 0.55, 'brightness': 0.45, 'geometry': 0.70}),
    ("frosty window pane", {'warmth': 0.12, 'brightness': 0.60, 'texture': 0.45, 'color_temperature': 0.15}),

    # MID WARMTH
    ("room temperature water", {'warmth': 0.50, 'texture': 0.20, 'valence': 0.55, 'intensity': 0.20}),
    ("mild spring day", {'warmth': 0.55, 'brightness': 0.70, 'valence': 0.75, 'arousal': 0.40}),
    ("temperate climate", {'warmth': 0.50, 'valence': 0.65, 'arousal': 0.35, 'intensity': 0.30}),
]

BRIGHTNESS_EXAMPLES = [
    # HIGH BRIGHTNESS
    ("blinding white light", {'brightness': 0.98, 'intensity': 0.90, 'arousal': 0.70}),
    ("blazing midday sun", {'brightness': 0.95, 'warmth': 0.90, 'intensity': 0.75, 'color_temperature': 0.85}),
    ("brilliant sunlight streaming in", {'brightness': 0.92, 'warmth': 0.75, 'valence': 0.85}),
    ("dazzling snow field", {'brightness': 0.95, 'warmth': 0.10, 'texture': 0.30, 'color_temperature': 0.15}),
    ("gleaming white marble", {'brightness': 0.88, 'texture': 0.10, 'geometry': 0.70, 'color_temperature': 0.40}),
    ("harsh fluorescent lights", {'brightness': 0.90, 'warmth': 0.35, 'valence': 0.35, 'geometry': 0.80}),
    ("bright sunny day", {'brightness': 0.90, 'warmth': 0.80, 'valence': 0.85, 'color_temperature': 0.75}),
    ("brilliant white clouds", {'brightness': 0.85, 'texture': 0.35, 'valence': 0.75}),
    ("glaring spotlight", {'brightness': 0.92, 'intensity': 0.85, 'geometry': 0.75}),
    ("luminous screen glow", {'brightness': 0.80, 'geometry': 0.80, 'color_temperature': 0.40}),

    # LOW BRIGHTNESS
    ("pitch black darkness", {'brightness': 0.02, 'intensity': 0.50, 'arousal': 0.30}),
    ("complete darkness", {'brightness': 0.03, 'intensity': 0.45}),
    ("deep shadows", {'brightness': 0.10, 'valence': 0.35, 'intensity': 0.55}),
    ("dim candle flicker", {'brightness': 0.18, 'warmth': 0.75, 'arousal': 0.25, 'color_temperature': 0.80}),
    ("murky underwater depths", {'brightness': 0.12, 'texture': 0.55, 'valence': 0.35, 'color_temperature': 0.30}),
    ("gloomy overcast sky", {'brightness': 0.25, 'valence': 0.35, 'arousal': 0.25, 'color_temperature': 0.35}),
    ("dark moonless night", {'brightness': 0.08, 'warmth': 0.25, 'arousal': 0.20}),
    ("shadowy alleyway", {'brightness': 0.15, 'valence': 0.25, 'arousal': 0.50, 'intensity': 0.55}),
    ("faint starlight", {'brightness': 0.12, 'valence': 0.60, 'intensity': 0.35, 'color_temperature': 0.30}),
    ("dusky twilight hour", {'brightness': 0.25, 'warmth': 0.55, 'valence': 0.60, 'arousal': 0.30}),

    # MID BRIGHTNESS
    ("soft diffused light", {'brightness': 0.55, 'texture': 0.20, 'valence': 0.70, 'intensity': 0.35}),
    ("gentle morning light", {'brightness': 0.60, 'warmth': 0.65, 'valence': 0.75, 'arousal': 0.35}),
    ("overcast afternoon", {'brightness': 0.50, 'valence': 0.50, 'arousal': 0.35}),
]

TEXTURE_EXAMPLES = [
    # SMOOTH (low texture)
    ("silk flowing like water", {'texture': 0.05, 'warmth': 0.60, 'valence': 0.80, 'geometry': 0.15}),
    ("polished glass surface", {'texture': 0.08, 'brightness': 0.75, 'geometry': 0.70}),
    ("smooth velvet cushion", {'texture': 0.10, 'warmth': 0.70, 'valence': 0.80}),
    ("satin sheets gliding", {'texture': 0.08, 'warmth': 0.65, 'valence': 0.85, 'arousal': 0.30}),
    ("buttery smooth cream", {'texture': 0.10, 'warmth': 0.60, 'valence': 0.75}),
    ("sleek polished metal", {'texture': 0.12, 'brightness': 0.70, 'geometry': 0.75, 'color_temperature': 0.30}),
    ("liquid mercury flowing", {'texture': 0.08, 'brightness': 0.75, 'geometry': 0.20}),
    ("soft cotton clouds", {'texture': 0.15, 'brightness': 0.80, 'valence': 0.75}),
    ("baby skin soft", {'texture': 0.10, 'warmth': 0.75, 'valence': 0.85}),
    ("smooth river stones", {'texture': 0.20, 'valence': 0.65, 'geometry': 0.40}),

    # ROUGH (high texture)
    ("rough sandpaper grating", {'texture': 0.95, 'valence': 0.25, 'intensity': 0.60}),
    ("jagged broken glass", {'texture': 0.92, 'valence': 0.15, 'intensity': 0.80, 'geometry': 0.90}),
    ("coarse burlap sack", {'texture': 0.85, 'warmth': 0.45, 'valence': 0.35}),
    ("rough tree bark", {'texture': 0.88, 'warmth': 0.50, 'geometry': 0.55}),
    ("gritty sand beach", {'texture': 0.80, 'warmth': 0.70, 'brightness': 0.75}),
    ("spiky cactus needles", {'texture': 0.90, 'valence': 0.25, 'intensity': 0.65, 'geometry': 0.85}),
    ("abrasive concrete surface", {'texture': 0.85, 'geometry': 0.75, 'color_temperature': 0.40}),
    ("rusted corroded metal", {'texture': 0.88, 'warmth': 0.45, 'valence': 0.25, 'color_temperature': 0.50}),
    ("cracked dry earth", {'texture': 0.82, 'warmth': 0.60, 'brightness': 0.65, 'valence': 0.35}),
    ("gravel crunching underfoot", {'texture': 0.80, 'arousal': 0.40, 'intensity': 0.45}),

    # MID TEXTURE
    ("linen fabric weave", {'texture': 0.55, 'warmth': 0.55, 'valence': 0.65}),
    ("wooden grain pattern", {'texture': 0.50, 'warmth': 0.60, 'geometry': 0.55}),
]

VALENCE_EXAMPLES = [
    # HIGH VALENCE (positive)
    ("pure ecstatic joy", {'valence': 0.98, 'arousal': 0.85, 'warmth': 0.80, 'brightness': 0.85}),
    ("blissful happiness", {'valence': 0.95, 'arousal': 0.70, 'warmth': 0.80, 'brightness': 0.80}),
    ("overwhelming love", {'valence': 0.95, 'intensity': 0.90, 'warmth': 0.90, 'arousal': 0.75}),
    ("peaceful contentment", {'valence': 0.85, 'arousal': 0.15, 'warmth': 0.70, 'intensity': 0.40}),
    ("triumphant victory", {'valence': 0.92, 'arousal': 0.90, 'intensity': 0.85, 'brightness': 0.80}),
    ("gentle comfort", {'valence': 0.82, 'warmth': 0.75, 'arousal': 0.20, 'texture': 0.20}),
    ("pure delight", {'valence': 0.90, 'arousal': 0.75, 'brightness': 0.80}),
    ("serene tranquility", {'valence': 0.85, 'arousal': 0.08, 'warmth': 0.60, 'intensity': 0.25}),
    ("heartwarming kindness", {'valence': 0.88, 'warmth': 0.85, 'intensity': 0.55}),
    ("joyful celebration", {'valence': 0.92, 'arousal': 0.85, 'brightness': 0.85, 'warmth': 0.75}),

    # LOW VALENCE (negative)
    ("crushing despair", {'valence': 0.02, 'intensity': 0.95, 'brightness': 0.15, 'arousal': 0.40}),
    ("devastating grief", {'valence': 0.05, 'intensity': 0.95, 'warmth': 0.25, 'arousal': 0.45}),
    ("overwhelming dread", {'valence': 0.08, 'arousal': 0.85, 'intensity': 0.90}),
    ("bitter resentment", {'valence': 0.15, 'warmth': 0.35, 'intensity': 0.70, 'texture': 0.65}),
    ("profound sadness", {'valence': 0.10, 'intensity': 0.80, 'arousal': 0.25, 'brightness': 0.20}),
    ("utter hopelessness", {'valence': 0.05, 'intensity': 0.85, 'brightness': 0.15}),
    ("gnawing anxiety", {'valence': 0.18, 'arousal': 0.80, 'intensity': 0.75, 'texture': 0.65}),
    ("deep depression", {'valence': 0.08, 'arousal': 0.15, 'brightness': 0.15, 'intensity': 0.75}),
    ("cold emptiness", {'valence': 0.12, 'warmth': 0.15, 'intensity': 0.60}),
    ("terrifying fear", {'valence': 0.10, 'arousal': 0.95, 'intensity': 0.92, 'warmth': 0.20}),

    # MID VALENCE (neutral/mixed)
    ("bittersweet memory", {'valence': 0.45, 'warmth': 0.55, 'intensity': 0.65, 'arousal': 0.30}),
    ("melancholic nostalgia", {'valence': 0.40, 'warmth': 0.55, 'intensity': 0.60, 'arousal': 0.25}),
    ("wistful longing", {'valence': 0.42, 'warmth': 0.50, 'intensity': 0.65, 'arousal': 0.30}),
    ("stoic acceptance", {'valence': 0.50, 'intensity': 0.55, 'arousal': 0.25}),
    ("complex ambivalence", {'valence': 0.50, 'intensity': 0.50, 'arousal': 0.45}),
]

AROUSAL_EXAMPLES = [
    # HIGH AROUSAL
    ("heart-pounding terror", {'arousal': 0.98, 'valence': 0.08, 'intensity': 0.95}),
    ("explosive excitement", {'arousal': 0.95, 'valence': 0.85, 'intensity': 0.90, 'brightness': 0.80}),
    ("manic energy surge", {'arousal': 0.95, 'intensity': 0.90, 'brightness': 0.75}),
    ("adrenaline rush", {'arousal': 0.92, 'intensity': 0.88, 'valence': 0.70}),
    ("electric anticipation", {'arousal': 0.88, 'valence': 0.75, 'intensity': 0.80}),
    ("frenzied chaos", {'arousal': 0.95, 'intensity': 0.90, 'texture': 0.75, 'valence': 0.40}),
    ("wild exhilaration", {'arousal': 0.92, 'valence': 0.85, 'intensity': 0.85}),
    ("panicked alarm", {'arousal': 0.95, 'valence': 0.15, 'intensity': 0.90}),
    ("ecstatic euphoria", {'arousal': 0.90, 'valence': 0.95, 'intensity': 0.85}),
    ("intense concentration", {'arousal': 0.75, 'intensity': 0.80, 'valence': 0.60}),

    # LOW AROUSAL
    ("deep peaceful sleep", {'arousal': 0.02, 'valence': 0.70, 'intensity': 0.15}),
    ("meditative stillness", {'arousal': 0.05, 'valence': 0.75, 'intensity': 0.20, 'warmth': 0.55}),
    ("drowsy relaxation", {'arousal': 0.10, 'valence': 0.70, 'warmth': 0.65}),
    ("serene calm", {'arousal': 0.08, 'valence': 0.80, 'intensity': 0.20}),
    ("quiet contemplation", {'arousal': 0.15, 'valence': 0.65, 'intensity': 0.30}),
    ("gentle tranquility", {'arousal': 0.10, 'valence': 0.80, 'warmth': 0.65, 'texture': 0.20}),
    ("sleepy afternoon", {'arousal': 0.12, 'valence': 0.65, 'warmth': 0.60, 'brightness': 0.55}),
    ("peaceful rest", {'arousal': 0.08, 'valence': 0.75, 'intensity': 0.20}),
    ("lazy Sunday morning", {'arousal': 0.15, 'valence': 0.75, 'warmth': 0.70}),
    ("still water surface", {'arousal': 0.05, 'texture': 0.10, 'valence': 0.70}),

    # MID AROUSAL
    ("steady focus", {'arousal': 0.50, 'valence': 0.60, 'intensity': 0.50}),
    ("mild interest", {'arousal': 0.45, 'valence': 0.65}),
]

INTENSITY_EXAMPLES = [
    # HIGH INTENSITY
    ("overwhelming catastrophe", {'intensity': 0.98, 'valence': 0.05, 'arousal': 0.85}),
    ("earth-shattering revelation", {'intensity': 0.95, 'arousal': 0.80, 'brightness': 0.70}),
    ("devastating impact", {'intensity': 0.95, 'valence': 0.10, 'texture': 0.75}),
    ("profound emotional weight", {'intensity': 0.90, 'valence': 0.35, 'arousal': 0.45}),
    ("crushing pressure", {'intensity': 0.92, 'valence': 0.20, 'arousal': 0.70}),
    ("overwhelming power", {'intensity': 0.95, 'arousal': 0.80}),
    ("mind-blowing revelation", {'intensity': 0.88, 'arousal': 0.85, 'valence': 0.75}),
    ("all-consuming passion", {'intensity': 0.92, 'warmth': 0.85, 'arousal': 0.85, 'valence': 0.80}),
    ("thundering crescendo", {'intensity': 0.90, 'arousal': 0.90, 'brightness': 0.65}),
    ("massive explosion", {'intensity': 0.98, 'arousal': 0.95, 'brightness': 0.90, 'texture': 0.85}),

    # LOW INTENSITY
    ("faint whisper", {'intensity': 0.08, 'arousal': 0.15, 'brightness': 0.40}),
    ("subtle hint", {'intensity': 0.12, 'arousal': 0.25}),
    ("gentle breeze", {'intensity': 0.15, 'arousal': 0.25, 'valence': 0.70, 'texture': 0.30}),
    ("soft murmur", {'intensity': 0.12, 'arousal': 0.20, 'texture': 0.20}),
    ("delicate touch", {'intensity': 0.15, 'texture': 0.15, 'valence': 0.70, 'warmth': 0.65}),
    ("mild sensation", {'intensity': 0.20, 'arousal': 0.30}),
    ("barely perceptible", {'intensity': 0.08, 'arousal': 0.15}),
    ("slight movement", {'intensity': 0.15, 'arousal': 0.25}),
    ("faint memory", {'intensity': 0.18, 'valence': 0.55, 'arousal': 0.20}),
    ("gentle nudge", {'intensity': 0.20, 'warmth': 0.60, 'valence': 0.65}),

    # MID INTENSITY
    ("moderate force", {'intensity': 0.50, 'arousal': 0.50}),
    ("steady presence", {'intensity': 0.55, 'arousal': 0.45, 'valence': 0.60}),
]

GEOMETRY_EXAMPLES = [
    # ANGULAR (high geometry)
    ("sharp geometric edges", {'geometry': 0.95, 'texture': 0.65}),
    ("rigid angular structure", {'geometry': 0.92, 'texture': 0.60, 'warmth': 0.35}),
    ("crystalline facets", {'geometry': 0.90, 'brightness': 0.75, 'texture': 0.45}),
    ("brutalist concrete blocks", {'geometry': 0.92, 'texture': 0.75, 'warmth': 0.30, 'valence': 0.35}),
    ("stark minimalist lines", {'geometry': 0.88, 'brightness': 0.70, 'intensity': 0.50}),
    ("harsh angular shadows", {'geometry': 0.85, 'brightness': 0.25, 'valence': 0.35}),
    ("industrial metal framework", {'geometry': 0.90, 'texture': 0.65, 'warmth': 0.30, 'color_temperature': 0.30}),
    ("precise mathematical grid", {'geometry': 0.95, 'brightness': 0.60, 'arousal': 0.40}),
    ("jagged mountain peaks", {'geometry': 0.85, 'texture': 0.70, 'intensity': 0.70}),
    ("cubist fragmentation", {'geometry': 0.92, 'texture': 0.55, 'arousal': 0.55}),

    # ORGANIC (low geometry)
    ("flowing organic curves", {'geometry': 0.08, 'texture': 0.25, 'valence': 0.70}),
    ("soft natural forms", {'geometry': 0.12, 'texture': 0.30, 'warmth': 0.60, 'valence': 0.75}),
    ("undulating waves", {'geometry': 0.10, 'texture': 0.25, 'arousal': 0.40}),
    ("gentle rolling hills", {'geometry': 0.15, 'valence': 0.75, 'warmth': 0.55}),
    ("swirling cloud patterns", {'geometry': 0.12, 'brightness': 0.65, 'texture': 0.25}),
    ("meandering river path", {'geometry': 0.10, 'texture': 0.30, 'valence': 0.70}),
    ("smooth pebble beach", {'geometry': 0.18, 'texture': 0.35, 'valence': 0.70}),
    ("drooping willow branches", {'geometry': 0.15, 'valence': 0.60, 'arousal': 0.20}),
    ("curving art nouveau lines", {'geometry': 0.12, 'valence': 0.75, 'warmth': 0.60}),
    ("rounded bubble shapes", {'geometry': 0.10, 'valence': 0.70, 'texture': 0.10}),

    # MID GEOMETRY
    ("balanced composition", {'geometry': 0.50, 'valence': 0.65}),
    ("mixed angular and curved", {'geometry': 0.50, 'texture': 0.50}),
]

COLOR_TEMPERATURE_EXAMPLES = [
    # WARM (high color_temperature)
    ("golden sunset glow", {'color_temperature': 0.95, 'warmth': 0.85, 'brightness': 0.70, 'valence': 0.85}),
    ("amber firelight", {'color_temperature': 0.92, 'warmth': 0.90, 'brightness': 0.40, 'valence': 0.80}),
    ("fiery orange flames", {'color_temperature': 0.98, 'warmth': 0.95, 'brightness': 0.75, 'intensity': 0.85}),
    ("rustic autumn leaves", {'color_temperature': 0.85, 'warmth': 0.70, 'valence': 0.70}),
    ("copper metallic sheen", {'color_temperature': 0.80, 'warmth': 0.65, 'brightness': 0.65, 'texture': 0.35}),
    ("burnt sienna earth", {'color_temperature': 0.82, 'warmth': 0.65, 'texture': 0.60}),
    ("honey golden light", {'color_temperature': 0.88, 'warmth': 0.80, 'brightness': 0.70, 'valence': 0.80}),
    ("candlelit warmth", {'color_temperature': 0.85, 'warmth': 0.85, 'brightness': 0.25, 'valence': 0.75}),
    ("terracotta tones", {'color_temperature': 0.80, 'warmth': 0.70, 'texture': 0.55}),
    ("ruby red glow", {'color_temperature': 0.88, 'warmth': 0.75, 'intensity': 0.65, 'valence': 0.65}),

    # COOL (low color_temperature)
    ("icy blue winter", {'color_temperature': 0.05, 'warmth': 0.08, 'brightness': 0.65}),
    ("steel grey industrial", {'color_temperature': 0.20, 'warmth': 0.30, 'geometry': 0.80, 'texture': 0.60}),
    ("silver moonlight", {'color_temperature': 0.15, 'warmth': 0.30, 'brightness': 0.50, 'valence': 0.60}),
    ("azure ocean depths", {'color_temperature': 0.12, 'warmth': 0.35, 'intensity': 0.55}),
    ("cool mint green", {'color_temperature': 0.25, 'warmth': 0.40, 'valence': 0.70}),
    ("polar white ice", {'color_temperature': 0.08, 'warmth': 0.05, 'brightness': 0.85, 'texture': 0.45}),
    ("clinical fluorescent", {'color_temperature': 0.30, 'brightness': 0.90, 'valence': 0.35, 'geometry': 0.80}),
    ("deep navy blue", {'color_temperature': 0.15, 'brightness': 0.25, 'intensity': 0.55}),
    ("slate grey overcast", {'color_temperature': 0.35, 'brightness': 0.45, 'valence': 0.40}),
    ("frosted glass blue", {'color_temperature': 0.18, 'warmth': 0.25, 'brightness': 0.60, 'texture': 0.30}),

    # MID COLOR TEMPERATURE
    ("neutral daylight", {'color_temperature': 0.50, 'brightness': 0.70, 'valence': 0.60}),
    ("white balance neutral", {'color_temperature': 0.50, 'brightness': 0.65}),
]

# Multi-dimensional scene examples
SCENE_EXAMPLES = [
    # Cozy/warm scenes
    ("grandmother's kitchen filled with baking bread", {'warmth': 0.92, 'brightness': 0.55, 'texture': 0.35, 'valence': 0.95, 'arousal': 0.30, 'intensity': 0.55, 'geometry': 0.30, 'color_temperature': 0.88}),
    ("crackling fireplace on winter evening", {'warmth': 0.95, 'brightness': 0.35, 'texture': 0.50, 'valence': 0.90, 'arousal': 0.25, 'intensity': 0.55, 'geometry': 0.35, 'color_temperature': 0.92}),
    ("cozy cabin in the mountains", {'warmth': 0.85, 'brightness': 0.45, 'texture': 0.45, 'valence': 0.88, 'arousal': 0.20, 'intensity': 0.45, 'geometry': 0.40, 'color_temperature': 0.80}),

    # Cold/industrial scenes
    ("abandoned factory in winter rain", {'warmth': 0.15, 'brightness': 0.30, 'texture': 0.85, 'valence': 0.15, 'arousal': 0.30, 'intensity': 0.65, 'geometry': 0.80, 'color_temperature': 0.25}),
    ("sterile hospital corridor at night", {'warmth': 0.30, 'brightness': 0.85, 'texture': 0.40, 'valence': 0.25, 'arousal': 0.50, 'intensity': 0.55, 'geometry': 0.85, 'color_temperature': 0.30}),
    ("brutalist concrete parking garage", {'warmth': 0.28, 'brightness': 0.45, 'texture': 0.75, 'valence': 0.25, 'arousal': 0.30, 'intensity': 0.55, 'geometry': 0.92, 'color_temperature': 0.32}),

    # Natural scenes
    ("tropical beach at sunset", {'warmth': 0.90, 'brightness': 0.75, 'texture': 0.40, 'valence': 0.92, 'arousal': 0.45, 'intensity': 0.60, 'geometry': 0.20, 'color_temperature': 0.90}),
    ("frozen lake under northern lights", {'warmth': 0.08, 'brightness': 0.55, 'texture': 0.35, 'valence': 0.75, 'arousal': 0.50, 'intensity': 0.80, 'geometry': 0.35, 'color_temperature': 0.15}),
    ("foggy forest at dawn", {'warmth': 0.45, 'brightness': 0.40, 'texture': 0.35, 'valence': 0.65, 'arousal': 0.25, 'intensity': 0.50, 'geometry': 0.35, 'color_temperature': 0.45}),
    ("thunderstorm over mountains", {'warmth': 0.40, 'brightness': 0.25, 'texture': 0.65, 'valence': 0.35, 'arousal': 0.85, 'intensity': 0.92, 'geometry': 0.55, 'color_temperature': 0.38}),
    ("cherry blossoms in spring", {'warmth': 0.70, 'brightness': 0.80, 'texture': 0.25, 'valence': 0.90, 'arousal': 0.40, 'intensity': 0.50, 'geometry': 0.20, 'color_temperature': 0.68}),

    # Urban scenes
    ("neon-lit cyberpunk city", {'warmth': 0.50, 'brightness': 0.80, 'texture': 0.55, 'valence': 0.55, 'arousal': 0.85, 'intensity': 0.80, 'geometry': 0.85, 'color_temperature': 0.55}),
    ("quiet library at midnight", {'warmth': 0.55, 'brightness': 0.30, 'texture': 0.40, 'valence': 0.65, 'arousal': 0.10, 'intensity': 0.35, 'geometry': 0.70, 'color_temperature': 0.50}),
    ("jazz club smoky atmosphere", {'warmth': 0.65, 'brightness': 0.22, 'texture': 0.35, 'valence': 0.72, 'arousal': 0.45, 'intensity': 0.55, 'geometry': 0.40, 'color_temperature': 0.65}),

    # Emotional moments
    ("children laughing in summer sunshine", {'warmth': 0.88, 'brightness': 0.90, 'texture': 0.30, 'valence': 0.95, 'arousal': 0.75, 'intensity': 0.65, 'geometry': 0.25, 'color_temperature': 0.82}),
    ("funeral in the rain", {'warmth': 0.30, 'brightness': 0.25, 'texture': 0.45, 'valence': 0.08, 'arousal': 0.30, 'intensity': 0.85, 'geometry': 0.55, 'color_temperature': 0.35}),
    ("wedding ceremony at golden hour", {'warmth': 0.85, 'brightness': 0.75, 'texture': 0.25, 'valence': 0.95, 'arousal': 0.65, 'intensity': 0.75, 'geometry': 0.45, 'color_temperature': 0.88}),
    ("lonely vigil by candlelight", {'warmth': 0.70, 'brightness': 0.18, 'texture': 0.30, 'valence': 0.35, 'arousal': 0.15, 'intensity': 0.70, 'geometry': 0.40, 'color_temperature': 0.75}),
]


def build_dataset():
    """Build the ultimate vibe dataset."""
    all_examples = []

    # Collect all dimension-specific examples
    dimension_sources = [
        WARMTH_EXAMPLES,
        BRIGHTNESS_EXAMPLES,
        TEXTURE_EXAMPLES,
        VALENCE_EXAMPLES,
        AROUSAL_EXAMPLES,
        INTENSITY_EXAMPLES,
        GEOMETRY_EXAMPLES,
        COLOR_TEMPERATURE_EXAMPLES,
        SCENE_EXAMPLES,
    ]

    for source in dimension_sources:
        for text, dims in source:
            row = {'text': text}
            # Fill in specified dimensions
            for dim in DIMENSIONS:
                if dim in dims:
                    row[dim] = dims[dim]
                else:
                    # Use neutral for unspecified
                    row[dim] = 0.5
            all_examples.append(row)

    df = pd.DataFrame(all_examples)
    return df


def augment_dataset(df, target_size=5000):
    """Augment dataset to target size with variations."""
    augmented = df.copy().to_dict('records')

    # Variation templates
    prefixes = [
        "the feeling of ",
        "experiencing ",
        "like ",
        "evokes ",
        "reminds me of ",
        "pure ",
        "absolute ",
        "total ",
        "",
    ]

    suffixes = [
        "",
        " energy",
        " vibes",
        " atmosphere",
        " sensation",
        " feeling",
    ]

    intensifiers = {
        'very': 0.15,
        'extremely': 0.25,
        'slightly': -0.15,
        'somewhat': -0.10,
        'barely': -0.20,
        'intensely': 0.20,
    }

    # Generate variations until target size
    original_texts = set(r['text'] for r in augmented)

    while len(augmented) < target_size:
        # Pick random original
        orig = random.choice(df.to_dict('records'))

        # Apply random transformation
        transform = random.choice(['prefix', 'suffix', 'intensifier', 'combo'])

        new_text = orig['text']
        new_row = orig.copy()

        if transform == 'prefix':
            prefix = random.choice(prefixes)
            new_text = prefix + orig['text']
        elif transform == 'suffix':
            suffix = random.choice(suffixes)
            new_text = orig['text'] + suffix
        elif transform == 'intensifier':
            intensifier = random.choice(list(intensifiers.keys()))
            new_text = intensifier + " " + orig['text']
            # Adjust values
            delta = intensifiers[intensifier]
            for dim in DIMENSIONS:
                if new_row[dim] != 0.5:  # Only adjust active dimensions
                    if new_row[dim] > 0.5:
                        new_row[dim] = min(1.0, new_row[dim] + delta)
                    else:
                        new_row[dim] = max(0.0, new_row[dim] - delta)
        else:  # combo
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            new_text = prefix + orig['text'] + suffix

        # Only add if unique
        if new_text not in original_texts:
            new_row['text'] = new_text
            augmented.append(new_row)
            original_texts.add(new_text)

    return pd.DataFrame(augmented[:target_size])


if __name__ == "__main__":
    print("=" * 70)
    print("BUILDING ULTIMATE VIBE DATASET")
    print("=" * 70)

    # Build base dataset
    print("\n1. Building base dataset...")
    base_df = build_dataset()
    print(f"   Base examples: {len(base_df)}")

    # Augment to target size
    print("\n2. Augmenting dataset...")
    final_df = augment_dataset(base_df, target_size=5000)
    print(f"   Final examples: {len(final_df)}")

    # Save
    output_path = "/Users/erinsaintgull/P/ultimate_vibe_training_data.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\n   Saved to: {output_path}")

    # Show stats
    print(f"\n{'=' * 70}")
    print("DIMENSION STATISTICS")
    print(f"{'=' * 70}")
    for dim in DIMENSIONS:
        values = final_df[dim]
        active = ((values < 0.4) | (values > 0.6)).sum() / len(values) * 100
        print(f"  {dim:<18} min={values.min():.2f} max={values.max():.2f} mean={values.mean():.2f} active={active:.1f}%")
