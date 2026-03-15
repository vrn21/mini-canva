"""Observation and action space builders for MarketCanvasEnv."""

from __future__ import annotations

from typing import Literal

from gymnasium import spaces
import numpy as np

NUM_ELEMENT_FEATURES = 15
DEFAULT_PIXEL_SIZE = (128, 96)

OBSERVATION_MODE_SEMANTIC = "semantic"
OBSERVATION_MODE_SEMANTIC_PIXELS = "semantic+pixels"
OBSERVATION_MODE_PIXELS = "pixels"
OBSERVATION_MODES = (
    OBSERVATION_MODE_SEMANTIC,
    OBSERVATION_MODE_SEMANTIC_PIXELS,
    OBSERVATION_MODE_PIXELS,
)
ObservationMode = Literal["semantic", "semantic+pixels", "pixels"]

ACTION_ADD_TEXT = 0
ACTION_ADD_SHAPE = 1
ACTION_ADD_IMAGE = 2
ACTION_MOVE = 3
ACTION_RECOLOR = 4
ACTION_REMOVE = 5
ACTION_DONE = 6
NUM_ACTION_TYPES = 7

COLOR_PALETTE: tuple[str, ...] = (
    "#FFFFFF",
    "#000000",
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFD700",
    "#FF6600",
    "#800080",
    "#00CED1",
    "#FF69B4",
    "#228B22",
    "#8B4513",
    "#708090",
    "#DC143C",
    "#4169E1",
    "#CCCCCC",
)

CONTENT_TEMPLATES: tuple[str, ...] = (
    "Summer Sale",
    "Shop Now",
    "Buy Today",
    "New Arrival",
    "Limited Offer",
    "Subscribe",
    "Learn More",
    "Get Started",
    "Free Shipping",
    "50% Off",
    "Coming Soon",
    "Join Now",
    "Best Seller",
    "Product Name",
    "Enter Email",
    "Happy Holidays",
    "Flash Sale",
    "Exclusive Deal",
    "Order Now",
    "Sign Up",
)


def build_observation_space(max_elements: int, num_prompts: int) -> spaces.Dict:
    """Build the Gymnasium observation space."""

    return spaces.Dict(
        {
            "elements": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(max_elements, NUM_ELEMENT_FEATURES),
                dtype=np.float32,
            ),
            "element_mask": spaces.MultiBinary(max_elements),
            "step_fraction": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "prompt_id": spaces.Discrete(num_prompts),
        }
    )


def build_pixel_observation_space(size: tuple[int, int]) -> spaces.Box:
    """Build the pixel observation space for rollout-time RGB observations."""

    width, height = size
    return spaces.Box(
        low=0,
        high=255,
        shape=(height, width, 3),
        dtype=np.uint8,
    )


def build_action_space(
    max_elements: int, canvas_width: int = 800, canvas_height: int = 600
) -> spaces.Dict:
    """Build the Gymnasium action space."""

    return spaces.Dict(
        {
            "action_type": spaces.Discrete(NUM_ACTION_TYPES),
            "element_idx": spaces.Discrete(max_elements),
            "x": spaces.Discrete(canvas_width),
            "y": spaces.Discrete(canvas_height),
            "width": spaces.Discrete(canvas_width),
            "height": spaces.Discrete(canvas_height),
            "color_idx": spaces.Discrete(len(COLOR_PALETTE)),
            "content_idx": spaces.Discrete(len(CONTENT_TEMPLATES)),
        }
    )
