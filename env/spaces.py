"""Observation and action space builders for MarketCanvasEnv."""

from __future__ import annotations

from gymnasium import spaces
import numpy as np

NUM_ELEMENT_FEATURES = 15

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


def build_observation_space(max_elements: int) -> spaces.Dict:
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
        }
    )


def build_action_space(max_elements: int) -> spaces.Dict:
    """Build the Gymnasium action space."""

    return spaces.Dict(
        {
            "action_type": spaces.Discrete(NUM_ACTION_TYPES),
            "element_idx": spaces.Discrete(max_elements),
            "x": spaces.Discrete(800),
            "y": spaces.Discrete(600),
            "width": spaces.Discrete(800),
            "height": spaces.Discrete(600),
            "color_idx": spaces.Discrete(len(COLOR_PALETTE)),
            "content_idx": spaces.Discrete(len(CONTENT_TEMPLATES)),
        }
    )
