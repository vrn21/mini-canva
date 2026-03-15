"""Observation and action space builders for MarketCanvasEnv."""

from __future__ import annotations

from typing import Any, Literal

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

ACTION_INTERFACE_SEMANTIC = "semantic"
ACTION_INTERFACE_LOW_LEVEL = "low_level"
ACTION_INTERFACES = (
    ACTION_INTERFACE_SEMANTIC,
    ACTION_INTERFACE_LOW_LEVEL,
)
ActionInterface = Literal["semantic", "low_level"]

ACTION_ADD_TEXT = 0
ACTION_ADD_SHAPE = 1
ACTION_ADD_IMAGE = 2
ACTION_MOVE = 3
ACTION_RECOLOR = 4
ACTION_REMOVE = 5
ACTION_DONE = 6
NUM_ACTION_TYPES = 7

LOW_LEVEL_ACTION_MOUSE_MOVE = 0
LOW_LEVEL_ACTION_MOUSE_CLICK = 1
LOW_LEVEL_ACTION_MOUSE_DRAG = 2
LOW_LEVEL_ACTION_KEYBOARD_TYPE = 3
LOW_LEVEL_ACTION_SET_TOOL = 4
LOW_LEVEL_ACTION_DONE = 5
NUM_LOW_LEVEL_ACTION_TYPES = 6

ACTIVE_TOOL_SELECT = 0
ACTIVE_TOOL_TEXT = 1
ACTIVE_TOOL_SHAPE = 2
ACTIVE_TOOL_IMAGE = 3
NUM_ACTIVE_TOOLS = 4

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


def build_observation_space(
    max_elements: int,
    num_prompts: int,
    *,
    include_interaction: bool = False,
) -> spaces.Dict:
    """Build the Gymnasium observation space."""

    observation_spaces: dict[str, spaces.Space[Any]] = {
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

    if include_interaction:
        observation_spaces["cursor"] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        observation_spaces["active_tool"] = spaces.Discrete(NUM_ACTIVE_TOOLS)
        observation_spaces["selected_element_idx"] = spaces.Discrete(max_elements + 1)
        observation_spaces["focused_element_idx"] = spaces.Discrete(max_elements + 1)

    return spaces.Dict(observation_spaces)


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


def build_low_level_action_space(
    canvas_width: int = 800,
    canvas_height: int = 600,
    max_text_length: int = 64,
) -> spaces.Dict:
    """Build the optional low-level action space."""

    return spaces.Dict(
        {
            "action_type": spaces.Discrete(NUM_LOW_LEVEL_ACTION_TYPES),
            "x": spaces.Discrete(canvas_width),
            "y": spaces.Discrete(canvas_height),
            "x2": spaces.Discrete(canvas_width),
            "y2": spaces.Discrete(canvas_height),
            "tool": spaces.Discrete(NUM_ACTIVE_TOOLS),
            "text": spaces.Text(max_length=max_text_length),
        }
    )
