"""Core data types for the canvas engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    """The three element archetypes supported by the canvas.

    Inherits from str so that json.dumps(ElementType.TEXT) → '"TEXT"'
    without a custom encoder.
    """

    TEXT = "TEXT"
    SHAPE = "SHAPE"
    IMAGE = "IMAGE"


@dataclass(slots=True)
class Element:
    """A single visual element on the canvas.

    Every property here appears in the JSON state representation.
    No property should exist that isn't serializable.

    Note: z-order is NOT stored on the element. It is determined by the
    element's position in the Canvas._elements list (index 0 = backmost).
    The z_index is derived from list position only during serialization.

    Attributes:
        id: Unique, stable identifier (e.g. "element_0"). Assigned by Canvas.
        type: One of TEXT, SHAPE, IMAGE.
        x: Top-left X position in pixels.
        y: Top-left Y position in pixels.
        width: Bounding-box width in pixels. Must be > 0.
        height: Bounding-box height in pixels. Must be > 0.
        color: Fill / background color as 7-char hex string (e.g. "#FF0000").
        text_color: Text color as 7-char hex string.
        content: Text content for TEXT/SHAPE labels, or descriptor for IMAGE.
        font_size: Font size in pixels. Only meaningful for TEXT and SHAPE-with-content.
    """

    id: str
    type: ElementType
    x: int
    y: int
    width: int
    height: int
    color: str = "#CCCCCC"
    text_color: str = "#000000"
    content: str = ""
    font_size: int = 16

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Returns (left, top, right, bottom) — the bounding rectangle."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> tuple[float, float]:
        """Returns (center_x, center_y) as floats."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> int:
        """Returns the area of the bounding box in square pixels."""
        return self.width * self.height

    def to_dict(self, z_index: int = 0) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Args:
            z_index: Drawing order position, passed by Canvas from list index.
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "z_index": z_index,
            "color": self.color,
            "text_color": self.text_color,
            "content": self.content,
            "font_size": self.font_size,
        }


@dataclass(frozen=True)
class CanvasConfig:
    """Configuration for a Canvas instance. Immutable after creation.

    Attributes:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        background_color: Canvas background as hex string.
        max_elements: Optional cap on elements. None means unlimited.
    """

    width: int = 800
    height: int = 600
    background_color: str = "#FFFFFF"
    max_elements: int | None = None
