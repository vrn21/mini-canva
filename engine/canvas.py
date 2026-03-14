"""Canvas — the scene graph / state container.

The Canvas is the single source of truth for canvas state. All mutations
and queries go through this class.

Internal storage uses a list for elements (ordered back→front) with a
sidecar dict index for O(1) ID lookups. List position IS the z-order:
index 0 is the backmost element, last index is the frontmost.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from typing import Any, Optional

import numpy as np

from engine.types import CanvasConfig, Element, ElementType

# Feature dimension for to_numpy(): type, x, y, width, height, color_r, color_g, color_b,
# text_color_r, text_color_g, text_color_b, font_size, content_length
_FEATURE_DIM = 13


class Canvas:
    """A 2D canvas holding an ordered collection of elements.

    Elements are stored in a list ordered back-to-front (list position = z-order).
    A sidecar dict maps element IDs to list indices for O(1) lookups.

    The Canvas knows nothing about RL, rewards, or rendering.
    It is a pure data container with mutation methods.
    """

    def __init__(self, config: CanvasConfig | None = None) -> None:
        self.config = config or CanvasConfig()
        self._elements: list[Element] = []
        self._index: dict[str, int] = {}
        self._next_id: int = 0

    # ════════════════════════════════════════════════════════════════════════
    #  QUERIES — read-only access to elements
    # ════════════════════════════════════════════════════════════════════════

    @property
    def element_count(self) -> int:
        """Number of elements currently on the canvas."""
        return len(self._elements)

    def get_element(self, element_id: str) -> Optional[Element]:
        """Get a single element by ID. Returns None if not found."""
        idx = self._index.get(element_id)
        if idx is None:
            return None
        return self._elements[idx]

    def get_all_elements(self) -> list[Element]:
        """Get all elements in z-order (back to front).

        Returns a new list each call — safe to iterate while mutating.
        """
        return list(self._elements)

    def has_element(self, element_id: str) -> bool:
        """Check if an element with the given ID exists."""
        return element_id in self._index

    # ════════════════════════════════════════════════════════════════════════
    #  MUTATIONS — modify the canvas state
    # ════════════════════════════════════════════════════════════════════════

    def add_element(
        self,
        element_type: ElementType,
        x: int,
        y: int,
        width: int,
        height: int,
        color: str = "#CCCCCC",
        text_color: str = "#000000",
        content: str = "",
        font_size: int = 16,
    ) -> Optional[Element]:
        """Add a new element to the front (top) of the canvas.

        Args:
            element_type: TEXT, SHAPE, or IMAGE.
            x, y: Top-left position.
            width, height: Bounding box dimensions. Must be > 0.
            color: Fill color hex.
            text_color: Text color hex.
            content: Text content or label.
            font_size: Font size in pixels.

        Returns:
            The newly created Element, or None if the canvas is full
            (when max_elements is set) or if width/height are <= 0.
        """
        if self.config.max_elements is not None and self.element_count >= self.config.max_elements:
            return None

        if width <= 0 or height <= 0:
            return None

        element_id = f"element_{self._next_id}"
        self._next_id += 1

        element = Element(
            id=element_id,
            type=element_type,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            text_color=text_color,
            content=content,
            font_size=font_size,
        )

        self._index[element_id] = len(self._elements)
        self._elements.append(element)
        return element

    def remove_element(self, element_id: str) -> bool:
        """Remove an element by ID.

        Returns:
            True if the element existed and was removed, False otherwise.
        """
        idx = self._index.get(element_id)
        if idx is None:
            return False

        self._elements.pop(idx)
        self._rebuild_index()
        return True

    def move_element(self, element_id: str, new_x: int, new_y: int) -> bool:
        """Move an element to a new (x, y) position.

        Does NOT clamp to canvas bounds. Off-canvas elements are legal
        (the reward function penalizes them, not the engine).

        Returns:
            True if the element was found and moved, False otherwise.
        """
        element = self.get_element(element_id)
        if element is None:
            return False
        element.x = new_x
        element.y = new_y
        return True

    def resize_element(self, element_id: str, new_width: int, new_height: int) -> bool:
        """Resize an element. Width and height must be > 0.

        Returns:
            True if the element was found and resized, False otherwise.
        """
        element = self.get_element(element_id)
        if element is None:
            return False
        if new_width <= 0 or new_height <= 0:
            return False
        element.width = new_width
        element.height = new_height
        return True

    def update_element(self, element_id: str, **kwargs: Any) -> bool:
        """Update arbitrary properties of an element.

        Args:
            element_id: ID of the element to update.
            **kwargs: Field names and new values. Invalid field names are ignored.
                      'id' and 'type' cannot be changed (silently ignored).
                      'width' and 'height' must be > 0 (returns False if not).

        Returns:
            True if the element was found and at least one field was updated.
        """
        element = self.get_element(element_id)
        if element is None:
            return False

        immutable = {"id", "type"}
        valid_fields = {f.name for f in fields(element)} - immutable

        updated = False
        for key, value in kwargs.items():
            if key not in valid_fields:
                continue
            if key in ("width", "height") and value <= 0:
                return False
            setattr(element, key, value)
            updated = True

        return updated

    def reorder_element(self, element_id: str, new_index: int) -> bool:
        """Move an element to a new position in the z-order.

        Index 0 = backmost, len-1 = frontmost.
        The index is clamped to valid bounds.

        Returns:
            True if the element was found and reordered, False otherwise.
        """
        idx = self._index.get(element_id)
        if idx is None:
            return False

        new_index = max(0, min(new_index, len(self._elements) - 1))
        if idx == new_index:
            return True

        element = self._elements.pop(idx)
        self._elements.insert(new_index, element)
        self._rebuild_index()
        return True

    def bring_to_front(self, element_id: str) -> bool:
        """Move an element to the front (topmost z-order)."""
        return self.reorder_element(element_id, len(self._elements) - 1)

    def send_to_back(self, element_id: str) -> bool:
        """Move an element to the back (bottommost z-order)."""
        return self.reorder_element(element_id, 0)

    def clear(self) -> None:
        """Remove all elements and reset the ID counter.

        Called at the start of each RL episode via env.reset().
        """
        self._elements.clear()
        self._index.clear()
        self._next_id = 0

    # ════════════════════════════════════════════════════════════════════════
    #  SPATIAL QUERIES — used by reward engine downstream
    # ════════════════════════════════════════════════════════════════════════

    def get_elements_at(self, x: int, y: int) -> list[Element]:
        """Get all elements whose bounding box contains the point (x, y).

        Returns elements sorted topmost first (reverse of list order).
        """
        hits = []
        for element in self._elements:
            left, top, right, bottom = element.bounds
            if left <= x < right and top <= y < bottom:
                hits.append(element)
        hits.reverse()
        return hits

    def get_overlapping_pairs(self) -> list[tuple[str, str, int]]:
        """Find all pairs of elements with overlapping bounding boxes.

        Returns:
            List of (id_a, id_b, overlap_area) tuples where overlap_area > 0.
            Each pair appears only once.
        """
        elements = self._elements
        n = len(elements)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                overlap = self._intersection_area(elements[i], elements[j])
                if overlap > 0:
                    pairs.append((elements[i].id, elements[j].id, overlap))

        return pairs

    def get_elements_by_type(self, element_type: ElementType) -> list[Element]:
        """Get all elements of a specific type, in z-order (back to front)."""
        return [e for e in self._elements if e.type == element_type]

    def get_element_behind(self, element_id: str) -> Optional[Element]:
        """Get the element directly behind the given element that overlaps with it.

        Used by WCAG contrast checker: "what color is behind this text?"
        Returns None if no overlapping element is behind, meaning the
        canvas background is the effective background.
        """
        idx = self._index.get(element_id)
        if idx is None:
            return None

        target = self._elements[idx]
        best: Optional[Element] = None

        for i in range(idx - 1, -1, -1):
            candidate = self._elements[i]
            if self._intersection_area(target, candidate) > 0:
                best = candidate
                break

        return best

    # ════════════════════════════════════════════════════════════════════════
    #  SERIALIZATION
    # ════════════════════════════════════════════════════════════════════════

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire canvas state to a JSON-compatible dict.

        z_index is derived from each element's list position.
        """
        return {
            "canvas": {
                "width": self.config.width,
                "height": self.config.height,
                "background_color": self.config.background_color,
            },
            "elements": [e.to_dict(z_index=i) for i, e in enumerate(self._elements)],
            "element_count": self.element_count,
        }

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Export canvas state as fixed-size numeric arrays for RL.

        Values are NOT normalized — the RL wrapper should normalize as needed.

        Returns:
            (features, mask) where:
            - features: float32 array of shape (n_elements, 13) with columns:
              [type, x, y, width, height, color_r, color_g, color_b,
               text_color_r, text_color_g, text_color_b, font_size, content_length]
            - mask: bool array of shape (n_elements,), all True.
              Included so the RL wrapper can pad to a fixed size with False entries.
        """
        n = len(self._elements)
        if n == 0:
            return np.empty((0, _FEATURE_DIM), dtype=np.float32), np.empty(0, dtype=np.bool_)

        features = np.empty((n, _FEATURE_DIM), dtype=np.float32)
        for i, e in enumerate(self._elements):
            cr, cg, cb = _hex_to_rgb(e.color)
            tr, tg, tb = _hex_to_rgb(e.text_color)
            features[i] = [
                _ELEMENT_TYPE_TO_INT[e.type],
                e.x, e.y, e.width, e.height,
                cr, cg, cb,
                tr, tg, tb,
                e.font_size,
                len(e.content),
            ]

        mask = np.ones(n, dtype=np.bool_)
        return features, mask

    # ════════════════════════════════════════════════════════════════════════
    #  SNAPSHOT / RESTORE — fast episode reset
    # ════════════════════════════════════════════════════════════════════════

    def snapshot(self) -> dict[str, Any]:
        """Capture the full canvas state for later restoration."""
        return {
            "elements": copy.deepcopy(self._elements),
            "next_id": self._next_id,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore canvas state from a snapshot."""
        self._elements = copy.deepcopy(snap["elements"])
        self._next_id = snap["next_id"]
        self._rebuild_index()

    # ════════════════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════════════════

    def _rebuild_index(self) -> None:
        """Rebuild the id→position sidecar index from the element list."""
        self._index = {e.id: i for i, e in enumerate(self._elements)}

    @staticmethod
    def _intersection_area(a: Element, b: Element) -> int:
        """Compute the area of intersection of two elements' bounding boxes."""
        left_a, top_a, right_a, bottom_a = a.bounds
        left_b, top_b, right_b, bottom_b = b.bounds

        inter_left = max(left_a, left_b)
        inter_top = max(top_a, top_b)
        inter_right = min(right_a, right_b)
        inter_bottom = min(bottom_a, bottom_b)

        if inter_left >= inter_right or inter_top >= inter_bottom:
            return 0

        return (inter_right - inter_left) * (inter_bottom - inter_top)


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════

_ELEMENT_TYPE_TO_INT: dict[ElementType, int] = {
    ElementType.TEXT: 0,
    ElementType.SHAPE: 1,
    ElementType.IMAGE: 2,
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (r, g, b) integers. Assumes valid 7-char hex."""
    return (
        int(hex_color[1:3], 16),
        int(hex_color[3:5], 16),
        int(hex_color[5:7], 16),
    )
