"""Constraint checker — evaluates if a canvas satisfies prompt constraints."""

from __future__ import annotations

from typing import Any

from engine.canvas import Canvas
from engine.types import ElementType

from rewards.prompts import ConstraintType, PromptConstraint


class ConstraintChecker:
    """Checks if a canvas state satisfies a list of prompt constraints."""

    def score(self, canvas: Canvas, constraints: tuple[PromptConstraint, ...]) -> float:
        """Fraction of satisfied constraints."""

        if not constraints:
            return 1.0

        satisfied = sum(1 for constraint in constraints if self._check_one(canvas, constraint))
        return satisfied / len(constraints)

    def _check_one(self, canvas: Canvas, constraint: PromptConstraint) -> bool:
        """Dispatch a single constraint to the appropriate checker."""

        handlers = {
            ConstraintType.HAS_ELEMENT: self._check_has_element,
            ConstraintType.ELEMENT_COLOR: self._check_element_color,
            ConstraintType.MIN_ELEMENTS: self._check_min_elements,
        }
        handler = handlers.get(constraint.type)
        if handler is None:
            return False
        return handler(canvas, constraint.params)

    def _check_has_element(self, canvas: Canvas, params: dict[str, Any]) -> bool:
        """Check if an element of the given type exists, optionally matching keywords."""

        target_type = ElementType(params["type"])
        elements = canvas.get_elements_by_type(target_type)
        if not elements:
            return False

        keywords = params.get("keywords")
        if not keywords:
            return True

        for element in elements:
            content_lower = element.content.lower()
            if any(keyword.lower() in content_lower for keyword in keywords):
                return True

        return False

    def _check_element_color(self, canvas: Canvas, params: dict[str, Any]) -> bool:
        """Check if an element matching criteria has a color close to the target."""

        target_type = ElementType(params["type"])
        elements = canvas.get_elements_by_type(target_type)
        keywords = params.get("keywords")
        target_color = params["target_color"]
        tolerance = params.get("tolerance", 80)

        for element in elements:
            if keywords:
                content_lower = element.content.lower()
                if not any(keyword.lower() in content_lower for keyword in keywords):
                    continue

            if _color_distance(element.color, target_color) <= tolerance:
                return True

        return False

    def _check_min_elements(self, canvas: Canvas, params: dict[str, Any]) -> bool:
        """Check if the canvas has at least N elements."""

        return canvas.element_count >= params["count"]


def _color_distance(hex1: str, hex2: str) -> float:
    """Euclidean distance between two hex colors in RGB space."""

    r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
    r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
