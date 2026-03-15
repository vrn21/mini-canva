"""WCAG 2.1 AA contrast ratio compliance checker."""

from __future__ import annotations

from engine.canvas import Canvas
from engine.types import Element, ElementType

_LARGE_TEXT_MIN_PX = 24


class AccessibilityChecker:
    """Checks WCAG 2.1 AA contrast compliance for text elements."""

    def score(self, canvas: Canvas) -> float:
        """Fraction of text-bearing elements meeting WCAG AA contrast."""

        text_elements = [
            element
            for element in canvas.get_all_elements()
            if element.content and element.type in (ElementType.TEXT, ElementType.SHAPE)
        ]
        if not text_elements:
            return 1.0

        passing = sum(1 for element in text_elements if self._check_contrast(element, canvas))
        return passing / len(text_elements)

    def _check_contrast(self, element: Element, canvas: Canvas) -> bool:
        """Check if an element's text meets WCAG AA contrast."""

        bg_color = self._get_effective_background(element, canvas)
        ratio = contrast_ratio(element.text_color, bg_color)
        # WCAG's relaxed threshold applies to large text (roughly 18pt regular),
        # which maps more closely to ~24 CSS px than 18 px.
        threshold = 3.0 if element.font_size >= _LARGE_TEXT_MIN_PX else 4.5
        return ratio >= threshold

    def _get_effective_background(self, element: Element, canvas: Canvas) -> str:
        """Determine the effective background color behind an element's text."""

        if element.type == ElementType.SHAPE:
            return element.color

        behind = canvas.get_element_behind(element.id)
        if behind is not None:
            return behind.color
        return canvas.config.background_color


def contrast_ratio(color1: str, color2: str) -> float:
    """WCAG 2.1 contrast ratio between two hex colors."""

    luminance_1 = relative_luminance(color1)
    luminance_2 = relative_luminance(color2)
    lighter = max(luminance_1, luminance_2)
    darker = min(luminance_1, luminance_2)
    return (lighter + 0.05) / (darker + 0.05)


def relative_luminance(hex_color: str) -> float:
    """Relative luminance of a color per WCAG 2.1."""

    r_srgb = int(hex_color[1:3], 16) / 255.0
    g_srgb = int(hex_color[3:5], 16) / 255.0
    b_srgb = int(hex_color[5:7], 16) / 255.0

    r = r_srgb / 12.92 if r_srgb <= 0.04045 else ((r_srgb + 0.055) / 1.055) ** 2.4
    g = g_srgb / 12.92 if g_srgb <= 0.04045 else ((g_srgb + 0.055) / 1.055) ** 2.4
    b = b_srgb / 12.92 if b_srgb <= 0.04045 else ((b_srgb + 0.055) / 1.055) ** 2.4

    return 0.2126 * r + 0.7152 * g + 0.0722 * b
