"""CanvasRenderer — PIL-based stateless canvas renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.types import Element, ElementType
from engine.canvas import Canvas


class CanvasRenderer:
    """Renders a Canvas to a PIL Image or numpy array.

    Stateless apart from a font cache (keyed by size). Takes a Canvas,
    returns an image. No side effects, no internal canvas state.

    Font handling:
        Uses PIL's default bitmap font unless a TTF path is provided.
        The default font does NOT support arbitrary font sizes.
        For proper font-size rendering, pass a .ttf path to the constructor.
    """

    def __init__(self, font_path: Optional[str] = None) -> None:
        self._font_path = font_path
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def render(self, canvas: Canvas) -> Image.Image:
        """Render the canvas state to a PIL Image.

        Rendering order:
        1. Fill canvas with background_color
        2. Iterate elements in list order (back to front)
        3. Draw each element based on its type
        """
        config = canvas.config
        img = Image.new("RGB", (config.width, config.height), config.background_color)
        draw = ImageDraw.Draw(img)

        for element in canvas.get_all_elements():
            if element.type == ElementType.SHAPE:
                self._draw_shape(draw, element)
            elif element.type == ElementType.TEXT:
                self._draw_text(draw, element)
            elif element.type == ElementType.IMAGE:
                self._draw_image_placeholder(draw, element)

        return img

    def render_to_array(self, canvas: Canvas) -> np.ndarray:
        """Render to a numpy array of shape (height, width, 3), dtype uint8.

        This is the format expected by gymnasium's rgb_array render mode
        and by CNN-based observation encoders.
        """
        return np.array(self.render(canvas))

    def save(self, canvas: Canvas, path: str | Path) -> None:
        """Render and save to a file (PNG, JPEG, etc. based on extension)."""
        self.render(canvas).save(str(path))

    # ── Private drawing methods ─────────────────────────────────────────────

    def _draw_shape(self, draw: ImageDraw.ImageDraw, element: Element) -> None:
        """Draw a SHAPE element: filled rectangle + optional text label."""
        left, top, right, bottom = element.bounds
        draw.rectangle([left, top, right, bottom], fill=element.color)

        if element.content:
            font = self._get_font(element.font_size)
            bbox = draw.textbbox((0, 0), element.content, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = element.x + (element.width - text_w) / 2
            text_y = element.y + (element.height - text_h) / 2
            draw.text((text_x, text_y), element.content, fill=element.text_color, font=font)

    def _draw_text(self, draw: ImageDraw.ImageDraw, element: Element) -> None:
        """Draw a TEXT element: text rendered at (x, y) with no background fill."""
        if not element.content:
            return

        font = self._get_font(element.font_size)
        draw.text((element.x, element.y), element.content, fill=element.text_color, font=font)

    def _draw_image_placeholder(self, draw: ImageDraw.ImageDraw, element: Element) -> None:
        """Draw an IMAGE element: colored rectangle with a diagonal cross."""
        left, top, right, bottom = element.bounds
        draw.rectangle([left, top, right, bottom], fill=element.color)

        line_color = self._contrasting_line_color(element.color)
        draw.line([(left, top), (right, bottom)], fill=line_color, width=1)
        draw.line([(right, top), (left, bottom)], fill=line_color, width=1)

        if element.content:
            font = self._get_font(max(12, element.font_size - 4))
            draw.text(
                (element.x + 4, element.y + 4),
                element.content,
                fill=line_color,
                font=font,
            )

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Get a font at the specified size, with caching."""
        if size in self._font_cache:
            return self._font_cache[size]

        if self._font_path:
            try:
                font = ImageFont.truetype(self._font_path, size)
            except (OSError, IOError):
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        self._font_cache[size] = font
        return font

    @staticmethod
    def _contrasting_line_color(hex_color: str) -> str:
        """Return black or white depending on the brightness of the background."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return "#000000" if brightness > 128 else "#FFFFFF"
