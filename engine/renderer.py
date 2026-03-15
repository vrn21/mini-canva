"""CanvasRenderer — PIL-based stateless canvas renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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

    def render(
        self,
        canvas: Canvas,
        size: tuple[int, int] | None = None,
        resample: Image.Resampling = Image.Resampling.BILINEAR,
        overlay: dict[str, Any] | None = None,
    ) -> Image.Image:
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

        if size is not None:
            target_width, target_height = size
            if target_width <= 0 or target_height <= 0:
                raise ValueError("size must contain positive width and height")
            img = img.resize((target_width, target_height), resample=resample)

        if overlay is not None:
            overlay_draw = ImageDraw.Draw(img)
            self._draw_overlay(overlay_draw, canvas, img.size, overlay)

        return img

    def render_to_array(
        self,
        canvas: Canvas,
        size: tuple[int, int] | None = None,
        resample: Image.Resampling = Image.Resampling.BILINEAR,
        overlay: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Render to a numpy array of shape (height, width, 3), dtype uint8.

        This is the format expected by gymnasium's rgb_array render mode
        and by CNN-based observation encoders.
        """
        return np.asarray(
            self.render(canvas, size=size, resample=resample, overlay=overlay),
            dtype=np.uint8,
        )

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

    def _draw_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        canvas: Canvas,
        image_size: tuple[int, int],
        overlay: dict[str, Any],
    ) -> None:
        """Draw low-level interaction state on top of the rendered canvas."""

        image_width, image_height = image_size
        canvas_width = max(1, canvas.config.width)
        canvas_height = max(1, canvas.config.height)
        scale_x = image_width / canvas_width
        scale_y = image_height / canvas_height

        selected_id = overlay.get("selected_element_id")
        focused_id = overlay.get("focused_element_id")
        if selected_id:
            element = canvas.get_element(str(selected_id))
            if element is not None:
                self._draw_scaled_outline(draw, element, scale_x, scale_y, "#00D1FF", width=2)
        if focused_id:
            element = canvas.get_element(str(focused_id))
            if element is not None:
                self._draw_scaled_outline(draw, element, scale_x, scale_y, "#FF4FD8", width=1)

        cursor = overlay.get("cursor")
        if isinstance(cursor, dict):
            cursor_x = int(round(float(cursor.get("x", 0)) * scale_x))
            cursor_y = int(round(float(cursor.get("y", 0)) * scale_y))
            self._draw_cursor(draw, cursor_x, cursor_y, image_width, image_height)

        active_tool = overlay.get("active_tool")
        if isinstance(active_tool, str):
            self._draw_tool_indicator(draw, active_tool)

    @staticmethod
    def _draw_scaled_outline(
        draw: ImageDraw.ImageDraw,
        element: Element,
        scale_x: float,
        scale_y: float,
        color: str,
        *,
        width: int,
    ) -> None:
        """Draw a scaled rectangle outline around an element."""

        left, top, right, bottom = element.bounds
        scaled_bounds = [
            int(round(left * scale_x)),
            int(round(top * scale_y)),
            max(int(round(left * scale_x)) + 1, int(round(right * scale_x))),
            max(int(round(top * scale_y)) + 1, int(round(bottom * scale_y))),
        ]
        draw.rectangle(scaled_bounds, outline=color, width=width)

    @staticmethod
    def _draw_cursor(
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        image_width: int,
        image_height: int,
    ) -> None:
        """Draw a visible cursor crosshair."""

        arm = 6
        left = max(0, x - arm)
        right = min(image_width - 1, x + arm)
        top = max(0, y - arm)
        bottom = min(image_height - 1, y + arm)

        draw.line([(left, y), (right, y)], fill="#FF3B30", width=1)
        draw.line([(x, top), (x, bottom)], fill="#FF3B30", width=1)
        draw.ellipse(
            [max(0, x - 2), max(0, y - 2), min(image_width - 1, x + 2), min(image_height - 1, y + 2)],
            outline="#FFFFFF",
            width=1,
        )

    @staticmethod
    def _draw_tool_indicator(draw: ImageDraw.ImageDraw, active_tool: str) -> None:
        """Draw a compact tool legend in the top-left corner."""

        tool_colors = {
            "select": "#708090",
            "text": "#00CED1",
            "shape": "#FFD700",
            "image": "#4169E1",
        }
        start_x = 6
        start_y = 6
        box_size = 8
        gap = 4

        for index, tool_name in enumerate(("select", "text", "shape", "image")):
            left = start_x + index * (box_size + gap)
            top = start_y
            right = left + box_size
            bottom = top + box_size
            fill = tool_colors[tool_name]
            outline = "#FFFFFF" if tool_name == active_tool else "#000000"
            border = 2 if tool_name == active_tool else 1
            draw.rectangle([left, top, right, bottom], fill=fill, outline=outline, width=border)

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
