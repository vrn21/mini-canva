"""Minimal low-level interaction model for MarketCanvasEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.canvas import Canvas
from engine.types import ElementType
from rewards.accessibility import relative_luminance

from env.spaces import (
    ACTIVE_TOOL_IMAGE,
    ACTIVE_TOOL_SELECT,
    ACTIVE_TOOL_SHAPE,
    ACTIVE_TOOL_TEXT,
)

_MIN_ELEMENT_SIZE = 20
_DEFAULT_TEXT_SIZE = (320, 60)
_DEFAULT_SHAPE_SIZE = (200, 60)
_DEFAULT_IMAGE_SIZE = (260, 220)
_TOOL_NAMES = {
    ACTIVE_TOOL_SELECT: "select",
    ACTIVE_TOOL_TEXT: "text",
    ACTIVE_TOOL_SHAPE: "shape",
    ACTIVE_TOOL_IMAGE: "image",
}
_TOOL_CODES = {name: code for code, name in _TOOL_NAMES.items()}


def tool_code_to_name(tool_code: int) -> str:
    """Return the stable MCP-facing name for an active tool."""

    if tool_code not in _TOOL_NAMES:
        raise ValueError(f"Unsupported active tool code '{tool_code}'.")
    return _TOOL_NAMES[tool_code]


def tool_name_to_code(tool_name: str) -> int:
    """Return the env-facing integer code for an active tool."""

    if tool_name not in _TOOL_CODES:
        raise ValueError(
            f"Unsupported active tool '{tool_name}'. Expected one of {tuple(_TOOL_CODES)}."
        )
    return _TOOL_CODES[tool_name]


@dataclass
class InteractionState:
    """Serializable UI-interaction state for low-level control."""

    cursor_x: int = 0
    cursor_y: int = 0
    active_tool: int = ACTIVE_TOOL_SELECT
    selected_element_id: str | None = None
    focused_element_id: str | None = None

    def to_dict(self, *, action_interface: str) -> dict[str, Any]:
        """Serialize the current interaction state for semantic/MCP consumers."""

        return {
            "action_interface": action_interface,
            "active_tool": tool_code_to_name(self.active_tool),
            "cursor": {"x": self.cursor_x, "y": self.cursor_y},
            "selected_element_id": self.selected_element_id,
            "focused_element_id": self.focused_element_id,
        }


class LowLevelController:
    """Interprets a tiny set of low-level gestures over the canvas."""

    def __init__(self, canvas_width: int, canvas_height: int) -> None:
        self._canvas_width = int(canvas_width)
        self._canvas_height = int(canvas_height)
        self.state = InteractionState()

    def reset(self) -> None:
        """Reset low-level interaction state for a new episode."""

        self.state = InteractionState()

    def sync_with_canvas(self, canvas: Canvas) -> None:
        """Clear dangling focus/selection references after canvas mutations."""

        if self.state.selected_element_id and not canvas.has_element(self.state.selected_element_id):
            self.state.selected_element_id = None
        if self.state.focused_element_id and not canvas.has_element(self.state.focused_element_id):
            self.state.focused_element_id = None

    def move_cursor(self, x: int, y: int) -> dict[str, Any]:
        """Move the cursor without mutating the canvas."""

        self.state.cursor_x = self._clamp_x(x)
        self.state.cursor_y = self._clamp_y(y)
        return {
            "action": "mouse_move",
            "success": True,
            "cursor": {"x": self.state.cursor_x, "y": self.state.cursor_y},
        }

    def set_active_tool(self, tool: int) -> dict[str, Any]:
        """Switch the active insertion/selection tool."""

        tool_name = tool_code_to_name(int(tool))
        self.state.active_tool = int(tool)
        return {
            "action": "set_active_tool",
            "success": True,
            "active_tool": tool_name,
        }

    def mouse_click(self, canvas: Canvas) -> dict[str, Any]:
        """Interpret a click at the current cursor position."""

        self.sync_with_canvas(canvas)
        cursor_x = self.state.cursor_x
        cursor_y = self.state.cursor_y

        if self.state.active_tool == ACTIVE_TOOL_SELECT:
            hits = canvas.get_elements_at(cursor_x, cursor_y)
            if not hits:
                cleared = self.state.selected_element_id is not None or self.state.focused_element_id is not None
                self.state.selected_element_id = None
                self.state.focused_element_id = None
                return {
                    "action": "mouse_click",
                    "success": cleared,
                    "effect": "clear_selection",
                }

            element = hits[0]
            self.state.selected_element_id = element.id
            self.state.focused_element_id = (
                element.id if element.type in {ElementType.TEXT, ElementType.SHAPE} else None
            )
            return {
                "action": "mouse_click",
                "success": True,
                "effect": "select",
                "element_id": element.id,
            }

        element = self._create_element_at_cursor(canvas)
        return {
            "action": "mouse_click",
            "success": element is not None,
            "effect": "create" if element is not None else "canvas_full",
            "element_id": element.id if element is not None else None,
        }

    def mouse_drag(self, canvas: Canvas, x1: int, y1: int, x2: int, y2: int) -> dict[str, Any]:
        """Interpret a complete drag gesture."""

        self.sync_with_canvas(canvas)
        start_x = self._clamp_x(x1)
        start_y = self._clamp_y(y1)
        end_x = self._clamp_x(x2)
        end_y = self._clamp_y(y2)
        self.state.cursor_x = end_x
        self.state.cursor_y = end_y

        if self.state.active_tool == ACTIVE_TOOL_SELECT:
            hits = canvas.get_elements_at(start_x, start_y)
            if not hits:
                return {
                    "action": "mouse_drag",
                    "success": False,
                    "effect": "no_target",
                }

            element = hits[0]
            delta_x = end_x - start_x
            delta_y = end_y - start_y
            success = canvas.move_element(element.id, element.x + delta_x, element.y + delta_y)
            if success:
                self.state.selected_element_id = element.id
                self.state.focused_element_id = (
                    element.id if element.type in {ElementType.TEXT, ElementType.SHAPE} else None
                )
            return {
                "action": "mouse_drag",
                "success": success,
                "effect": "move",
                "element_id": element.id,
            }

        left = min(start_x, end_x)
        top = min(start_y, end_y)
        width = max(_MIN_ELEMENT_SIZE, abs(end_x - start_x))
        height = max(_MIN_ELEMENT_SIZE, abs(end_y - start_y))
        left, top = self._clamp_box_origin(left, top, width, height)
        element = self._create_element(
            canvas,
            x=left,
            y=top,
            width=width,
            height=height,
        )
        return {
            "action": "mouse_drag",
            "success": element is not None,
            "effect": "create" if element is not None else "canvas_full",
            "element_id": element.id if element is not None else None,
        }

    def keyboard_type(self, canvas: Canvas, text: str) -> dict[str, Any]:
        """Replace content on the currently focused editable element."""

        self.sync_with_canvas(canvas)
        element_id = self.state.focused_element_id
        if element_id is None:
            return {
                "action": "keyboard_type",
                "success": False,
                "effect": "no_focus",
            }

        element = canvas.get_element(element_id)
        if element is None:
            self.state.focused_element_id = None
            self.state.selected_element_id = None
            return {
                "action": "keyboard_type",
                "success": False,
                "effect": "stale_focus",
            }

        if element.type not in {ElementType.TEXT, ElementType.SHAPE}:
            return {
                "action": "keyboard_type",
                "success": False,
                "effect": "not_text_editable",
                "element_id": element_id,
            }

        success = canvas.update_element(element_id, content=str(text))
        return {
            "action": "keyboard_type",
            "success": success,
            "effect": "replace_content" if success else "update_failed",
            "element_id": element_id,
        }

    def selected_element_index(self, canvas: Canvas, max_elements: int) -> int:
        """Return selected element index or sentinel."""

        return self._element_index_or_sentinel(canvas, self.state.selected_element_id, max_elements)

    def focused_element_index(self, canvas: Canvas, max_elements: int) -> int:
        """Return focused element index or sentinel."""

        return self._element_index_or_sentinel(canvas, self.state.focused_element_id, max_elements)

    def _create_element_at_cursor(self, canvas: Canvas):
        if self.state.active_tool == ACTIVE_TOOL_TEXT:
            width, height = _DEFAULT_TEXT_SIZE
        elif self.state.active_tool == ACTIVE_TOOL_SHAPE:
            width, height = _DEFAULT_SHAPE_SIZE
        elif self.state.active_tool == ACTIVE_TOOL_IMAGE:
            width, height = _DEFAULT_IMAGE_SIZE
        else:
            return None

        x = self.state.cursor_x - width // 2
        y = self.state.cursor_y - height // 2
        x, y = self._clamp_box_origin(x, y, width, height)
        return self._create_element(canvas, x=x, y=y, width=width, height=height)

    def _create_element(self, canvas: Canvas, *, x: int, y: int, width: int, height: int):
        if self.state.active_tool == ACTIVE_TOOL_TEXT:
            element_type = ElementType.TEXT
            color = "#CCCCCC"
            text_color = "#000000"
            content = "Summer Sale"
            font_size = 24
        elif self.state.active_tool == ACTIVE_TOOL_SHAPE:
            element_type = ElementType.SHAPE
            color = "#FFD700"
            text_color = "#000000" if relative_luminance(color) > 0.179 else "#FFFFFF"
            content = "Shop Now"
            font_size = 16
        elif self.state.active_tool == ACTIVE_TOOL_IMAGE:
            element_type = ElementType.IMAGE
            color = "#4169E1"
            text_color = "#000000"
            content = "Product Name"
            font_size = 16
        else:
            return None

        element = canvas.add_element(
            element_type=element_type,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            text_color=text_color,
            content=content,
            font_size=font_size,
        )
        if element is not None:
            self.state.selected_element_id = element.id
            self.state.focused_element_id = (
                element.id if element.type in {ElementType.TEXT, ElementType.SHAPE} else None
            )
        return element

    def _element_index_or_sentinel(
        self,
        canvas: Canvas,
        element_id: str | None,
        max_elements: int,
    ) -> int:
        if element_id is None:
            return max_elements

        for index, element in enumerate(canvas.get_all_elements()[:max_elements]):
            if element.id == element_id:
                return index
        return max_elements

    def _clamp_box_origin(self, x: int, y: int, width: int, height: int) -> tuple[int, int]:
        max_x = max(0, self._canvas_width - width)
        max_y = max(0, self._canvas_height - height)
        return (
            max(0, min(max_x, int(x))),
            max(0, min(max_y, int(y))),
        )

    def _clamp_x(self, x: int) -> int:
        return max(0, min(self._canvas_width - 1, int(x)))

    def _clamp_y(self, y: int) -> int:
        return max(0, min(self._canvas_height - 1, int(y)))
