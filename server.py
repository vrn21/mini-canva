"""FastMCP server for MarketCanvas-Env."""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from engine.types import ElementType
from env.market_canvas_env import MarketCanvasEnv
from env.spaces import (
    ACTION_ADD_IMAGE,
    ACTION_ADD_SHAPE,
    ACTION_ADD_TEXT,
    ACTION_DONE,
    ACTION_MOVE,
    ACTION_RECOLOR,
    ACTION_REMOVE,
    COLOR_PALETTE,
    CONTENT_TEMPLATES,
)
from rewards.accessibility import relative_luminance

ActionName = Literal[
    "add_text",
    "add_shape",
    "add_image",
    "move",
    "recolor",
    "remove",
    "done",
]

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_MIN_ELEMENT_SIZE = 20


@dataclass
class ServerSession:
    """Single in-process MCP session state."""

    env: MarketCanvasEnv | None = None
    seed: int | None = None
    session_id: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


SESSION = ServerSession()
mcp = FastMCP("MarketCanvas-MCP")


def _require_env(session_id: str) -> MarketCanvasEnv:
    """Return the active environment for a validated session."""

    if SESSION.env is None:
        raise RuntimeError("Environment not initialized. Call initialize_env first.")
    if SESSION.session_id != session_id:
        raise RuntimeError(
            "Invalid or stale session_id. Re-run initialize_env and use the returned session_id."
        )
    return SESSION.env


def _semantic_state(env: MarketCanvasEnv) -> dict[str, Any]:
    """Return the MCP-facing semantic state snapshot."""

    state = env.get_semantic_state()
    state["prompt_id"] = env._current_prompt_id
    state["initialized"] = True
    state["session_id"] = SESSION.session_id
    return state


def _element_id_to_idx(env: MarketCanvasEnv, element_id: str) -> int | None:
    """Resolve a stable element ID to the current action-space index."""

    for index, element in enumerate(env._canvas.get_all_elements()):
        if element.id == element_id:
            return index
    return None


def _validate_hex_color(hex_color: str) -> None:
    """Validate '#RRGGBB' color strings."""

    if not _HEX_COLOR_RE.fullmatch(hex_color):
        raise ValueError(f"Invalid color '{hex_color}'. Expected '#RRGGBB'.")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to integer RGB."""

    return (
        int(hex_color[1:3], 16),
        int(hex_color[3:5], 16),
        int(hex_color[5:7], 16),
    )


def _nearest_color_index(hex_color: str) -> int:
    """Find the nearest palette entry to an arbitrary hex color."""

    _validate_hex_color(hex_color)
    target_r, target_g, target_b = _hex_to_rgb(hex_color)

    best_idx = 0
    best_distance = float("inf")
    for index, candidate in enumerate(COLOR_PALETTE):
        cand_r, cand_g, cand_b = _hex_to_rgb(candidate)
        distance = (
            (target_r - cand_r) ** 2
            + (target_g - cand_g) ** 2
            + (target_b - cand_b) ** 2
        )
        if distance < best_distance:
            best_distance = distance
            best_idx = index

    return best_idx


def _content_index(content: str | None, action_type: ActionName) -> int:
    """Resolve free-form content to the env's template index."""

    if content and content in CONTENT_TEMPLATES:
        return CONTENT_TEMPLATES.index(content)

    if action_type == "add_shape":
        return CONTENT_TEMPLATES.index("Shop Now")
    if action_type == "add_image":
        return CONTENT_TEMPLATES.index("Product Name")
    return CONTENT_TEMPLATES.index("Summer Sale")


def _shape_text_color(fill_color: str) -> str:
    """Choose black or white for readable text on a shape fill."""

    luminance = relative_luminance(fill_color)
    return "#000000" if luminance > 0.179 else "#FFFFFF"


def _default_box(action_type: ActionName) -> tuple[int, int, int, int]:
    """Default geometry for MCP add actions."""

    if action_type == "add_shape":
        return (300, 320, 200, 60)
    if action_type == "add_image":
        return (80, 140, 260, 220)
    return (120, 80, 520, 80)


def _base_action(env: MarketCanvasEnv) -> dict[str, int]:
    """Construct a fully-populated env action dict."""

    return {
        "action_type": ACTION_DONE,
        "element_idx": max(0, env.max_elements - 1),
        "x": 0,
        "y": 0,
        "width": _MIN_ELEMENT_SIZE,
        "height": _MIN_ELEMENT_SIZE,
        "color_idx": 0,
        "content_idx": 0,
    }


def _resolve_add_geometry(
    action_type: ActionName,
    x: int | None,
    y: int | None,
    width: int | None,
    height: int | None,
) -> tuple[int, int, int, int]:
    """Resolve optional MCP geometry to env-compatible dimensions."""

    default_x, default_y, default_w, default_h = _default_box(action_type)
    resolved_x = default_x if x is None else int(x)
    resolved_y = default_y if y is None else int(y)
    resolved_w = max(_MIN_ELEMENT_SIZE, default_w if width is None else int(width))
    resolved_h = max(_MIN_ELEMENT_SIZE, default_h if height is None else int(height))
    return resolved_x, resolved_y, resolved_w, resolved_h


def _patch_added_or_recolored_element(
    env: MarketCanvasEnv,
    *,
    element_id: str,
    action_type: ActionName,
    color: str | None,
    content: str | None,
) -> None:
    """Apply MCP-level exact content/color values after env.step."""

    element = env._canvas.get_element(element_id)
    if element is None:
        return

    updates: dict[str, Any] = {}

    if content is not None:
        updates["content"] = content

    if action_type == "add_text":
        if color is not None:
            updates["text_color"] = color
    elif action_type in {"add_shape", "recolor"} and element.type == ElementType.SHAPE:
        if color is not None:
            updates["color"] = color
            if content is not None or element.content:
                updates["text_color"] = _shape_text_color(color)
    elif action_type in {"add_image", "recolor"} and element.type == ElementType.IMAGE:
        if color is not None:
            updates["color"] = color
    elif action_type == "recolor" and element.type == ElementType.TEXT and color is not None:
        updates["text_color"] = color

    if updates:
        env._canvas.update_element(element_id, **updates)


def _make_action(
    env: MarketCanvasEnv,
    *,
    action_type: ActionName,
    element_id: str | None,
    x: int | None,
    y: int | None,
    width: int | None,
    height: int | None,
    color: str | None,
    content: str | None,
) -> dict[str, int]:
    """Translate an MCP action into the env's discrete action dict."""

    if color is not None:
        _validate_hex_color(color)

    action = _base_action(env)

    if action_type == "done":
        action["action_type"] = ACTION_DONE
        return action

    if action_type in {"add_text", "add_shape", "add_image"}:
        resolved_x, resolved_y, resolved_w, resolved_h = _resolve_add_geometry(
            action_type,
            x,
            y,
            width,
            height,
        )
        action["action_type"] = {
            "add_text": ACTION_ADD_TEXT,
            "add_shape": ACTION_ADD_SHAPE,
            "add_image": ACTION_ADD_IMAGE,
        }[action_type]
        action["x"] = resolved_x
        action["y"] = resolved_y
        action["width"] = resolved_w
        action["height"] = resolved_h
        if color is not None:
            action["color_idx"] = _nearest_color_index(color)
        action["content_idx"] = _content_index(content, action_type)
        return action

    if action_type == "move":
        if element_id is None or x is None or y is None:
            raise ValueError("move requires element_id, x, and y.")
        idx = _element_id_to_idx(env, element_id)
        if idx is None:
            raise LookupError(element_id)
        action["action_type"] = ACTION_MOVE
        action["element_idx"] = idx
        action["x"] = int(x)
        action["y"] = int(y)
        return action

    if action_type == "recolor":
        if element_id is None or color is None:
            raise ValueError("recolor requires element_id and color.")
        idx = _element_id_to_idx(env, element_id)
        if idx is None:
            raise LookupError(element_id)
        action["action_type"] = ACTION_RECOLOR
        action["element_idx"] = idx
        action["color_idx"] = _nearest_color_index(color)
        return action

    if action_type == "remove":
        if element_id is None:
            raise ValueError("remove requires element_id.")
        idx = _element_id_to_idx(env, element_id)
        if idx is None:
            raise LookupError(element_id)
        action["action_type"] = ACTION_REMOVE
        action["element_idx"] = idx
        return action

    raise ValueError(f"Unsupported action_type '{action_type}'.")


def _noop_failure(
    env: MarketCanvasEnv,
    action_type: ActionName,
    *,
    element_id: str | None = None,
) -> dict[str, Any]:
    """Return a structured failure result without mutating env state."""

    current_reward, current_breakdown = env.compute_reward()
    return {
        "canvas_state": _semantic_state(env),
        "reward": 0.0,
        "current_reward": float(current_reward),
        "terminated": False,
        "truncated": False,
        "reward_breakdown": {},
        "current_reward_breakdown": current_breakdown,
        "action_result": {
            "action": action_type,
            "success": False,
            "element_id": element_id,
        },
    }


@mcp.tool
def initialize_env(
    canvas_width: int = 800,
    canvas_height: int = 600,
    max_steps: int = 50,
    max_elements: int = 20,
    seed: int | None = None,
) -> dict[str, Any]:
    """Initialize a new MarketCanvas environment session."""

    with SESSION.lock:
        if canvas_width <= 0 or canvas_height <= 0 or max_steps <= 0 or max_elements <= 0:
            raise ValueError(
                "canvas_width, canvas_height, max_steps, and max_elements must be > 0."
            )

        if SESSION.env is not None:
            SESSION.env.close()

        env = MarketCanvasEnv(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_steps=max_steps,
            max_elements=max_elements,
        )
        obs, info = env.reset(seed=seed)
        SESSION.env = env
        SESSION.seed = seed
        SESSION.session_id = uuid.uuid4().hex

        return {
            "status": "initialized",
            "session_id": SESSION.session_id,
            "prompt": info["prompt"],
            "prompt_id": int(obs["prompt_id"]),
            "canvas_state": _semantic_state(env),
            "element_count": info["element_count"],
            "step_count": info["step_count"],
        }


@mcp.tool
def get_canvas_state(session_id: str) -> dict[str, Any]:
    """Return the current semantic state of the canvas."""

    with SESSION.lock:
        return _semantic_state(_require_env(session_id))


@mcp.tool
def execute_action(
    session_id: str,
    action_type: ActionName,
    element_id: str | None = None,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    color: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    """Execute a semantic design action against the active environment."""

    with SESSION.lock:
        env = _require_env(session_id)

        try:
            action = _make_action(
                env,
                action_type=action_type,
                element_id=element_id,
                x=x,
                y=y,
                width=width,
                height=height,
                color=color,
                content=content,
            )
        except LookupError:
            return _noop_failure(env, action_type, element_id=element_id)

        _, step_reward, terminated, truncated, info = env.step(action)
        action_result = dict(info.get("action_result", {}))

        target_element_id = action_result.get("element_id")
        if action_result.get("success") and target_element_id is not None:
            _patch_added_or_recolored_element(
                env,
                element_id=target_element_id,
                action_type=action_type,
                color=color,
                content=content,
            )

        current_reward, current_breakdown = env.compute_reward()
        reward_breakdown: dict[str, Any] = {}
        reward = float(step_reward)
        if terminated or truncated:
            reward = float(current_reward)
            reward_breakdown = current_breakdown

        return {
            "canvas_state": _semantic_state(env),
            "reward": reward,
            "current_reward": float(current_reward),
            "terminated": terminated,
            "truncated": truncated,
            "reward_breakdown": reward_breakdown,
            "current_reward_breakdown": current_breakdown,
            "action_result": action_result,
        }


@mcp.tool
def get_current_reward(session_id: str) -> dict[str, Any]:
    """Return the current reward for the active canvas without stepping."""

    with SESSION.lock:
        env = _require_env(session_id)
        reward, breakdown = env.compute_reward()
        prompt = env._current_prompt
        return {
            "reward": float(reward),
            "breakdown": breakdown,
            "step_count": env._step_count,
            "max_steps": env.max_steps,
            "prompt": prompt.text if prompt is not None else "",
            "prompt_id": env._current_prompt_id,
            "session_id": SESSION.session_id,
        }


@mcp.tool
def save_canvas(session_id: str, filepath: str = "canvas_output.png") -> dict[str, Any]:
    """Render and save the current canvas as an image file."""

    with SESSION.lock:
        env = _require_env(session_id)
        path = Path(filepath).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        env._renderer.save(env._canvas, path)
        return {
            "status": "saved",
            "path": str(path.resolve()),
            "element_count": env._canvas.element_count,
            "session_id": SESSION.session_id,
        }


@mcp.resource("canvas://state")
def canvas_state_resource() -> str:
    """Read-only JSON snapshot of the active canvas state."""

    with SESSION.lock:
        if SESSION.env is None:
            return json.dumps(
                {
                    "initialized": False,
                    "message": "Call initialize_env first.",
                },
                indent=2,
                sort_keys=True,
            )
        return json.dumps(_semantic_state(SESSION.env), indent=2, sort_keys=True)


if __name__ == "__main__":
    mcp.run()
