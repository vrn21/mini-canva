"""MarketCanvasEnv — Gymnasium-compliant RL environment for design tasks."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from engine.canvas import Canvas
from engine.renderer import CanvasRenderer
from engine.types import CanvasConfig, ElementType

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
    NUM_ELEMENT_FEATURES,
    build_action_space,
    build_observation_space,
)
from rewards.accessibility import relative_luminance
from rewards.calculator import RewardCalculator
from rewards.prompts import PromptBank, TargetPrompt

# Minimum element dimension to count as substantive (anti reward-hacking)
_MIN_ELEMENT_SIZE = 20


class MarketCanvasEnv(gymnasium.Env):
    """MarketCanvas-Env: a minimalist 2D design canvas RL environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        canvas_width: int = 800,
        canvas_height: int = 600,
        max_steps: int = 50,
        max_elements: int = 20,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = render_mode
        self._canvas_width = canvas_width
        self._canvas_height = canvas_height
        self.max_steps = max_steps
        self.max_elements = max_elements

        self._config = CanvasConfig(
            width=canvas_width,
            height=canvas_height,
            max_elements=max_elements,
        )
        self._canvas = Canvas(self._config)
        self._renderer = CanvasRenderer()

        self._reward_calc = RewardCalculator(weights=reward_weights)
        self._prompt_bank = PromptBank()

        self.observation_space = build_observation_space(
            max_elements, num_prompts=len(self._prompt_bank.PROMPTS)
        )
        self.action_space = build_action_space(
            max_elements, canvas_width=canvas_width, canvas_height=canvas_height
        )

        self._current_prompt: TargetPrompt | None = None
        self._current_prompt_id: int = 0
        self._step_count = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode."""

        super().reset(seed=seed)
        options = options or {}

        self._canvas.clear()
        self._step_count = 0
        self._current_prompt_id, self._current_prompt = self._resolve_prompt(options)

        return self._get_obs(), self._get_info()

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one action on the canvas."""

        action_result = self._execute_action(action)
        self._step_count += 1

        terminated = int(action["action_type"]) == ACTION_DONE
        truncated = self._step_count >= self.max_steps

        reward_breakdown: dict[str, Any] = {}
        if terminated or truncated:
            reward, reward_breakdown = self.compute_reward()
        else:
            reward = 0.0

        info = self._get_info(terminal=terminated or truncated)
        info["reward_breakdown"] = reward_breakdown
        info["action_result"] = action_result

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the canvas to an RGB numpy array."""

        if self.render_mode == "rgb_array":
            return self._renderer.render_to_array(self._canvas)
        return None

    def close(self) -> None:
        """Release environment resources."""

    def get_semantic_state(self) -> dict[str, Any]:
        """Full JSON state for MCP/LLM agents."""

        state = self._canvas.to_dict()
        state["target_prompt"] = self._current_prompt.text if self._current_prompt else ""
        state["step_count"] = self._step_count
        state["max_steps"] = self.max_steps
        state["spatial_relationships"] = self._build_spatial_relationships()
        return state

    def compute_reward(self) -> tuple[float, dict[str, Any]]:
        """Compute the reward for the current canvas state."""

        prompt = self._current_prompt
        if prompt is None:
            raise RuntimeError("Environment must be reset before computing reward")

        return self._reward_calc.calculate(
            self._canvas,
            prompt,
            self._step_count,
            self.max_steps,
        )

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Parse and apply an action to the canvas."""

        action_type = int(action["action_type"])

        if action_type == ACTION_ADD_TEXT:
            return self._action_add_element(action, ElementType.TEXT)
        if action_type == ACTION_ADD_SHAPE:
            return self._action_add_element(action, ElementType.SHAPE)
        if action_type == ACTION_ADD_IMAGE:
            return self._action_add_element(action, ElementType.IMAGE)
        if action_type == ACTION_MOVE:
            return self._action_move(action)
        if action_type == ACTION_RECOLOR:
            return self._action_recolor(action)
        if action_type == ACTION_REMOVE:
            return self._action_remove(action)
        if action_type == ACTION_DONE:
            return {"action": "done"}

        return {"action": "unknown", "success": False}

    def _action_add_element(
        self, action: dict[str, Any], element_type: ElementType
    ) -> dict[str, Any]:
        """Add an element to the canvas."""

        x = int(action["x"])
        y = int(action["y"])
        width = max(_MIN_ELEMENT_SIZE, int(action["width"]))
        height = max(_MIN_ELEMENT_SIZE, int(action["height"]))
        color_idx = int(action["color_idx"]) % len(COLOR_PALETTE)
        content_idx = int(action["content_idx"]) % len(CONTENT_TEMPLATES)
        color = COLOR_PALETTE[color_idx]

        if element_type == ElementType.SHAPE:
            luminance = relative_luminance(color)
            text_color = "#000000" if luminance > 0.179 else "#FFFFFF"
        else:
            text_color = "#000000"

        element = self._canvas.add_element(
            element_type=element_type,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            text_color=text_color,
            content=CONTENT_TEMPLATES[content_idx],
            font_size=24 if element_type == ElementType.TEXT else 16,
        )

        return {
            "action": f"add_{element_type.value.lower()}",
            "success": element is not None,
            "element_id": element.id if element else None,
        }

    def _action_move(self, action: dict[str, Any]) -> dict[str, Any]:
        """Move an existing element."""

        element_id = self._idx_to_element_id(int(action["element_idx"]))
        if element_id is None:
            return {"action": "move", "success": False}

        success = self._canvas.move_element(element_id, int(action["x"]), int(action["y"]))
        return {"action": "move", "success": success, "element_id": element_id}

    def _action_recolor(self, action: dict[str, Any]) -> dict[str, Any]:
        """Change an element's color. For shapes with content, also updates text_color."""

        element_id = self._idx_to_element_id(int(action["element_idx"]))
        if element_id is None:
            return {"action": "recolor", "success": False}

        color_idx = int(action["color_idx"]) % len(COLOR_PALETTE)
        new_color = COLOR_PALETTE[color_idx]
        updates: dict[str, Any] = {"color": new_color}

        element = self._canvas.get_element(element_id)
        if element is not None and element.type == ElementType.SHAPE and element.content:
            luminance = relative_luminance(new_color)
            updates["text_color"] = "#000000" if luminance > 0.179 else "#FFFFFF"

        success = self._canvas.update_element(element_id, **updates)
        return {"action": "recolor", "success": success, "element_id": element_id}

    def _action_remove(self, action: dict[str, Any]) -> dict[str, Any]:
        """Remove an element from the canvas."""

        element_id = self._idx_to_element_id(int(action["element_idx"]))
        if element_id is None:
            return {"action": "remove", "success": False}

        success = self._canvas.remove_element(element_id)
        return {"action": "remove", "success": success, "element_id": element_id}

    def _get_obs(self) -> dict[str, Any]:
        """Build a Gymnasium-compatible observation dict."""

        elements = self._canvas.get_all_elements()
        features = np.zeros((self.max_elements, NUM_ELEMENT_FEATURES), dtype=np.float32)
        mask = np.zeros(self.max_elements, dtype=np.int8)

        canvas_width = float(self._canvas_width)
        canvas_height = float(self._canvas_height)

        for index, element in enumerate(elements[: self.max_elements]):
            type_vec = [0.0, 0.0, 0.0]
            if element.type == ElementType.TEXT:
                type_vec[0] = 1.0
            elif element.type == ElementType.SHAPE:
                type_vec[1] = 1.0
            else:
                type_vec[2] = 1.0

            norm_x = max(0.0, min(1.0, element.x / canvas_width))
            norm_y = max(0.0, min(1.0, element.y / canvas_height))
            norm_w = max(0.0, min(1.0, element.width / canvas_width))
            norm_h = max(0.0, min(1.0, element.height / canvas_height))
            color_r, color_g, color_b = _hex_to_floats(element.color)
            text_r, text_g, text_b = _hex_to_floats(element.text_color)
            norm_font_size = min(1.0, element.font_size / 72.0)
            has_content = 1.0 if element.content else 0.0

            features[index] = [
                *type_vec,
                norm_x,
                norm_y,
                norm_w,
                norm_h,
                color_r,
                color_g,
                color_b,
                text_r,
                text_g,
                text_b,
                norm_font_size,
                has_content,
            ]
            mask[index] = 1

        step_fraction = np.array(
            [min(1.0, self._step_count / self.max_steps) if self.max_steps > 0 else 0.0],
            dtype=np.float32,
        )
        return {
            "elements": features,
            "element_mask": mask,
            "step_fraction": step_fraction,
            "prompt_id": np.int64(self._current_prompt_id),
        }

    def _get_info(self, terminal: bool = False) -> dict[str, Any]:
        """Build info dict with human-readable debug data."""

        info: dict[str, Any] = {
            "element_count": self._canvas.element_count,
            "step_count": self._step_count,
            "prompt": self._current_prompt.text if self._current_prompt else "",
        }
        if terminal:
            info["semantic_state"] = self.get_semantic_state()
        return info

    def _resolve_prompt(self, options: dict[str, Any]) -> tuple[int, TargetPrompt]:
        """Resolve the episode prompt from reset options or RNG sampling."""

        prompt_id = options.get("prompt_id")
        prompt_text = options.get("prompt_text")

        if prompt_id is not None and prompt_text is not None:
            raise ValueError("Provide at most one of prompt_id or prompt_text in reset options.")

        prompts = self._prompt_bank.PROMPTS

        if prompt_id is not None:
            prompt_idx = int(prompt_id)
            if not 0 <= prompt_idx < len(prompts):
                raise ValueError(f"prompt_id must be in [0, {len(prompts) - 1}]")
            return prompt_idx, prompts[prompt_idx]

        if prompt_text is not None:
            for idx, prompt in enumerate(prompts):
                if prompt.text == prompt_text:
                    return idx, prompt
            raise ValueError("prompt_text must match one of PromptBank.PROMPTS exactly.")

        prompt_idx = int(self.np_random.integers(0, len(prompts)))
        return prompt_idx, prompts[prompt_idx]

    def _build_spatial_relationships(self) -> list[dict[str, Any]]:
        """Build explicit pairwise spatial relationships for semantic-state consumers."""

        elements = self._canvas.get_all_elements()
        relationships: list[dict[str, Any]] = []

        for i, element_a in enumerate(elements):
            ax1, ay1, ax2, ay2 = element_a.bounds
            acx, acy = element_a.center
            for element_b in elements[i + 1 :]:
                bx1, by1, bx2, by2 = element_b.bounds
                bcx, bcy = element_b.center

                overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
                overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
                overlap_area = int(overlap_x * overlap_y)

                relationships.append(
                    {
                        "element_a": element_a.id,
                        "element_b": element_b.id,
                        "a_left_of_b": ax2 <= bx1,
                        "a_right_of_b": ax1 >= bx2,
                        "a_above_b": ay2 <= by1,
                        "a_below_b": ay1 >= by2,
                        "overlaps": overlap_area > 0,
                        "overlap_area": overlap_area,
                        "a_contains_b": ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2,
                        "b_contains_a": bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2,
                        "center_dx": float(bcx - acx),
                        "center_dy": float(bcy - acy),
                    }
                )

        return relationships

    def _idx_to_element_id(self, idx: int) -> str | None:
        """Convert an action's element_idx to an actual element ID."""

        elements = self._canvas.get_all_elements()
        if 0 <= idx < len(elements):
            return elements[idx].id
        return None


def _hex_to_floats(hex_color: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' to (r, g, b) floats in [0.0, 1.0]."""

    return (
        int(hex_color[1:3], 16) / 255.0,
        int(hex_color[3:5], 16) / 255.0,
        int(hex_color[5:7], 16) / 255.0,
    )
