"""Optional wrappers for MarketCanvasEnv."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np


class DenseRewardWrapper(gymnasium.Wrapper):
    """Provides intermediate reward at every step, not just terminal."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self._last_reward = -1.0

    def reset(self, **kwargs: Any) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        reward, _ = self.unwrapped.compute_reward()
        self._last_reward = reward
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self._last_reward = reward
            return obs, reward, terminated, truncated, info

        current_reward, breakdown = self.unwrapped.compute_reward()
        delta = current_reward - self._last_reward
        self._last_reward = current_reward
        info["reward_breakdown"] = breakdown
        return obs, float(delta), terminated, truncated, info


class PixelObservationWrapper(gymnasium.ObservationWrapper):
    """Adds pixel rendering to the observation space."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        canvas_height = self.unwrapped._canvas_height
        canvas_width = self.unwrapped._canvas_width

        new_spaces = dict(self.observation_space.spaces)
        new_spaces["pixels"] = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(canvas_height, canvas_width, 3),
            dtype=np.uint8,
        )
        self.observation_space = gymnasium.spaces.Dict(new_spaces)

    def observation(self, obs: dict) -> dict:
        pixels = self.unwrapped._renderer.render_to_array(self.unwrapped._canvas)
        return {**obs, "pixels": pixels}


class FlatActionWrapper(gymnasium.ActionWrapper):
    """Flattens Dict action space to MultiDiscrete."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        action_dict = self.env.action_space
        self._keys = list(action_dict.spaces.keys())
        nvec = [action_dict.spaces[key].n for key in self._keys]
        self.action_space = gymnasium.spaces.MultiDiscrete(nvec)

    def action(self, action: np.ndarray) -> dict[str, Any]:
        """Convert MultiDiscrete array back to Dict."""

        return {key: int(action[index]) for index, key in enumerate(self._keys)}
