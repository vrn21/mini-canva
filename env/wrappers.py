"""Optional wrappers for MarketCanvasEnv."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from env.spaces import DEFAULT_PIXEL_SIZE


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

        current_reward, breakdown = self.unwrapped.compute_reward()
        delta = current_reward - self._last_reward
        self._last_reward = current_reward
        info["reward_breakdown"] = breakdown
        return obs, float(delta), terminated, truncated, info


class PixelObservationWrapper(gymnasium.ObservationWrapper):
    """Adds training-resolution pixel rendering to the observation space.

    By default this preserves the semantic observation dict and appends a
    `pixels` key. When `include_semantic=False`, the wrapper returns only the
    rendered RGB array for pure-vision policies.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        size: tuple[int, int] = DEFAULT_PIXEL_SIZE,
        include_semantic: bool = True,
        pixels_key: str = "pixels",
    ) -> None:
        super().__init__(env)
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("size must contain positive width and height")
        if include_semantic and not isinstance(self.observation_space, gymnasium.spaces.Dict):
            raise ValueError("include_semantic=True requires a Dict observation space")

        self._size = (int(width), int(height))
        self._include_semantic = include_semantic
        self._pixels_key = pixels_key
        pixel_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )

        if include_semantic:
            new_spaces = dict(self.observation_space.spaces)
            new_spaces[pixels_key] = pixel_space
            self.observation_space = gymnasium.spaces.Dict(new_spaces)
        else:
            self.observation_space = pixel_space

    def observation(self, obs: dict[str, Any]) -> dict[str, Any] | np.ndarray:
        pixels = self.unwrapped._renderer.render_to_array(self.unwrapped._canvas, size=self._size)
        if self._include_semantic:
            return {**obs, self._pixels_key: pixels}
        return pixels


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
