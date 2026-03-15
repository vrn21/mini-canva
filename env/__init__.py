"""RL Environment — Gymnasium-compliant MarketCanvas environment."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from env.market_canvas_env import MarketCanvasEnv

__all__ = ["MarketCanvasEnv", "register_envs"]


def register_envs() -> None:
    """Register MarketCanvas environments with Gymnasium."""

    if "MarketCanvas-v0" in registry:
        return

    register(
        id="MarketCanvas-v0",
        entry_point="env.market_canvas_env:MarketCanvasEnv",
        max_episode_steps=50,
    )
