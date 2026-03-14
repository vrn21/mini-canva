"""RL Environment — Gymnasium-compliant MarketCanvas environment."""

from __future__ import annotations

from gymnasium.error import Error as GymnasiumError
from gymnasium.envs.registration import register

from env.market_canvas_env import MarketCanvasEnv

__all__ = ["MarketCanvasEnv", "register_envs"]


def register_envs() -> None:
    """Register MarketCanvas environments with Gymnasium."""

    try:
        register(
            id="MarketCanvas-v0",
            entry_point="env.market_canvas_env:MarketCanvasEnv",
            max_episode_steps=50,
        )
    except GymnasiumError as exc:
        if "already registered" not in str(exc):
            raise
