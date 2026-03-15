"""Demo script for MarketCanvas-Env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from PIL import Image

from env import register_envs
from env.spaces import (
    ACTION_ADD_IMAGE,
    ACTION_ADD_SHAPE,
    ACTION_ADD_TEXT,
    ACTION_DONE,
    COLOR_PALETTE,
    CONTENT_TEMPLATES,
)


def _color_idx(hex_color: str) -> int:
    """Return the palette index for an exact hex color match."""

    if hex_color not in COLOR_PALETTE:
        raise ValueError(f"Color '{hex_color}' is not in COLOR_PALETTE.")
    return COLOR_PALETTE.index(hex_color)


def _content_idx(text: str) -> int:
    """Return the template index for an exact content match."""

    if text not in CONTENT_TEMPLATES:
        raise ValueError(f"Content '{text}' is not in CONTENT_TEMPLATES.")
    return CONTENT_TEMPLATES.index(text)


def _base_action() -> dict[str, int]:
    """Return a fully-populated default env action dict."""

    return {
        "action_type": ACTION_ADD_TEXT,
        "element_idx": 0,
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 60,
        "color_idx": 0,
        "content_idx": 0,
    }


def _make_action(label: str, **overrides: int) -> tuple[str, dict[str, int]]:
    """Build a labeled action from the default action payload."""

    action = _base_action()
    action.update(overrides)
    return label, action


def _programmatic_actions(prompt: str) -> list[tuple[str, dict[str, int]]]:
    """Return a deterministic action script tailored to the prompt."""

    prompt_lower = prompt.lower()

    if "summer sale" in prompt_lower:
        return [
            _make_action(
                "add headline",
                action_type=ACTION_ADD_TEXT,
                x=140,
                y=60,
                width=520,
                height=80,
                content_idx=_content_idx("Summer Sale"),
            ),
            _make_action(
                "add cta button",
                action_type=ACTION_ADD_SHAPE,
                x=300,
                y=320,
                width=200,
                height=60,
                color_idx=_color_idx("#FFD700"),
                content_idx=_content_idx("Shop Now"),
            ),
            _make_action("done", action_type=ACTION_DONE),
        ]

    if "product launch" in prompt_lower:
        return [
            _make_action(
                "add hero image",
                action_type=ACTION_ADD_IMAGE,
                x=80,
                y=120,
                width=260,
                height=220,
                color_idx=_color_idx("#4169E1"),
                content_idx=_content_idx("Product Name"),
            ),
            _make_action(
                "add product name",
                action_type=ACTION_ADD_TEXT,
                x=390,
                y=140,
                width=260,
                height=70,
                content_idx=_content_idx("Product Name"),
            ),
            _make_action(
                "add launch date",
                action_type=ACTION_ADD_TEXT,
                x=390,
                y=240,
                width=220,
                height=60,
                content_idx=_content_idx("Coming Soon"),
            ),
            _make_action("done", action_type=ACTION_DONE),
        ]

    if "newsletter signup" in prompt_lower:
        return [
            _make_action(
                "add heading",
                action_type=ACTION_ADD_TEXT,
                x=140,
                y=80,
                width=500,
                height=80,
                content_idx=_content_idx("Sign Up"),
            ),
            _make_action(
                "add email field",
                action_type=ACTION_ADD_SHAPE,
                x=160,
                y=260,
                width=300,
                height=60,
                color_idx=_color_idx("#FFFFFF"),
                content_idx=_content_idx("Enter Email"),
            ),
            _make_action(
                "add subscribe button",
                action_type=ACTION_ADD_SHAPE,
                x=490,
                y=260,
                width=150,
                height=60,
                color_idx=_color_idx("#00CED1"),
                content_idx=_content_idx("Subscribe"),
            ),
            _make_action("done", action_type=ACTION_DONE),
        ]

    if "holiday greeting" in prompt_lower:
        return [
            _make_action(
                "add festive image",
                action_type=ACTION_ADD_IMAGE,
                x=90,
                y=110,
                width=240,
                height=220,
                color_idx=_color_idx("#228B22"),
                content_idx=_content_idx("Happy Holidays"),
            ),
            _make_action(
                "add greeting",
                action_type=ACTION_ADD_TEXT,
                x=360,
                y=140,
                width=280,
                height=80,
                content_idx=_content_idx("Happy Holidays"),
            ),
            _make_action(
                "add decorative border",
                action_type=ACTION_ADD_SHAPE,
                x=350,
                y=280,
                width=260,
                height=70,
                color_idx=_color_idx("#FF69B4"),
                content_idx=_content_idx("Happy Holidays"),
            ),
            _make_action("done", action_type=ACTION_DONE),
        ]

    if "flash sale" in prompt_lower:
        return [
            _make_action(
                "add headline",
                action_type=ACTION_ADD_TEXT,
                x=140,
                y=60,
                width=520,
                height=80,
                content_idx=_content_idx("Flash Sale"),
            ),
            _make_action(
                "add discount text",
                action_type=ACTION_ADD_TEXT,
                x=220,
                y=180,
                width=360,
                height=70,
                content_idx=_content_idx("50% Off"),
            ),
            _make_action(
                "add red cta",
                action_type=ACTION_ADD_SHAPE,
                x=300,
                y=320,
                width=200,
                height=60,
                color_idx=_color_idx("#FF0000"),
                content_idx=_content_idx("Order Now"),
            ),
            _make_action("done", action_type=ACTION_DONE),
        ]

    return [
        _make_action(
            "add fallback text",
            action_type=ACTION_ADD_TEXT,
            x=140,
            y=80,
            width=500,
            height=80,
            content_idx=_content_idx("Product Name"),
        ),
        _make_action(
            "add fallback shape",
            action_type=ACTION_ADD_SHAPE,
            x=300,
            y=300,
            width=200,
            height=60,
            color_idx=_color_idx("#CCCCCC"),
            content_idx=_content_idx("Shop Now"),
        ),
        _make_action("done", action_type=ACTION_DONE),
    ]


def demo_programmatic(
    seed: int = 42,
    prompt_text: str | None = None,
    output_path: str | Path = "outputs/demo_programmatic.png",
) -> dict[str, Any]:
    """Run a deterministic scripted demo against the registered env."""

    register_envs()
    env = gym.make("MarketCanvas-v0", render_mode="rgb_array")

    try:
        reset_options = {"prompt_text": prompt_text} if prompt_text is not None else None
        obs, info = env.reset(seed=seed, options=reset_options)
        prompt = info["prompt"]
        print(f"Prompt: {prompt}")
        print(f"Prompt ID: {int(obs['prompt_id'])}")
        print(f"Initial elements: {info['element_count']}")

        actions = _programmatic_actions(prompt)
        steps_executed = 0
        final_reward = 0.0
        reward_breakdown: dict[str, Any] = {}
        terminated = False
        truncated = False
        last_info = info

        for step_index, (label, action) in enumerate(actions, start=1):
            obs, reward, terminated, truncated, last_info = env.step(action)
            steps_executed = step_index
            final_reward = float(reward)
            print(
                f"Step {step_index} | {label} | reward={reward:.3f} | "
                f"elements={last_info['element_count']} | "
                f"terminated={terminated} | truncated={truncated}"
            )
            if terminated or truncated:
                reward_breakdown = dict(last_info.get("reward_breakdown", {}))
                break

        if not (terminated or truncated):
            final_reward, reward_breakdown = env.unwrapped.compute_reward()

        rendered = env.render()
        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rendered).save(resolved_output_path)

        print(f"Final reward: {float(final_reward):.3f}")
        print(f"Reward breakdown: {reward_breakdown}")
        print(f"Saved image: {resolved_output_path}")

        return {
            "prompt": prompt,
            "prompt_id": int(obs["prompt_id"]),
            "steps_executed": steps_executed,
            "terminated": terminated,
            "truncated": truncated,
            "final_reward": float(final_reward),
            "reward_breakdown": reward_breakdown,
            "output_path": str(resolved_output_path),
            "semantic_state": env.unwrapped.get_semantic_state(),
        }
    finally:
        env.close()


def demo_random(
    seed: int = 0,
    total_steps: int = 25,
) -> dict[str, Any]:
    """Run a random-action smoke demo."""

    register_envs()
    env = gym.make("MarketCanvas-v0")

    try:
        _, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        episodes_started = 1
        episodes_finished = 0
        last_info = info

        for step_index in range(1, total_steps + 1):
            _, reward, terminated, truncated, last_info = env.step(env.action_space.sample())
            print(
                f"Random step {step_index} | reward={reward:.3f} | "
                f"elements={last_info['element_count']} | "
                f"terminated={terminated} | truncated={truncated}"
            )
            if terminated or truncated:
                episodes_finished += 1
                _, last_info = env.reset()
                episodes_started += 1

        return {
            "total_steps": total_steps,
            "episodes_started": episodes_started,
            "episodes_finished": episodes_finished,
            "last_prompt": last_info["prompt"],
            "last_element_count": last_info["element_count"],
        }
    finally:
        env.close()


def main() -> None:
    """Run both demo modes."""

    register_envs()
    print("=== Programmatic Demo ===")
    demo_programmatic()
    print()
    print("=== Random Demo ===")
    demo_random()


if __name__ == "__main__":
    main()
