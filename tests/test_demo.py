"""Tests for the demo module."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

import demo
from env import register_envs
from env.spaces import ACTION_DONE


class TestProgrammaticDemo:
    def test_demo_programmatic_returns_summary(self, tmp_path: Path):
        result = demo.demo_programmatic(
            output_path=tmp_path / "programmatic.png",
            pixel_output_path=tmp_path / "programmatic_pixels.png",
        )

        assert result["prompt"]
        assert isinstance(result["prompt_id"], int)
        assert result["steps_executed"] >= 1
        assert isinstance(result["final_reward"], float)
        assert result["action_interface"] == "semantic"
        assert result["observation"]["observation_mode"] == "semantic+pixels"
        assert result["observation"]["pixel_shape"] == [96, 128, 3]
        assert "target_prompt" in result["semantic_state"]

    def test_demo_programmatic_writes_output_image(self, tmp_path: Path):
        output_path = tmp_path / "programmatic.png"
        pixel_output_path = tmp_path / "programmatic_pixels.png"
        result = demo.demo_programmatic(
            output_path=output_path,
            pixel_output_path=pixel_output_path,
        )

        assert output_path.exists()
        assert pixel_output_path.exists()
        assert result["output_path"] == str(output_path.resolve())
        assert result["pixel_output_path"] == str(pixel_output_path.resolve())

    def test_demo_programmatic_is_deterministic_for_fixed_seed(self, tmp_path: Path):
        first = demo.demo_programmatic(
            seed=42,
            output_path=tmp_path / "first.png",
            pixel_output_path=tmp_path / "first_pixels.png",
        )
        second = demo.demo_programmatic(
            seed=42,
            output_path=tmp_path / "second.png",
            pixel_output_path=tmp_path / "second_pixels.png",
        )

        assert first["prompt"] == second["prompt"]
        assert first["prompt_id"] == second["prompt_id"]
        assert first["steps_executed"] == second["steps_executed"]
        assert first["terminated"] == second["terminated"]
        assert first["truncated"] == second["truncated"]
        assert first["final_reward"] == second["final_reward"]

    def test_demo_programmatic_semantic_state_has_target_prompt(self, tmp_path: Path):
        result = demo.demo_programmatic(
            output_path=tmp_path / "semantic.png",
            pixel_output_path=tmp_path / "semantic_pixels.png",
        )

        assert result["semantic_state"]["target_prompt"] == result["prompt"]

    def test_demo_programmatic_accepts_mock_target_prompt(self, tmp_path: Path):
        prompt_text = (
            "Create a Summer Sale email banner with a headline, "
            "a yellow CTA button, and good contrast"
        )
        result = demo.demo_programmatic(
            prompt_text=prompt_text,
            output_path=tmp_path / "summer-sale.png",
            pixel_output_path=tmp_path / "summer-sale-pixels.png",
        )

        assert result["prompt"] == prompt_text
        assert result["prompt_id"] == 0

    def test_demo_programmatic_can_run_pixels_only(self, tmp_path: Path):
        result = demo.demo_programmatic(
            output_path=tmp_path / "pixels-only-full.png",
            pixel_output_path=tmp_path / "pixels-only-rollout.png",
            observation_mode="pixels",
        )

        assert result["observation"]["observation_mode"] == "pixels"
        assert result["observation"]["semantic_keys"] is None
        assert result["pixel_output_path"]


class TestLowLevelDemo:
    def test_demo_low_level_returns_summary(self, tmp_path: Path):
        result = demo.demo_low_level(
            output_path=tmp_path / "low-level.png",
            pixel_output_path=tmp_path / "low-level-pixels.png",
        )

        assert result["prompt"]
        assert result["action_interface"] == "low_level"
        assert result["observation"]["observation_mode"] == "semantic+pixels"
        assert result["observation"]["pixel_shape"] == [96, 128, 3]
        assert "cursor" in result["interaction"]
        assert "interaction" in result["semantic_state"]

    def test_demo_low_level_writes_images(self, tmp_path: Path):
        output_path = tmp_path / "low-level.png"
        pixel_output_path = tmp_path / "low-level-pixels.png"
        result = demo.demo_low_level(
            output_path=output_path,
            pixel_output_path=pixel_output_path,
        )

        assert output_path.exists()
        assert pixel_output_path.exists()
        assert result["output_path"] == str(output_path.resolve())
        assert result["pixel_output_path"] == str(pixel_output_path.resolve())


class TestRandomDemo:
    def test_demo_random_returns_summary(self):
        result = demo.demo_random(seed=0, total_steps=10)

        assert result["total_steps"] == 10
        assert result["episodes_started"] >= 1
        assert result["episodes_finished"] >= 0
        assert result["last_prompt"]

    def test_demo_random_runs_without_crashing(self):
        result = demo.demo_random(seed=1, total_steps=12)

        assert result["last_element_count"] >= 0

    def test_demo_random_can_cross_episode_boundaries(self, monkeypatch):
        register_envs()
        env = gym.make("MarketCanvas-v0")

        done_action = {
            "action_type": ACTION_DONE,
            "element_idx": 0,
            "x": 0,
            "y": 0,
            "width": 20,
            "height": 20,
            "color_idx": 0,
            "content_idx": 0,
        }

        monkeypatch.setattr(env.action_space, "sample", lambda: dict(done_action))
        monkeypatch.setattr(demo.gym, "make", lambda *args, **kwargs: env)

        result = demo.demo_random(seed=0, total_steps=5)

        assert result["episodes_finished"] >= 1
        assert result["episodes_started"] > 1


class TestMain:
    def test_main_runs_both_demos(self, monkeypatch):
        calls: list[str] = []

        monkeypatch.setattr(
            demo,
            "demo_programmatic",
            lambda *args, **kwargs: calls.append("programmatic") or {},
        )
        monkeypatch.setattr(
            demo,
            "demo_low_level",
            lambda *args, **kwargs: calls.append("low_level") or {},
        )
        monkeypatch.setattr(
            demo,
            "demo_random",
            lambda *args, **kwargs: calls.append("random") or {},
        )

        demo.main()

        assert calls == ["programmatic", "low_level", "random"]
