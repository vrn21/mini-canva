"""Tests for the Gymnasium RL environment."""

import gymnasium
import numpy as np

from env.market_canvas_env import MarketCanvasEnv
from engine.types import ElementType
from env.spaces import (
    ACTION_ADD_SHAPE,
    ACTION_ADD_TEXT,
    ACTION_DONE,
    ACTION_MOVE,
    ACTION_RECOLOR,
    ACTION_REMOVE,
    DEFAULT_PIXEL_SIZE,
    NUM_ELEMENT_FEATURES,
    OBSERVATION_MODE_PIXELS,
    OBSERVATION_MODE_SEMANTIC,
    OBSERVATION_MODE_SEMANTIC_PIXELS,
)
from env.wrappers import (
    DenseRewardWrapper,
    FlatActionWrapper,
    PixelObservationWrapper,
)


class TestEnvCreation:
    def test_default_construction(self):
        env = MarketCanvasEnv()
        assert env.observation_space is not None
        assert env.action_space is not None
        env.close()

    def test_custom_params(self):
        env = MarketCanvasEnv(canvas_width=400, canvas_height=300, max_steps=25, max_elements=10)
        assert env.max_steps == 25
        assert env.max_elements == 10
        env.close()

    def test_observation_space_shape(self):
        env = MarketCanvasEnv(max_elements=10)
        obs_space = env.observation_space
        assert obs_space["elements"].shape == (10, NUM_ELEMENT_FEATURES)
        assert obs_space["element_mask"].n == 10
        assert obs_space["step_fraction"].shape == (1,)
        env.close()

    def test_semantic_plus_pixels_mode_observation_space(self):
        env = MarketCanvasEnv(observation_mode=OBSERVATION_MODE_SEMANTIC_PIXELS)
        assert env.observation_space["elements"].shape == (20, NUM_ELEMENT_FEATURES)
        assert env.observation_space["pixels"].shape == (DEFAULT_PIXEL_SIZE[1], DEFAULT_PIXEL_SIZE[0], 3)
        env.close()

    def test_pixels_only_mode_observation_space(self):
        env = MarketCanvasEnv(observation_mode=OBSERVATION_MODE_PIXELS)
        assert env.observation_space.shape == (DEFAULT_PIXEL_SIZE[1], DEFAULT_PIXEL_SIZE[0], 3)
        env.close()


class TestEnvReset:
    def test_reset_returns_obs_and_info(self):
        env = MarketCanvasEnv()
        obs, info = env.reset(seed=42)
        assert "elements" in obs
        assert "element_mask" in obs
        assert "step_fraction" in obs
        assert "prompt" in info
        assert info["prompt"]
        env.close()

    def test_reset_clears_canvas(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        action["action_type"] = ACTION_ADD_TEXT
        env.step(action)
        _, info = env.reset(seed=42)
        assert info["element_count"] == 0
        env.close()

    def test_reset_is_seeded(self):
        env1 = MarketCanvasEnv()
        env2 = MarketCanvasEnv()
        _, info1 = env1.reset(seed=42)
        _, info2 = env2.reset(seed=42)
        assert info1["prompt"] == info2["prompt"]
        env1.close()
        env2.close()

    def test_observation_in_space(self):
        env = MarketCanvasEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_returns_semantic_plus_pixels_observation(self):
        env = MarketCanvasEnv(observation_mode=OBSERVATION_MODE_SEMANTIC_PIXELS)
        obs, _ = env.reset(seed=42)
        assert "pixels" in obs
        assert obs["pixels"].shape == (DEFAULT_PIXEL_SIZE[1], DEFAULT_PIXEL_SIZE[0], 3)
        assert obs["pixels"].dtype == np.uint8
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_returns_pixels_only_observation(self):
        env = MarketCanvasEnv(observation_mode=OBSERVATION_MODE_PIXELS)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (DEFAULT_PIXEL_SIZE[1], DEFAULT_PIXEL_SIZE[0], 3)
        assert obs.dtype == np.uint8
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_accepts_prompt_text_option(self):
        env = MarketCanvasEnv()
        obs, info = env.reset(
            seed=42,
            options={
                "prompt_text": (
                    "Create a Summer Sale email banner with a headline, "
                    "a yellow CTA button, and good contrast"
                )
            },
        )
        assert info["prompt"].startswith("Create a Summer Sale")
        assert int(obs["prompt_id"]) == 0
        env.close()


class TestEnvStep:
    @staticmethod
    def _make_action(action_type: int = ACTION_ADD_TEXT, **overrides: int) -> dict[str, int]:
        action = {
            "action_type": action_type,
            "element_idx": 0,
            "x": 100,
            "y": 100,
            "width": 200,
            "height": 50,
            "color_idx": 0,
            "content_idx": 0,
        }
        action.update(overrides)
        return action

    def test_add_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        _, reward, terminated, truncated, info = env.step(self._make_action())
        assert info["element_count"] == 1
        assert info["action_result"]["success"] is True
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        env.close()

    def test_add_shape_auto_sets_contrasting_text_color(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        env.step(self._make_action(action_type=ACTION_ADD_SHAPE, color_idx=1))
        assert env._canvas.get_all_elements()[0].text_color == "#FFFFFF"
        env.close()

    def test_done_terminates(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        _, reward, terminated, truncated, _ = env.step(self._make_action(action_type=ACTION_DONE))
        assert terminated is True
        assert truncated is False
        assert reward != 0.0
        env.close()

    def test_truncation_at_max_steps(self):
        env = MarketCanvasEnv(max_steps=3)
        env.reset(seed=42)
        reward = 0.0
        truncated = False
        for _ in range(3):
            _, reward, _, truncated, _ = env.step(self._make_action())
        assert truncated is True
        assert reward != 0.0
        env.close()

    def test_move_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        env.step(self._make_action())
        _, _, _, _, info = env.step(self._make_action(action_type=ACTION_MOVE, element_idx=0, x=300, y=300))
        assert info["action_result"]["success"] is True
        env.close()

    def test_move_nonexistent_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(self._make_action(action_type=ACTION_MOVE, element_idx=5, x=300, y=300))
        assert info["action_result"]["success"] is False
        env.close()

    def test_recolor_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        env.step(self._make_action())
        _, _, _, _, info = env.step(self._make_action(action_type=ACTION_RECOLOR, element_idx=0, color_idx=2))
        assert info["action_result"]["success"] is True
        env.close()

    def test_remove_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        env.step(self._make_action())
        assert env._canvas.element_count == 1
        _, _, _, _, info = env.step(self._make_action(action_type=ACTION_REMOVE, element_idx=0))
        assert info["action_result"]["success"] is True
        assert env._canvas.element_count == 0
        env.close()

    def test_random_actions_dont_crash(self):
        env = MarketCanvasEnv(max_steps=20)
        env.reset(seed=42)
        for _ in range(20):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        env.close()

    def test_observation_always_in_space(self):
        env = MarketCanvasEnv(max_steps=10)
        env.reset(seed=42)
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        env.close()

    def test_reward_in_range(self):
        env = MarketCanvasEnv(max_steps=5)
        env.reset(seed=42)
        for _ in range(5):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert -1.0 <= reward <= 1.0
            if terminated or truncated:
                break
        env.close()


class TestEnvRender:
    def test_render_rgb_array(self):
        env = MarketCanvasEnv(render_mode="rgb_array")
        env.reset(seed=42)
        img = env.render()
        assert isinstance(img, np.ndarray)
        assert img.shape == (600, 800, 3)
        assert img.dtype == np.uint8
        env.close()

    def test_render_none_mode(self):
        env = MarketCanvasEnv(render_mode=None)
        env.reset(seed=42)
        assert env.render() is None
        env.close()

    def test_pixels_only_mode_still_keeps_native_render_resolution(self):
        env = MarketCanvasEnv(
            render_mode="rgb_array",
            observation_mode=OBSERVATION_MODE_PIXELS,
        )
        obs, _ = env.reset(seed=42)
        img = env.render()
        assert obs.shape == (DEFAULT_PIXEL_SIZE[1], DEFAULT_PIXEL_SIZE[0], 3)
        assert img.shape == (600, 800, 3)
        env.close()


class TestSemanticState:
    def test_get_semantic_state(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        state = env.get_semantic_state()
        assert "canvas" in state
        assert "elements" in state
        assert "target_prompt" in state
        assert "step_count" in state
        assert "max_steps" in state
        assert "spatial_relationships" in state
        env.close()

    def test_semantic_state_includes_spatial_relationships(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        env._canvas.add_element(ElementType.TEXT, x=50, y=50, width=100, height=40, content="A")
        env._canvas.add_element(ElementType.SHAPE, x=200, y=50, width=120, height=60, content="B")
        state = env.get_semantic_state()
        relationships = state["spatial_relationships"]
        assert len(relationships) == 1
        relationship = relationships[0]
        assert relationship["element_a"] == "element_0"
        assert relationship["element_b"] == "element_1"
        assert relationship["a_left_of_b"] is True
        assert relationship["overlaps"] is False
        env.close()


class TestCheckEnv:
    def test_check_env_passes(self):
        from gymnasium.utils.env_checker import check_env

        env = MarketCanvasEnv()
        check_env(env.unwrapped)
        env.close()


class TestWrappers:
    def test_dense_reward_wrapper(self):
        env = DenseRewardWrapper(MarketCanvasEnv(max_steps=10))
        env.reset(seed=42)
        action = env.action_space.sample()
        action["action_type"] = ACTION_ADD_TEXT
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)
        env.close()

    def test_pixel_observation_wrapper(self):
        env = PixelObservationWrapper(MarketCanvasEnv())
        obs, _ = env.reset(seed=42)
        assert "pixels" in obs
        assert obs["pixels"].shape == (DEFAULT_PIXEL_SIZE[1], DEFAULT_PIXEL_SIZE[0], 3)
        assert obs["pixels"].dtype == np.uint8
        assert env.observation_space.contains(obs)
        env.close()

    def test_pixel_observation_wrapper_pixels_only(self):
        env = PixelObservationWrapper(MarketCanvasEnv(), include_semantic=False)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (DEFAULT_PIXEL_SIZE[1], DEFAULT_PIXEL_SIZE[0], 3)
        assert obs.dtype == np.uint8
        assert env.observation_space.contains(obs)
        env.close()

    def test_pixel_observation_wrapper_is_deterministic_for_same_state(self):
        first_env = PixelObservationWrapper(MarketCanvasEnv())
        second_env = PixelObservationWrapper(MarketCanvasEnv())

        first_obs, _ = first_env.reset(seed=42)
        second_obs, _ = second_env.reset(seed=42)
        assert np.array_equal(first_obs["pixels"], second_obs["pixels"])

        action = {
            "action_type": ACTION_ADD_TEXT,
            "element_idx": 0,
            "x": 100,
            "y": 100,
            "width": 200,
            "height": 50,
            "color_idx": 0,
            "content_idx": 0,
        }
        first_obs, _, _, _, _ = first_env.step(action)
        second_obs, _, _, _, _ = second_env.step(action)
        assert np.array_equal(first_obs["pixels"], second_obs["pixels"])

        first_env.close()
        second_env.close()

    def test_pixel_wrapper_keeps_native_render_full_resolution(self):
        env = PixelObservationWrapper(MarketCanvasEnv(render_mode="rgb_array"))
        obs, _ = env.reset(seed=42)
        native = env.render()

        assert obs["pixels"].shape == (DEFAULT_PIXEL_SIZE[1], DEFAULT_PIXEL_SIZE[0], 3)
        assert isinstance(native, np.ndarray)
        assert native.shape == (600, 800, 3)
        assert native.dtype == np.uint8
        env.close()

    def test_flat_action_wrapper(self):
        env = FlatActionWrapper(MarketCanvasEnv())
        assert isinstance(env.action_space, gymnasium.spaces.MultiDiscrete)
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        env.close()

    def test_flat_action_wrapper_composes_with_pixel_wrapper(self):
        env = FlatActionWrapper(PixelObservationWrapper(MarketCanvasEnv()))
        assert isinstance(env.action_space, gymnasium.spaces.MultiDiscrete)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        env.close()
