"""Tests for the reward engine."""

import numpy as np
import pytest

from engine.canvas import Canvas
from engine.types import CanvasConfig, ElementType
from rewards.accessibility import AccessibilityChecker, contrast_ratio, relative_luminance
from rewards.aesthetics import AestheticsScorer
from rewards.calculator import RewardCalculator
from rewards.constraints import ConstraintChecker, _color_distance
from rewards.prompts import ConstraintType, PromptBank, PromptConstraint, TargetPrompt


class TestAccessibility:
    def test_relative_luminance_black(self):
        assert relative_luminance("#000000") == pytest.approx(0.0)

    def test_relative_luminance_white(self):
        assert relative_luminance("#FFFFFF") == pytest.approx(1.0)

    def test_contrast_ratio_black_white(self):
        assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0, abs=0.1)

    def test_contrast_ratio_same_color(self):
        assert contrast_ratio("#FF0000", "#FF0000") == pytest.approx(1.0)

    def test_contrast_ratio_is_symmetric(self):
        assert contrast_ratio("#FF0000", "#0000FF") == pytest.approx(
            contrast_ratio("#0000FF", "#FF0000")
        )

    def test_score_no_text_elements(self):
        canvas = Canvas()
        canvas.add_element(ElementType.IMAGE, x=0, y=0, width=100, height=100)
        assert AccessibilityChecker().score(canvas) == 1.0

    def test_score_empty_canvas(self):
        assert AccessibilityChecker().score(Canvas()) == 1.0

    def test_score_black_text_on_white_bg(self):
        canvas = Canvas(CanvasConfig(background_color="#FFFFFF"))
        canvas.add_element(
            ElementType.TEXT,
            x=10,
            y=10,
            width=100,
            height=30,
            text_color="#000000",
            content="Hello",
        )
        assert AccessibilityChecker().score(canvas) == 1.0

    def test_score_white_text_on_white_bg_fails(self):
        canvas = Canvas(CanvasConfig(background_color="#FFFFFF"))
        canvas.add_element(
            ElementType.TEXT,
            x=10,
            y=10,
            width=100,
            height=30,
            text_color="#FFFFFF",
            content="Invisible",
        )
        assert AccessibilityChecker().score(canvas) == 0.0

    def test_score_shape_button_contrast(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE,
            x=10,
            y=10,
            width=200,
            height=60,
            color="#000080",
            text_color="#FFFFFF",
            content="Buy Now",
        )
        assert AccessibilityChecker().score(canvas) == 1.0

    def test_score_shape_bad_contrast(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE,
            x=10,
            y=10,
            width=200,
            height=60,
            color="#FFD700",
            text_color="#FFFF00",
            content="Buy Now",
        )
        assert AccessibilityChecker().score(canvas) == 0.0

    def test_text_on_shape_background(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE,
            x=0,
            y=0,
            width=800,
            height=600,
            color="#000000",
        )
        canvas.add_element(
            ElementType.TEXT,
            x=100,
            y=100,
            width=200,
            height=50,
            text_color="#FFFFFF",
            content="White on Black",
        )
        assert AccessibilityChecker().score(canvas) == 1.0

    def test_large_text_lower_threshold(self):
        canvas = Canvas(CanvasConfig(background_color="#FFFFFF"))
        canvas.add_element(
            ElementType.TEXT,
            x=10,
            y=10,
            width=200,
            height=50,
            text_color="#808080",
            content="Large",
            font_size=18,
        )
        canvas.add_element(
            ElementType.TEXT,
            x=10,
            y=80,
            width=200,
            height=50,
            text_color="#808080",
            content="Small",
            font_size=14,
        )
        assert AccessibilityChecker().score(canvas) == pytest.approx(0.5)


class TestAesthetics:
    def test_empty_canvas(self):
        assert AestheticsScorer().score(Canvas()) == 0.0

    def test_no_overlap(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=100)
        canvas.add_element(ElementType.SHAPE, x=200, y=200, width=100, height=100)
        assert AestheticsScorer()._overlap_score(canvas) == 1.0

    def test_full_overlap(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=100)
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=100)
        assert AestheticsScorer()._overlap_score(canvas) < 1.0

    def test_alignment_perfect_center_x(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=300, y=0, width=200, height=50)
        canvas.add_element(ElementType.SHAPE, x=300, y=100, width=200, height=50)
        canvas.add_element(ElementType.SHAPE, x=300, y=200, width=200, height=50)
        assert AestheticsScorer()._alignment_score(canvas) == 1.0

    def test_margins_respected(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=30, y=30, width=100, height=50)
        assert AestheticsScorer()._margin_score(canvas) == 1.0

    def test_margins_violated(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=5, y=5, width=100, height=50)
        assert AestheticsScorer()._margin_score(canvas) == 0.0

    def test_even_spacing(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=100, y=75, width=100, height=50)
        canvas.add_element(ElementType.SHAPE, x=100, y=275, width=100, height=50)
        canvas.add_element(ElementType.SHAPE, x=100, y=475, width=100, height=50)
        assert AestheticsScorer()._spacing_score(canvas) == 1.0

    def test_single_element(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=100, y=100, width=200, height=100)
        score = AestheticsScorer().score(canvas)
        assert 0.0 <= score <= 1.0


class TestConstraints:
    def test_empty_constraints(self):
        assert ConstraintChecker().score(Canvas(), ()) == 1.0

    def test_has_element_by_type(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=100, height=50, content="Hello")
        constraint = PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "TEXT"})
        assert ConstraintChecker().score(canvas, (constraint,)) == 1.0

    def test_has_element_missing(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=50)
        constraint = PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "TEXT"})
        assert ConstraintChecker().score(canvas, (constraint,)) == 0.0

    def test_has_element_with_keyword(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.TEXT,
            x=0,
            y=0,
            width=100,
            height=50,
            content="Summer Sale",
        )
        constraint = PromptConstraint(
            ConstraintType.HAS_ELEMENT, {"type": "TEXT", "keywords": ["sale"]}
        )
        assert ConstraintChecker().score(canvas, (constraint,)) == 1.0

    def test_has_element_keyword_no_match(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.TEXT,
            x=0,
            y=0,
            width=100,
            height=50,
            content="Hello World",
        )
        constraint = PromptConstraint(
            ConstraintType.HAS_ELEMENT, {"type": "TEXT", "keywords": ["sale"]}
        )
        assert ConstraintChecker().score(canvas, (constraint,)) == 0.0

    def test_element_color_match(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE,
            x=0,
            y=0,
            width=100,
            height=50,
            color="#FFD700",
            content="Shop Now",
        )
        constraint = PromptConstraint(
            ConstraintType.ELEMENT_COLOR,
            {
                "type": "SHAPE",
                "keywords": ["shop"],
                "target_color": "#FFD700",
                "tolerance": 80,
            },
        )
        assert ConstraintChecker().score(canvas, (constraint,)) == 1.0

    def test_element_color_within_tolerance(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE,
            x=0,
            y=0,
            width=100,
            height=50,
            color="#FFCC00",
            content="Shop Now",
        )
        constraint = PromptConstraint(
            ConstraintType.ELEMENT_COLOR,
            {
                "type": "SHAPE",
                "keywords": ["shop"],
                "target_color": "#FFD700",
                "tolerance": 80,
            },
        )
        assert ConstraintChecker().score(canvas, (constraint,)) == 1.0

    def test_min_elements(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=100, height=50)
        canvas.add_element(ElementType.SHAPE, x=0, y=100, width=100, height=50)
        constraint = PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 2})
        assert ConstraintChecker().score(canvas, (constraint,)) == 1.0

    def test_min_elements_not_met(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=100, height=50)
        constraint = PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 3})
        assert ConstraintChecker().score(canvas, (constraint,)) == 0.0

    def test_partial_constraint_satisfaction(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.TEXT,
            x=0,
            y=0,
            width=100,
            height=50,
            content="Summer Sale",
        )
        constraints = (
            PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "TEXT", "keywords": ["sale"]}),
            PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "SHAPE", "keywords": ["buy"]}),
        )
        assert ConstraintChecker().score(canvas, constraints) == pytest.approx(0.5)

    def test_color_distance(self):
        assert _color_distance("#000000", "#000000") == 0.0
        assert _color_distance("#FFFFFF", "#000000") == pytest.approx(441.67, abs=1.0)


class TestRewardCalculator:
    @staticmethod
    def _make_prompt() -> TargetPrompt:
        return TargetPrompt(
            text="Test prompt",
            constraints=(PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 1}),),
        )

    def test_empty_canvas_negative_reward(self):
        reward, breakdown = RewardCalculator().calculate(Canvas(), self._make_prompt(), 0, 50)
        assert reward < 0.0
        assert -1.0 <= reward <= 1.0
        assert breakdown["constraint"] == 0.0

    def test_reward_clamped(self):
        reward, _ = RewardCalculator().calculate(Canvas(), self._make_prompt(), 50, 50)
        assert -1.0 <= reward <= 1.0

    def test_breakdown_has_all_keys(self):
        _, breakdown = RewardCalculator().calculate(Canvas(), self._make_prompt(), 10, 50)
        assert set(breakdown) == {
            "constraint",
            "aesthetics",
            "accessibility",
            "coverage",
            "efficiency",
        }

    def test_all_sub_scores_in_range(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=100, y=100, width=200, height=50, content="Sale")
        _, breakdown = RewardCalculator().calculate(canvas, self._make_prompt(), 10, 50)
        for value in breakdown.values():
            assert 0.0 <= value <= 1.0

    def test_coverage_peak(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=100, y=50, width=438, height=438)
        _, breakdown = RewardCalculator().calculate(canvas, self._make_prompt(), 1, 50)
        assert breakdown["coverage"] > 0.9

    def test_coverage_empty(self):
        _, breakdown = RewardCalculator().calculate(Canvas(), self._make_prompt(), 0, 50)
        assert breakdown["coverage"] == 0.0

    def test_efficiency_decreases_with_steps(self):
        canvas = Canvas()
        calc = RewardCalculator()
        _, early = calc.calculate(canvas, self._make_prompt(), 10, 50)
        _, late = calc.calculate(canvas, self._make_prompt(), 40, 50)
        assert early["efficiency"] > late["efficiency"]

    def test_custom_weights(self):
        weights = {
            "constraint": 1.0,
            "aesthetics": 0.0,
            "accessibility": 0.0,
            "coverage": 0.0,
            "efficiency": 0.0,
        }
        reward, _ = RewardCalculator(weights=weights).calculate(Canvas(), self._make_prompt(), 0, 50)
        assert reward == pytest.approx(-1.0)


class TestPromptBank:
    def test_sample_returns_target_prompt(self):
        prompt = PromptBank().sample(np.random.default_rng(42))
        assert isinstance(prompt, TargetPrompt)
        assert prompt.text
        assert prompt.constraints

    def test_sample_is_deterministic_with_seed(self):
        bank = PromptBank()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        assert bank.sample(rng1).text == bank.sample(rng2).text

    def test_all_prompts_have_constraints(self):
        for prompt in PromptBank().PROMPTS:
            assert prompt.text
            assert prompt.constraints
