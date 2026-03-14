"""RewardCalculator — combines sub-reward components into a final scalar."""

from __future__ import annotations

from typing import Any

from engine.canvas import Canvas

from rewards.accessibility import AccessibilityChecker
from rewards.aesthetics import AestheticsScorer
from rewards.constraints import ConstraintChecker
from rewards.prompts import TargetPrompt

_DEFAULT_WEIGHTS: dict[str, float] = {
    "constraint": 0.35,
    "aesthetics": 0.25,
    "accessibility": 0.20,
    "coverage": 0.10,
    "efficiency": 0.10,
}


class RewardCalculator:
    """Combines sub-reward components into a scalar reward in [-1.0, 1.0]."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = self._validate_weights(weights or dict(_DEFAULT_WEIGHTS))
        self._constraint_checker = ConstraintChecker()
        self._aesthetics_scorer = AestheticsScorer()
        self._accessibility_checker = AccessibilityChecker()

    def calculate(
        self,
        canvas: Canvas,
        prompt: TargetPrompt,
        steps_taken: int,
        max_steps: int,
    ) -> tuple[float, dict[str, Any]]:
        """Compute the total reward and per-component breakdown."""

        breakdown = {
            "constraint": self._constraint_checker.score(canvas, prompt.constraints),
            "aesthetics": self._aesthetics_scorer.score(canvas),
            "accessibility": self._accessibility_checker.score(canvas),
            "coverage": self._coverage_score(canvas),
            "efficiency": max(0.0, 1.0 - (steps_taken / max_steps)) if max_steps > 0 else 1.0,
        }
        raw_reward = sum(self.weights[key] * breakdown[key] for key in breakdown)
        reward = max(-1.0, min(1.0, 2.0 * raw_reward - 1.0))
        return reward, breakdown

    def _coverage_score(self, canvas: Canvas) -> float:
        """Score based on total element bounding-box area vs canvas area."""

        canvas_area = canvas.config.width * canvas.config.height
        element_area = sum(element.area for element in canvas.get_all_elements())
        ratio = min(element_area / canvas_area, 1.0) if canvas_area > 0 else 0.0

        if ratio > 0.8:
            return 0.0
        if ratio <= 0.4:
            return ratio / 0.4
        return (0.8 - ratio) / 0.4

    @staticmethod
    def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
        """Validate reward weights are complete and normalized."""

        expected_keys = set(_DEFAULT_WEIGHTS)
        actual_keys = set(weights)
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            raise ValueError(
                f"Reward weights must match {sorted(expected_keys)}. "
                f"Missing={sorted(missing)}, extra={sorted(extra)}"
            )

        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Reward weights must sum to 1.0, got {total}")

        return dict(weights)
