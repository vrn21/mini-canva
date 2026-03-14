"""Prompt bank and constraint data types for design tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConstraintType(str, Enum):
    """Machine-checkable constraint types."""

    HAS_ELEMENT = "has_element"
    ELEMENT_COLOR = "element_color"
    MIN_ELEMENTS = "min_elements"


@dataclass(frozen=True)
class PromptConstraint:
    """A single machine-checkable constraint extracted from a prompt."""

    type: ConstraintType
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetPrompt:
    """A design task: natural language description + machine-checkable constraints."""

    text: str
    constraints: tuple[PromptConstraint, ...] = ()
    difficulty: str = "easy"


class PromptBank:
    """Repository of target prompts for episode initialization."""

    PROMPTS: tuple[TargetPrompt, ...] = (
        TargetPrompt(
            text=(
                "Create a Summer Sale email banner with a headline, "
                "a yellow CTA button, and good contrast"
            ),
            constraints=(
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "TEXT",
                        "keywords": ["sale", "summer"],
                    },
                ),
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "SHAPE",
                        "keywords": ["shop", "buy", "cta", "button", "order"],
                    },
                ),
                PromptConstraint(
                    ConstraintType.ELEMENT_COLOR,
                    {
                        "type": "SHAPE",
                        "keywords": ["shop", "buy", "cta", "button", "order"],
                        "target_color": "#FFD700",
                        "tolerance": 80,
                    },
                ),
                PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 2}),
            ),
            difficulty="easy",
        ),
        TargetPrompt(
            text=(
                "Design a product launch announcement with a hero image, "
                "product name, and launch date"
            ),
            constraints=(
                PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "IMAGE"}),
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "TEXT",
                        "keywords": ["product", "launch", "new", "introducing"],
                    },
                ),
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "TEXT",
                        "keywords": ["date", "coming", "available", "now"],
                    },
                ),
                PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 3}),
            ),
            difficulty="easy",
        ),
        TargetPrompt(
            text=(
                "Create a newsletter signup banner with a heading, "
                "an email input placeholder, and a subscribe button"
            ),
            constraints=(
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "TEXT",
                        "keywords": ["newsletter", "subscribe", "signup", "join"],
                    },
                ),
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "SHAPE",
                        "keywords": ["email", "input", "enter"],
                    },
                ),
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "SHAPE",
                        "keywords": ["subscribe", "submit", "join", "sign"],
                    },
                ),
                PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 3}),
            ),
            difficulty="medium",
        ),
        TargetPrompt(
            text=(
                "Design a holiday greeting card with a festive image, "
                "a greeting message, and a decorative border"
            ),
            constraints=(
                PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "IMAGE"}),
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "TEXT",
                        "keywords": ["happy", "merry", "holiday", "season", "wish"],
                    },
                ),
                PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 3}),
            ),
            difficulty="easy",
        ),
        TargetPrompt(
            text=(
                "Create a flash sale countdown banner with a bold headline, "
                "discount percentage, time remaining, and a red shop now button"
            ),
            constraints=(
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "TEXT",
                        "keywords": ["flash", "sale", "hurry", "limited"],
                    },
                ),
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "TEXT",
                        "keywords": ["%", "off", "discount", "save"],
                    },
                ),
                PromptConstraint(
                    ConstraintType.HAS_ELEMENT,
                    {
                        "type": "SHAPE",
                        "keywords": ["shop", "buy", "order"],
                    },
                ),
                PromptConstraint(
                    ConstraintType.ELEMENT_COLOR,
                    {
                        "type": "SHAPE",
                        "keywords": ["shop", "buy", "order"],
                        "target_color": "#FF0000",
                        "tolerance": 80,
                    },
                ),
                PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 4}),
            ),
            difficulty="hard",
        ),
    )

    def sample(self, rng: Any) -> TargetPrompt:
        """Sample a random prompt using the given numpy RNG."""

        idx = int(rng.integers(0, len(self.PROMPTS)))
        return self.PROMPTS[idx]
