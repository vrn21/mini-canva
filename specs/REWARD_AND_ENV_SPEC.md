# Phase 2 — Reward Engine & RL Environment Implementation Specification

> **Audience**: Implementing agent. This document contains the complete specification for building the Reward Engine and RL Environment layers of MarketCanvas-Env. Every module, class, method, and edge case is defined here. Implement these files exactly as specified.
>
> **Depends on**: `engine/` package (Canvas Engine — implemented + QA'd, 59/59 tests pass).
>
> **North star**: `specs/REQS.md` — all design decisions trace back to assignment requirements.

---

## 0. Requirements Traceability

Every piece of this spec maps to a REQS.md requirement:

| REQS.md Requirement | Where It's Implemented |
|---|---|
| "heuristic-based reward function that calculates a scalar reward (-1.0 to 1.0)" | `rewards/calculator.py` — `RewardCalculator.calculate()` |
| "Constraint Satisfaction: Are the required elements present?" | `rewards/constraints.py` — `ConstraintChecker` |
| "Aesthetics/Design Rules: Are elements overlapping illegibly? Are they aligned?" | `rewards/aesthetics.py` — `AestheticsScorer` |
| "Accessibility: Does the text contrast pass basic WCAG ratios?" | `rewards/accessibility.py` — `AccessibilityChecker` |
| "expose a standard RL interface (State, Action, Reward)" | `env/market_canvas_env.py` — `MarketCanvasEnv(gymnasium.Env)` |
| "Semantic State: A JSON representation" | `env/market_canvas_env.py` — `get_semantic_state()` |
| "Visual State: A rendered pixel array (RGB)" | `env/wrappers.py` — `PixelObservationWrapper` |
| "High-Level (Semantic UI): add_element, move_element, change_element_color" | `env/spaces.py` — `build_action_space()` |
| "target prompt (e.g., 'Create a Summer Sale email banner...')" | `rewards/prompts.py` — `PromptBank` |
| "demo.py" | `demo.py` (Phase 3, not this spec) |
| "MCP Server" | `mcp/server.py` (Phase 3, not this spec) |

---

## 1. Scope & Constraints

**What this spec covers**:
- `rewards/` package — 5 files: `__init__.py`, `accessibility.py`, `aesthetics.py`, `constraints.py`, `prompts.py`, `calculator.py`
- `env/` package — 4 files: `__init__.py`, `market_canvas_env.py`, `spaces.py`, `wrappers.py`
- `tests/test_rewards.py` — unit tests for all reward components
- `tests/test_env.py` — Gymnasium compliance + integration tests

**What this spec does NOT cover**: MCP server, demo script, WRITEUP.md. Those are Phase 3.

**Hard constraints from REQS.md**:
- Reward is a scalar in [-1.0, 1.0]
- Reward computed at episode end (terminal reward)
- Must support `gymnasium.Env` API: `reset()`, `step()`, `render()`
- High-level semantic action space (add_element, move_element, change_color)
- Observation includes semantic JSON state
- Visual state (RGB pixel array) is optional/bonus

**Design principles**:
1. **Each layer depends only on the layer below** — rewards/ imports from engine/, env/ imports from both
2. **No over-engineering** — implement what REQS.md asks for, nothing more
3. **Deterministic** — same state + action → same next state. Randomness only in `reset()` (prompt sampling)
4. **Testable in isolation** — each reward sub-scorer can be tested with a manually-built Canvas, no env needed
5. **`check_env` compliant** — must pass `gymnasium.utils.env_checker.check_env()`

---

## 2. File Structure

```
mini-canva/
├── engine/                    # ✅ Already implemented (Layer 1)
│   ├── __init__.py
│   ├── types.py
│   ├── canvas.py
│   └── renderer.py
│
├── rewards/                   # 🆕 This spec (Layer 2)
│   ├── __init__.py
│   ├── accessibility.py       # WCAG contrast checker
│   ├── aesthetics.py          # Layout quality scorer
│   ├── constraints.py         # Prompt constraint checker
│   ├── prompts.py             # PromptBank + data types
│   └── calculator.py          # RewardCalculator orchestrator
│
├── env/                       # 🆕 This spec (Layer 3)
│   ├── __init__.py
│   ├── market_canvas_env.py   # MarketCanvasEnv(gymnasium.Env)
│   ├── spaces.py              # Observation/action space builders
│   └── wrappers.py            # DenseRewardWrapper, PixelObservationWrapper
│
├── tests/
│   ├── test_engine.py         # ✅ Already implemented (59 tests)
│   ├── test_rewards.py        # 🆕 This spec
│   └── test_env.py            # 🆕 This spec
│
└── pyproject.toml             # 🔄 Updated (add gymnasium dependency)
```

---

## 3. Dependency Changes

`pyproject.toml` needs two additions:

```toml
[project]
dependencies = [
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "gymnasium>=1.0.0",       # NEW — RL environment framework
]

[tool.hatch.build.targets.wheel]
packages = ["engine", "rewards", "env"]   # NEW — add rewards and env packages
```

No other dependencies. No ML frameworks, no browser, no GPU libraries.

---

## 4. Layer 2: Reward Engine

### 4.1 Architecture Overview

```
                    ┌──────────────────┐
                    │ RewardCalculator │
                    │  (orchestrator)  │
                    └────────┬─────────┘
                             │ calls each, weighted sum
          ┌──────────────────┼──────────────────┬─────────────────┐
          │                  │                  │                 │
  ┌───────▼───────┐  ┌──────▼──────┐  ┌────────▼────────┐  (coverage +
  │ Constraint    │  │ Aesthetics  │  │ Accessibility   │   efficiency
  │ Checker       │  │ Scorer      │  │ Checker (WCAG)  │   computed
  └───────────────┘  └─────────────┘  └─────────────────┘   inline)
          │                  │                  │
          ▼                  ▼                  ▼
    PromptBank          Canvas API          Canvas API
    (prompts.py)     (get_overlapping_     (get_element_behind,
                      pairs, etc.)         config.background_color)
```

**Key design decision**: Each sub-scorer is a standalone class with a single `score(canvas, ...) → float` method returning a value in [0.0, 1.0]. The `RewardCalculator` combines them with weights and maps to [-1.0, 1.0]. This makes each component independently testable and the weights tunable.

---

### 4.2 `rewards/__init__.py`

```python
"""Reward Engine — heuristic reward calculator for design quality."""

from rewards.accessibility import AccessibilityChecker
from rewards.aesthetics import AestheticsScorer
from rewards.calculator import RewardCalculator
from rewards.constraints import ConstraintChecker
from rewards.prompts import ConstraintType, PromptBank, PromptConstraint, TargetPrompt

__all__ = [
    "AccessibilityChecker",
    "AestheticsScorer",
    "ConstraintChecker",
    "ConstraintType",
    "PromptBank",
    "PromptConstraint",
    "RewardCalculator",
    "TargetPrompt",
]
```

---

### 4.3 `rewards/prompts.py` — Prompt Bank & Data Types

This file defines the prompt data structures and the bank of target prompts. The prompt bank is what `env.reset()` samples from to give the agent a design task.

```python
"""Prompt bank and constraint data types for design tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConstraintType(str, Enum):
    """Machine-checkable constraint types.

    Inherits from str so they remain JSON-serializable
    and directly comparable to string values.
    """

    HAS_ELEMENT = "has_element"
    ELEMENT_COLOR = "element_color"
    MIN_ELEMENTS = "min_elements"


@dataclass(frozen=True)
class PromptConstraint:
    """A single machine-checkable constraint extracted from a prompt.

    Attributes:
        type: The constraint type enum.
        params: Type-specific parameters (see ConstraintChecker for schema).
    """

    type: ConstraintType
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetPrompt:
    """A design task: natural language description + machine-checkable constraints.

    Attributes:
        text: Human-readable prompt (shown in info dict, used by LLM agents).
        constraints: Machine-checkable constraints for the ConstraintChecker.
        difficulty: Informational label. Not used by reward logic.
    """

    text: str
    constraints: tuple[PromptConstraint, ...] = ()
    difficulty: str = "easy"


class PromptBank:
    """Repository of target prompts for episode initialization.

    Prompts are hardcoded. No NLP, no parsing. Each prompt has a human-readable
    text and a list of machine-checkable constraints that the ConstraintChecker
    can evaluate against a Canvas.
    """

    PROMPTS: tuple[TargetPrompt, ...] = (
        TargetPrompt(
            text=(
                "Create a Summer Sale email banner with a headline, "
                "a yellow CTA button, and good contrast"
            ),
            constraints=(
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "TEXT",
                    "keywords": ["sale", "summer"],
                }),
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "SHAPE",
                    "keywords": ["shop", "buy", "cta", "button", "order"],
                }),
                PromptConstraint(ConstraintType.ELEMENT_COLOR, {
                    "type": "SHAPE",
                    "keywords": ["shop", "buy", "cta", "button", "order"],
                    "target_color": "#FFD700",
                    "tolerance": 80,
                }),
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
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "TEXT",
                    "keywords": ["product", "launch", "new", "introducing"],
                }),
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "TEXT",
                    "keywords": ["date", "coming", "available", "now"],
                }),
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
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "TEXT",
                    "keywords": ["newsletter", "subscribe", "signup", "join"],
                }),
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "SHAPE",
                    "keywords": ["email", "input", "enter"],
                }),
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "SHAPE",
                    "keywords": ["subscribe", "submit", "join", "sign"],
                }),
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
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "TEXT",
                    "keywords": ["happy", "merry", "holiday", "season", "wish"],
                }),
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
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "TEXT",
                    "keywords": ["flash", "sale", "hurry", "limited"],
                }),
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "TEXT",
                    "keywords": ["%", "off", "discount", "save"],
                }),
                PromptConstraint(ConstraintType.HAS_ELEMENT, {
                    "type": "SHAPE",
                    "keywords": ["shop", "buy", "order"],
                }),
                PromptConstraint(ConstraintType.ELEMENT_COLOR, {
                    "type": "SHAPE",
                    "keywords": ["shop", "buy", "order"],
                    "target_color": "#FF0000",
                    "tolerance": 80,
                }),
                PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 4}),
            ),
            difficulty="hard",
        ),
    )

    def sample(self, rng) -> TargetPrompt:
        """Sample a random prompt using the given numpy RNG.

        Args:
            rng: A numpy random Generator (e.g., env.np_random).

        Returns:
            A randomly selected TargetPrompt.
        """
        idx = int(rng.integers(0, len(self.PROMPTS)))
        return self.PROMPTS[idx]
```

#### 4.3.1 Design Notes

- **`frozen=True` on both dataclasses**: Prompts and constraints are immutable. No accidental mutation during episodes.
- **`tuple` not `list`** for `PROMPTS` and `constraints`: Immutable collections for a frozen dataclass.
- **No NLP**: Constraint checking is keyword-based. The prompt text is for the LLM agent's consumption. The machine-checkable constraints are separate.
- **`sample(rng)`**: Takes an external RNG (Gymnasium's `self.np_random`) to ensure seeded reproducibility.
- **Expandable**: Adding a new prompt is just adding a `TargetPrompt` to the tuple. No code changes needed.

---

### 4.4 `rewards/accessibility.py` — WCAG Contrast Checker

This is the most algorithmically precise component. WCAG 2.1 AA contrast calculation is a well-defined standard.

```python
"""WCAG 2.1 AA contrast ratio compliance checker."""

from __future__ import annotations

from engine.canvas import Canvas
from engine.types import Element, ElementType


class AccessibilityChecker:
    """Checks WCAG 2.1 AA contrast compliance for text elements.

    WCAG AA requires:
    - Normal text (< 18px): contrast ratio ≥ 4.5:1
    - Large text (≥ 18px): contrast ratio ≥ 3.0:1

    The checker scores the fraction of text-bearing elements that pass.
    """

    def score(self, canvas: Canvas) -> float:
        """Fraction of text-bearing elements meeting WCAG AA contrast.

        Text-bearing elements: TEXT elements with content, or SHAPE elements
        with content (button labels).

        Returns:
            1.0 if all text passes (or no text exists). 0.0 if none pass.
            Range: [0.0, 1.0].
        """
        text_elements = [
            e for e in canvas.get_all_elements()
            if e.content and e.type in (ElementType.TEXT, ElementType.SHAPE)
        ]

        if not text_elements:
            return 1.0  # No text = no violations

        passing = sum(
            1 for e in text_elements
            if self._check_contrast(e, canvas)
        )
        return passing / len(text_elements)

    def _check_contrast(self, element: Element, canvas: Canvas) -> bool:
        """Check if an element's text meets WCAG AA contrast."""
        bg_color = self._get_effective_background(element, canvas)
        ratio = contrast_ratio(element.text_color, bg_color)
        threshold = 3.0 if element.font_size >= 18 else 4.5
        return ratio >= threshold

    def _get_effective_background(self, element: Element, canvas: Canvas) -> str:
        """Determine the effective background color behind an element's text.

        For SHAPE elements: the background IS the shape's own color
        (text is rendered on top of the shape fill).

        For TEXT elements: find the nearest overlapping element behind,
        or fall back to canvas background color.
        """
        if element.type == ElementType.SHAPE:
            return element.color

        # TEXT element — look behind
        behind = canvas.get_element_behind(element.id)
        if behind is not None:
            return behind.color
        return canvas.config.background_color


def contrast_ratio(color1: str, color2: str) -> float:
    """WCAG 2.1 contrast ratio between two hex colors.

    Args:
        color1: 7-char hex string (e.g., "#FF0000").
        color2: 7-char hex string.

    Returns:
        Contrast ratio ≥ 1.0. Higher = more contrast.
    """
    l1 = relative_luminance(color1)
    l2 = relative_luminance(color2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def relative_luminance(hex_color: str) -> float:
    """Relative luminance of a color per WCAG 2.1.

    Args:
        hex_color: 7-char hex string (e.g., "#FF0000").

    Returns:
        Luminance in [0.0, 1.0]. 0.0 = black, 1.0 = white.
    """
    r_srgb = int(hex_color[1:3], 16) / 255.0
    g_srgb = int(hex_color[3:5], 16) / 255.0
    b_srgb = int(hex_color[5:7], 16) / 255.0

    r = r_srgb / 12.92 if r_srgb <= 0.04045 else ((r_srgb + 0.055) / 1.055) ** 2.4
    g = g_srgb / 12.92 if g_srgb <= 0.04045 else ((g_srgb + 0.055) / 1.055) ** 2.4
    b = b_srgb / 12.92 if b_srgb <= 0.04045 else ((b_srgb + 0.055) / 1.055) ** 2.4

    return 0.2126 * r + 0.7152 * g + 0.0722 * b
```

#### 4.4.1 Design Notes

- **`contrast_ratio` and `relative_luminance` are module-level functions**: Not static methods. They have utility outside the class (e.g., tests, MCP server's reward peek).
- **sRGB linearization threshold is 0.04045**: This is the correct WCAG 2.1 value. Some older references use 0.03928 — that's a rounding from the IEC standard. 0.04045 is what the W3C spec says.
- **SHAPE background rule**: A SHAPE element's text (e.g., button label "Shop Now") is rendered ON TOP of the shape fill. So the bg for contrast is `element.color`, not what's behind the shape. This is the real-world behavior.
- **TEXT background rule**: A TEXT element is transparent — whatever is behind it (another element or the canvas) is the bg. `canvas.get_element_behind()` already does the right thing (finds nearest overlapping element with lower z-order).

---

### 4.5 `rewards/aesthetics.py` — Layout Quality Scorer

Four sub-components, equally weighted. Each returns [0.0, 1.0].

```python
"""Layout aesthetics scorer — overlap, alignment, margin, spacing."""

from __future__ import annotations

from engine.canvas import Canvas

# Minimum margin from canvas edges (pixels)
_MIN_MARGIN = 20

# Alignment tolerance (pixels). Elements within this distance
# of a shared axis are considered aligned.
_ALIGN_TOLERANCE = 10


class AestheticsScorer:
    """Scores visual layout quality of elements on a canvas.

    Four equally-weighted sub-scores:
    1. Overlap: penalizes overlapping elements
    2. Alignment: rewards elements sharing alignment axes
    3. Margin: rewards elements respecting edge margins
    4. Spacing: rewards even vertical spacing between elements
    """

    def score(self, canvas: Canvas) -> float:
        """Overall aesthetics score.

        Returns:
            0.0 for empty canvas or terrible layout, up to 1.0 for perfect.
            Range: [0.0, 1.0].
        """
        if canvas.element_count == 0:
            return 0.0

        if canvas.element_count == 1:
            # Single element: only margin matters. Overlap/alignment/spacing
            # are undefined for one element.
            return self._margin_score(canvas)

        scores = [
            self._overlap_score(canvas),
            self._alignment_score(canvas),
            self._margin_score(canvas),
            self._spacing_score(canvas),
        ]
        return sum(scores) / len(scores)

    def _overlap_score(self, canvas: Canvas) -> float:
        """1.0 = no overlaps, 0.0 = severe overlaps.

        Computed as: 1.0 - (total_overlap_area / total_element_area).
        Clamped to [0.0, 1.0].
        """
        elements = canvas.get_all_elements()
        total_element_area = sum(e.area for e in elements)
        if total_element_area == 0:
            return 1.0

        pairs = canvas.get_overlapping_pairs()
        total_overlap_area = sum(area for _, _, area in pairs)

        score = 1.0 - (total_overlap_area / total_element_area)
        return max(0.0, score)

    def _alignment_score(self, canvas: Canvas) -> float:
        """Score how many elements share common alignment axes.

        Checks three axes: center-X, center-Y, left-edge.
        Returns the best (maximum) alignment fraction across all axes.

        Example: 3 elements, 2 share center-X → score = 2/3 ≈ 0.67.
        """
        elements = canvas.get_all_elements()
        n = len(elements)
        if n < 2:
            return 1.0

        center_xs = [e.center[0] for e in elements]
        center_ys = [e.center[1] for e in elements]
        left_edges = [float(e.x) for e in elements]

        best = 0.0
        for values in (center_xs, center_ys, left_edges):
            best = max(best, _max_cluster_fraction(values, _ALIGN_TOLERANCE, n))

        return best

    def _margin_score(self, canvas: Canvas) -> float:
        """Fraction of elements with adequate margins from canvas edges.

        An element "respects margins" if all four edges are ≥ _MIN_MARGIN
        pixels from the corresponding canvas edge.
        """
        elements = canvas.get_all_elements()
        w, h = canvas.config.width, canvas.config.height

        respecting = 0
        for e in elements:
            left, top, right, bottom = e.bounds
            if (left >= _MIN_MARGIN and top >= _MIN_MARGIN
                    and right <= w - _MIN_MARGIN and bottom <= h - _MIN_MARGIN):
                respecting += 1

        return respecting / len(elements)

    def _spacing_score(self, canvas: Canvas) -> float:
        """Regularity of vertical gaps between elements.

        Sort elements by center-Y, compute gaps between consecutive centers.
        Score = 1.0 - normalized_stddev(gaps). Even spacing → high score.

        Returns 1.0 if only 1 or 2 elements (no meaningful gap variation).
        """
        elements = canvas.get_all_elements()
        if len(elements) < 3:
            return 1.0

        centers_y = sorted(e.center[1] for e in elements)
        gaps = [centers_y[i + 1] - centers_y[i] for i in range(len(centers_y) - 1)]

        mean_gap = sum(gaps) / len(gaps)
        if mean_gap == 0:
            return 1.0  # All elements stacked at same Y

        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        stddev = variance ** 0.5
        normalized_stddev = stddev / mean_gap

        return max(0.0, 1.0 - normalized_stddev)


def _max_cluster_fraction(values: list[float], tolerance: float, n: int) -> float:
    """Find the largest cluster of values within ±tolerance.

    Returns the fraction of values in the largest cluster.
    Simple O(n²) approach — fine for n ≤ 20.
    """
    if n == 0:
        return 0.0

    best_count = 1
    for i in range(n):
        count = sum(1 for j in range(n) if abs(values[i] - values[j]) <= tolerance)
        best_count = max(best_count, count)

    return best_count / n
```

#### 4.5.1 Design Notes

- **Empty canvas → 0.0**: An empty canvas has no aesthetic merit. This works with constraint scoring to ensure the agent doesn't game by submitting nothing.
- **Single element → margin only**: For 1 element, overlap/alignment/spacing are meaningless. Only margin applies.
- **O(n²) is fine**: Max elements in practice is ≤ 20. `get_overlapping_pairs()` is already O(n²).
- **`_max_cluster_fraction`**: A simple brute-force approach. For each value, count how many others are within tolerance. The largest cluster wins. No need for anything fancier.
- **Spacing uses center-Y**: We sort by vertical center, not top edge, because center-to-center distance is the perceptually meaningful metric.

---

### 4.6 `rewards/constraints.py` — Prompt Constraint Checker

```python
"""Constraint checker — evaluates if a canvas satisfies prompt constraints."""

from __future__ import annotations

from engine.canvas import Canvas
from engine.types import ElementType

from rewards.prompts import ConstraintType, PromptConstraint


class ConstraintChecker:
    """Checks if a canvas state satisfies a list of prompt constraints.

    Each constraint is checked independently. The score is the fraction
    of constraints satisfied.
    """

    def score(self, canvas: Canvas, constraints: tuple[PromptConstraint, ...]) -> float:
        """Fraction of satisfied constraints.

        Returns:
            1.0 if all constraints pass (or no constraints). 0.0 if none pass.
            Range: [0.0, 1.0].
        """
        if not constraints:
            return 1.0

        satisfied = sum(1 for c in constraints if self._check_one(canvas, c))
        return satisfied / len(constraints)

    def _check_one(self, canvas: Canvas, constraint: PromptConstraint) -> bool:
        """Dispatch a single constraint to the appropriate checker."""
        handlers = {
            ConstraintType.HAS_ELEMENT: self._check_has_element,
            ConstraintType.ELEMENT_COLOR: self._check_element_color,
            ConstraintType.MIN_ELEMENTS: self._check_min_elements,
        }
        handler = handlers.get(constraint.type)
        if handler is None:
            return False  # Unknown constraint type → not satisfied
        return handler(canvas, constraint.params)

    def _check_has_element(self, canvas: Canvas, params: dict) -> bool:
        """Check if an element of the given type exists, optionally matching keywords.

        params:
            type (str): ElementType value — "TEXT", "SHAPE", or "IMAGE".
            keywords (list[str], optional): at least one keyword must appear
                in the element's content (case-insensitive substring match).
        """
        target_type = ElementType(params["type"])
        elements = canvas.get_elements_by_type(target_type)

        if not elements:
            return False

        keywords = params.get("keywords")
        if not keywords:
            return True  # Type match is enough

        for e in elements:
            content_lower = e.content.lower()
            if any(kw.lower() in content_lower for kw in keywords):
                return True

        return False

    def _check_element_color(self, canvas: Canvas, params: dict) -> bool:
        """Check if an element matching criteria has a color close to the target.

        params:
            type (str): ElementType value.
            keywords (list[str], optional): content keywords.
            target_color (str): hex color to match (e.g., "#FFD700").
            tolerance (int): max Euclidean distance in RGB space (0-441).
        """
        target_type = ElementType(params["type"])
        elements = canvas.get_elements_by_type(target_type)
        keywords = params.get("keywords")
        target_color = params["target_color"]
        tolerance = params.get("tolerance", 80)

        for e in elements:
            # Check keyword match if required
            if keywords:
                content_lower = e.content.lower()
                if not any(kw.lower() in content_lower for kw in keywords):
                    continue

            if _color_distance(e.color, target_color) <= tolerance:
                return True

        return False

    def _check_min_elements(self, canvas: Canvas, params: dict) -> bool:
        """Check if the canvas has at least N elements.

        params:
            count (int): minimum number of elements.
        """
        return canvas.element_count >= params["count"]


def _color_distance(hex1: str, hex2: str) -> float:
    """Euclidean distance between two hex colors in RGB space.

    Returns:
        Distance in [0, ~441.67]. 0 = identical, 441 = black vs white.
    """
    r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
    r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
```

#### 4.6.1 Design Notes

- **Keyword matching is case-insensitive substring**: `"summer sale".lower()` contains `"sale"`. Simple and effective.
- **Color distance**: Euclidean RGB distance. Not perceptually uniform, but good enough for "is this yellow-ish?" checks. Tolerance of 80 means roughly ±30 per channel.
- **Unknown constraint types → False**: Fail-safe. If a prompt has a constraint type the checker doesn't handle, it counts as unsatisfied.

---

### 4.7 `rewards/calculator.py` — Reward Orchestrator

```python
"""RewardCalculator — combines sub-reward components into a final scalar."""

from __future__ import annotations

from typing import Any

from engine.canvas import Canvas

from rewards.accessibility import AccessibilityChecker
from rewards.aesthetics import AestheticsScorer
from rewards.constraints import ConstraintChecker
from rewards.prompts import TargetPrompt


# Default sub-reward weights. Must sum to 1.0.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "constraint": 0.35,
    "aesthetics": 0.25,
    "accessibility": 0.20,
    "coverage": 0.10,
    "efficiency": 0.10,
}


class RewardCalculator:
    """Combines sub-reward components into a scalar reward in [-1.0, 1.0].

    Sub-rewards:
    - constraint: are required elements present per the prompt?
    - aesthetics: layout quality (overlap, alignment, margins, spacing)
    - accessibility: WCAG contrast compliance
    - coverage: canvas area utilization (peaks at ~40%)
    - efficiency: step count penalty (fewer steps = higher score)

    Formula:
        raw = w_constraint * R_constraint
            + w_aesthetics * R_aesthetics
            + w_accessibility * R_accessibility
            + w_coverage * R_coverage
            + w_efficiency * R_efficiency

        reward = clamp(2.0 * raw - 1.0, -1.0, 1.0)

    This maps: raw=0.0 → reward=-1.0, raw=0.5 → reward=0.0, raw=1.0 → reward=1.0.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(_DEFAULT_WEIGHTS)
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
        """Compute the total reward and per-component breakdown.

        Args:
            canvas: Current canvas state.
            prompt: The target prompt for this episode.
            steps_taken: Number of actions taken so far.
            max_steps: Maximum allowed steps.

        Returns:
            (reward, breakdown) where reward ∈ [-1.0, 1.0] and breakdown
            is a dict of sub-scores, each in [0.0, 1.0].
        """
        breakdown = {
            "constraint": self._constraint_checker.score(canvas, prompt.constraints),
            "aesthetics": self._aesthetics_scorer.score(canvas),
            "accessibility": self._accessibility_checker.score(canvas),
            "coverage": self._coverage_score(canvas),
            "efficiency": max(0.0, 1.0 - (steps_taken / max_steps)) if max_steps > 0 else 1.0,
        }

        raw = sum(self.weights[k] * breakdown[k] for k in breakdown)
        reward = max(-1.0, min(1.0, 2.0 * raw - 1.0))

        return reward, breakdown

    def _coverage_score(self, canvas: Canvas) -> float:
        """Score based on total element bounding-box area vs canvas area.

        Uses the sum of individual element areas (not the union), so
        overlapping elements count their full area. The aesthetics overlap
        penalty already discourages stacking, so this is intentional.

        Peaks at ~40% of canvas area. Triangular ramp.

        Returns:
            0.0 for empty canvas or >80% total area.
            1.0 for exactly 40%.
            Range: [0.0, 1.0].
        """
        canvas_area = canvas.config.width * canvas.config.height
        elements = canvas.get_all_elements()
        element_area = sum(e.area for e in elements)
        ratio = min(element_area / canvas_area, 1.0)

        if ratio > 0.8:
            return 0.0

        # Linear ramp: 0 at ratio=0, peak 1.0 at ratio=0.4, back to 0 at ratio=0.8
        if ratio <= 0.4:
            return ratio / 0.4
        return (0.8 - ratio) / 0.4
```

#### 4.7.1 Design Notes

- **`2.0 * raw - 1.0` mapping**: This is crucial. Raw sub-scores are in [0,1]. An agent that does nothing gets raw ≈ 0.1 (only efficiency contributes meaningfully), mapping to reward ≈ -0.8. A perfect design gets raw ≈ 1.0, mapping to reward = 1.0. This gives a clear gradient.
- **Coverage is triangular, not Gaussian**: Linear ramp up to 40%, linear ramp down to 80%. Simpler than a Gaussian, equally effective, zero edge cases.
- **Weights are configurable**: The env can pass custom weights via `reward_weights` param.
- **Breakdown dict is always returned**: Essential for debugging, for the MCP `get_current_reward` tool, and for the info dict.

---

### 4.8 Reward Hacking Mitigations

| Hack | Why It Fails |
|------|-------------|
| Empty canvas (do nothing) | R_constraint = 0 (no elements → no matches), R_aesthetics = 0, R_coverage = 0. Total raw ≈ 0.1 → reward ≈ -0.8 |
| Single huge element covering 40% | R_constraint likely < 1.0 (needs specific content), R_aesthetics low (single element, no alignment), R_accessibility may pass. Total ≈ 0.3 → reward ≈ -0.4 |
| Off-canvas elements | Off-canvas elements don't help constraints (no visible content), aesthetics (bad margins), or coverage (area computed from element bounds, may inflate). Reward stays low |
| Invisible text (same fg/bg color) | R_accessibility = 0 (contrast ratio = 1:1, fails WCAG). Heavily penalized |
| Spam many tiny elements | R_coverage may be ok, but R_aesthetics penalizes overlap. R_constraint needs specific content/type matches. Not a productive strategy |

---

## 5. Layer 3: RL Environment

### 5.1 `env/__init__.py`

```python
"""RL Environment — Gymnasium-compliant MarketCanvas environment."""

from env.market_canvas_env import MarketCanvasEnv

__all__ = ["MarketCanvasEnv"]
```

---

### 5.2 `env/spaces.py` — Space Builders

```python
"""Observation and action space builders for MarketCanvasEnv."""

from __future__ import annotations

import gymnasium
from gymnasium import spaces
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

# Element feature vector layout for observation space:
#   [type_text, type_shape, type_image,  (one-hot, 3)
#    x, y, width, height,               (normalized, 4)
#    color_r, color_g, color_b,          (normalized, 3)
#    text_color_r, text_color_g, text_color_b,  (normalized, 3)
#    font_size,                          (normalized, 1)
#    has_content]                         (binary, 1)
# Total: 15 features per element
NUM_ELEMENT_FEATURES = 15

# Action types
ACTION_ADD_TEXT = 0
ACTION_ADD_SHAPE = 1
ACTION_ADD_IMAGE = 2
ACTION_MOVE = 3
ACTION_RECOLOR = 4
ACTION_REMOVE = 5
ACTION_DONE = 6
NUM_ACTION_TYPES = 7

# Color palette for discrete color selection (16 colors)
COLOR_PALETTE: tuple[str, ...] = (
    "#FFFFFF", "#000000", "#FF0000", "#00FF00",
    "#0000FF", "#FFD700", "#FF6600", "#800080",
    "#00CED1", "#FF69B4", "#228B22", "#8B4513",
    "#708090", "#DC143C", "#4169E1", "#CCCCCC",
)

# Content templates for discrete content selection
CONTENT_TEMPLATES: tuple[str, ...] = (
    "Summer Sale",
    "Shop Now",
    "Buy Today",
    "New Arrival",
    "Limited Offer",
    "Subscribe",
    "Learn More",
    "Get Started",
    "Free Shipping",
    "50% Off",
    "Coming Soon",
    "Join Now",
    "Best Seller",
    "Product Name",
    "Enter Email",
    "Happy Holidays",
    "Flash Sale",
    "Exclusive Deal",
    "Order Now",
    "Sign Up",
)


def build_observation_space(max_elements: int) -> spaces.Dict:
    """Build the Gymnasium observation space.

    Returns a Dict with:
    - elements: Box(max_elements, NUM_ELEMENT_FEATURES) — normalized feature matrix
    - element_mask: MultiBinary(max_elements) — which slots are active
    - step_fraction: Box(1,) — current step / max_steps in [0, 1]
    """
    return spaces.Dict({
        "elements": spaces.Box(
            low=0.0, high=1.0,
            shape=(max_elements, NUM_ELEMENT_FEATURES),
            dtype=np.float32,
        ),
        "element_mask": spaces.MultiBinary(max_elements),
        "step_fraction": spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        ),
    })


def build_action_space(max_elements: int) -> spaces.Dict:
    """Build the Gymnasium action space.

    Returns a Dict with:
    - action_type: Discrete(7) — ADD_TEXT/ADD_SHAPE/ADD_IMAGE/MOVE/RECOLOR/REMOVE/DONE
    - element_idx: Discrete(max_elements) — which element to target (for MOVE/RECOLOR/REMOVE)
    - x: Discrete(800) — X position (for ADD/MOVE)
    - y: Discrete(600) — Y position (for ADD/MOVE)
    - width: Discrete(800) — element width (for ADD, 0 is mapped to minimum)
    - height: Discrete(600) — element height (for ADD, 0 is mapped to minimum)
    - color_idx: Discrete(16) — index into COLOR_PALETTE (for ADD/RECOLOR)
    - content_idx: Discrete(20) — index into CONTENT_TEMPLATES (for ADD)

    Not all fields are meaningful for all action types. Irrelevant fields
    are ignored. Random sampling will produce many no-ops — this is fine
    for exploration and compatibility with check_env.
    """
    return spaces.Dict({
        "action_type": spaces.Discrete(NUM_ACTION_TYPES),
        "element_idx": spaces.Discrete(max_elements),
        "x": spaces.Discrete(800),
        "y": spaces.Discrete(600),
        "width": spaces.Discrete(800),
        "height": spaces.Discrete(600),
        "color_idx": spaces.Discrete(len(COLOR_PALETTE)),
        "content_idx": spaces.Discrete(len(CONTENT_TEMPLATES)),
    })
```

#### 5.2.1 Design Notes

- **Fixed-capacity padded array** for observation: `(max_elements, 15)` with a binary mask. This is the proven pattern from Minigrid (fixed-shape Box), not MiniWoB++'s `Sequence` (which has vectorization issues). RL algorithms (PPO, DQN) expect fixed-shape tensors.
- **All values normalized to [0, 1]** in the observation: coordinates divided by canvas width/height, colors divided by 255, font_size divided by 72 (max reasonable), type as one-hot.
- **REMOVE action (index 5)**: Added beyond the original design. Allows the agent to correct mistakes. MiniWoB++ supports delete actions, and PosterCopilot's iterative editing requires it.
- **Discrete spaces only**: No Box in the action space. This makes the space fully compatible with `MultiDiscrete` flattening for algorithms that don't handle Dict actions.
- **Irrelevant fields are ignored**: When `action_type=DONE`, all other fields are ignored. When `action_type=MOVE`, `width/height/content_idx` are ignored. This is the standard pattern from MiniWoB++ (fields like `coords` are only used for CLICK_COORDS actions).

---

### 5.3 `env/market_canvas_env.py` — The Gymnasium Environment

```python
"""MarketCanvasEnv — Gymnasium-compliant RL environment for design tasks."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from engine.canvas import Canvas
from engine.renderer import CanvasRenderer
from engine.types import CanvasConfig, ElementType

from env.spaces import (
    ACTION_ADD_IMAGE,
    ACTION_ADD_SHAPE,
    ACTION_ADD_TEXT,
    ACTION_DONE,
    ACTION_MOVE,
    ACTION_RECOLOR,
    ACTION_REMOVE,
    COLOR_PALETTE,
    CONTENT_TEMPLATES,
    NUM_ELEMENT_FEATURES,
    build_action_space,
    build_observation_space,
)
from rewards.accessibility import relative_luminance
from rewards.calculator import RewardCalculator
from rewards.prompts import PromptBank, TargetPrompt


class MarketCanvasEnv(gymnasium.Env):
    """MarketCanvas-Env: a minimalist 2D design canvas RL environment.

    The agent's goal is to design a marketing asset (banner, poster)
    that satisfies a target prompt's constraints while maintaining
    good aesthetics and accessibility.

    Follows the Gymnasium Env API: reset(), step(), render(), close().
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        canvas_width: int = 800,
        canvas_height: int = 600,
        max_steps: int = 50,
        max_elements: int = 20,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._canvas_width = canvas_width
        self._canvas_height = canvas_height
        self.max_steps = max_steps
        self.max_elements = max_elements

        # Layer 1: Canvas engine
        self._config = CanvasConfig(
            width=canvas_width,
            height=canvas_height,
            max_elements=max_elements,
        )
        self._canvas = Canvas(self._config)
        self._renderer = CanvasRenderer()

        # Layer 2: Reward engine
        self._reward_calc = RewardCalculator(weights=reward_weights)
        self._prompt_bank = PromptBank()

        # Gymnasium spaces
        self.observation_space = build_observation_space(max_elements)
        self.action_space = build_action_space(max_elements)

        # Episode state (set in reset)
        self._current_prompt: TargetPrompt | None = None
        self._step_count: int = 0

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode.

        1. Clear the canvas
        2. Sample a target prompt
        3. Return initial observation + info
        """
        super().reset(seed=seed)

        self._canvas.clear()
        self._step_count = 0
        self._current_prompt = self._prompt_bank.sample(self.np_random)

        return self._get_obs(), self._get_info()

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one action on the canvas.

        Returns:
            (observation, reward, terminated, truncated, info)

        Reward is 0.0 during the episode. Full reward is computed only
        at episode end (DONE action or max_steps reached).
        """
        # 1. Execute the action
        action_result = self._execute_action(action)
        self._step_count += 1

        # 2. Check termination
        terminated = int(action["action_type"]) == ACTION_DONE
        truncated = self._step_count >= self.max_steps

        # 3. Compute reward (terminal only)
        reward_breakdown: dict[str, Any] = {}
        if terminated or truncated:
            reward, reward_breakdown = self.compute_reward()
        else:
            reward = 0.0

        # 4. Build info
        info = self._get_info(terminal=terminated or truncated)
        info["reward_breakdown"] = reward_breakdown
        info["action_result"] = action_result

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the canvas to an RGB numpy array.

        Only works when render_mode="rgb_array".
        Returns shape (height, width, 3), dtype uint8.
        """
        if self.render_mode == "rgb_array":
            return self._renderer.render_to_array(self._canvas)
        return None

    def get_semantic_state(self) -> dict[str, Any]:
        """Full JSON state for MCP/LLM agents.

        Not part of the Gymnasium API. Used by the MCP server layer.
        """
        state = self._canvas.to_dict()
        state["target_prompt"] = (
            self._current_prompt.text if self._current_prompt else ""
        )
        state["step_count"] = self._step_count
        state["max_steps"] = self.max_steps
        return state

    def compute_reward(self) -> tuple[float, dict[str, Any]]:
        """Compute the reward for the current canvas state.

        Public API for wrappers (e.g. DenseRewardWrapper) to compute reward
        without reaching into private attributes.

        Returns:
            (reward, breakdown) where reward ∈ [-1.0, 1.0].
        """
        return self._reward_calc.calculate(
            self._canvas,
            self._current_prompt,
            self._step_count,
            self.max_steps,
        )

    # ── Action Execution ─────────────────────────────────────────────────────

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Parse and apply an action to the canvas.

        Returns a dict describing the result (for info dict).
        """
        action_type = int(action["action_type"])

        if action_type == ACTION_ADD_TEXT:
            return self._action_add_element(action, ElementType.TEXT)
        if action_type == ACTION_ADD_SHAPE:
            return self._action_add_element(action, ElementType.SHAPE)
        if action_type == ACTION_ADD_IMAGE:
            return self._action_add_element(action, ElementType.IMAGE)
        if action_type == ACTION_MOVE:
            return self._action_move(action)
        if action_type == ACTION_RECOLOR:
            return self._action_recolor(action)
        if action_type == ACTION_REMOVE:
            return self._action_remove(action)
        if action_type == ACTION_DONE:
            return {"action": "done"}

        return {"action": "unknown", "success": False}

    def _action_add_element(
        self, action: dict[str, Any], element_type: ElementType
    ) -> dict[str, Any]:
        """Add an element to the canvas."""
        x = int(action["x"])
        y = int(action["y"])
        width = max(1, int(action["width"]))   # Ensure > 0
        height = max(1, int(action["height"]))  # Ensure > 0
        color_idx = int(action["color_idx"]) % len(COLOR_PALETTE)
        content_idx = int(action["content_idx"]) % len(CONTENT_TEMPLATES)

        color = COLOR_PALETTE[color_idx]

        # Auto-pick text_color for WCAG accessibility:
        # Dark text on light fills, light text on dark fills.
        # Uses WCAG relative luminance to decide the crossover.
        if element_type == ElementType.SHAPE:
            lum = relative_luminance(color)
            text_color = "#000000" if lum > 0.179 else "#FFFFFF"
        else:
            text_color = "#000000"  # TEXT/IMAGE on canvas bg

        element = self._canvas.add_element(
            element_type=element_type,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            text_color=text_color,
            content=CONTENT_TEMPLATES[content_idx],
            font_size=24 if element_type == ElementType.TEXT else 16,
        )

        return {
            "action": f"add_{element_type.value.lower()}",
            "success": element is not None,
            "element_id": element.id if element else None,
        }

    def _action_move(self, action: dict[str, Any]) -> dict[str, Any]:
        """Move an existing element."""
        element_id = self._idx_to_element_id(int(action["element_idx"]))
        if element_id is None:
            return {"action": "move", "success": False}

        success = self._canvas.move_element(element_id, int(action["x"]), int(action["y"]))
        return {"action": "move", "success": success, "element_id": element_id}

    def _action_recolor(self, action: dict[str, Any]) -> dict[str, Any]:
        """Change an element's color."""
        element_id = self._idx_to_element_id(int(action["element_idx"]))
        if element_id is None:
            return {"action": "recolor", "success": False}

        color_idx = int(action["color_idx"]) % len(COLOR_PALETTE)
        success = self._canvas.update_element(
            element_id, color=COLOR_PALETTE[color_idx]
        )
        return {"action": "recolor", "success": success, "element_id": element_id}

    def _action_remove(self, action: dict[str, Any]) -> dict[str, Any]:
        """Remove an element from the canvas."""
        element_id = self._idx_to_element_id(int(action["element_idx"]))
        if element_id is None:
            return {"action": "remove", "success": False}

        success = self._canvas.remove_element(element_id)
        return {"action": "remove", "success": success, "element_id": element_id}

    # ── Observation Building ─────────────────────────────────────────────────

    def _get_obs(self) -> dict[str, Any]:
        """Build a Gymnasium-compatible observation dict.

        Elements are normalized to [0, 1] and padded to max_elements.
        """
        elements = self._canvas.get_all_elements()
        n = len(elements)

        features = np.zeros(
            (self.max_elements, NUM_ELEMENT_FEATURES), dtype=np.float32
        )
        mask = np.zeros(self.max_elements, dtype=np.int8)

        w = float(self._canvas_width)
        h = float(self._canvas_height)

        for i, e in enumerate(elements):
            if i >= self.max_elements:
                break

            # One-hot type encoding
            type_vec = [0.0, 0.0, 0.0]
            if e.type == ElementType.TEXT:
                type_vec[0] = 1.0
            elif e.type == ElementType.SHAPE:
                type_vec[1] = 1.0
            elif e.type == ElementType.IMAGE:
                type_vec[2] = 1.0

            # Normalized coordinates
            norm_x = max(0.0, min(1.0, e.x / w))
            norm_y = max(0.0, min(1.0, e.y / h))
            norm_w = max(0.0, min(1.0, e.width / w))
            norm_h = max(0.0, min(1.0, e.height / h))

            # Normalized colors
            cr, cg, cb = _hex_to_floats(e.color)
            tr, tg, tb = _hex_to_floats(e.text_color)

            # Normalized font size (cap at 72px for normalization)
            norm_fs = min(1.0, e.font_size / 72.0)

            # Has content flag
            has_content = 1.0 if e.content else 0.0

            features[i] = [
                *type_vec,          # 3
                norm_x, norm_y,     # 2
                norm_w, norm_h,     # 2
                cr, cg, cb,         # 3
                tr, tg, tb,         # 3
                norm_fs,            # 1
                has_content,        # 1
            ]                       # Total: 15
            mask[i] = 1

        step_fraction = np.array(
            [self._step_count / self.max_steps if self.max_steps > 0 else 0.0],
            dtype=np.float32,
        )

        return {
            "elements": features,
            "element_mask": mask,
            "step_fraction": step_fraction,
        }

    def _get_info(self, terminal: bool = False) -> dict[str, Any]:
        """Build info dict with human-readable debug data.

        Args:
            terminal: If True, includes full semantic_state (expensive).
                Only set on terminal steps to avoid per-step serialization
                overhead in training (10K envs × 50 steps = 500K calls).
        """
        info: dict[str, Any] = {
            "element_count": self._canvas.element_count,
            "step_count": self._step_count,
            "prompt": self._current_prompt.text if self._current_prompt else "",
        }
        if terminal:
            info["semantic_state"] = self.get_semantic_state()
        return info

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _idx_to_element_id(self, idx: int) -> str | None:
        """Convert an action's element_idx to an actual element ID.

        element_idx is an index into the current element list (z-ordered).
        Returns None if the index is out of bounds.
        """
        elements = self._canvas.get_all_elements()
        if 0 <= idx < len(elements):
            return elements[idx].id
        return None


def _hex_to_floats(hex_color: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' to (r, g, b) floats in [0.0, 1.0]."""
    return (
        int(hex_color[1:3], 16) / 255.0,
        int(hex_color[3:5], 16) / 255.0,
        int(hex_color[5:7], 16) / 255.0,
    )
```

#### 5.3.1 Design Notes

- **`_idx_to_element_id`**: The action space uses indices (0..max_elements-1), not string IDs. This converts. Out-of-bounds → None → no-op.
- **`max(1, width/height)`**: Ensures width/height are always > 0, which the Canvas requires. If the agent samples width=0, it becomes 1.
- **`text_color` defaults**: SHAPE gets white text (buttons typically have light text on colored bg), TEXT gets black text (on canvas bg). The agent can change colors via RECOLOR.
- **`font_size` defaults**: TEXT=24 (readable headline), SHAPE=16 (button labels). Reasonable defaults.
- **Terminal-only reward**: 0.0 mid-episode. Full reward at DONE or truncation. This is the MiniWoB++ pattern and matches REQS.md ("calculates a scalar reward at the end of an episode").
- **No `human` render mode**: Only `rgb_array`. Adding pygame/matplotlib for `human` mode is optional and would add dependencies. The PIL renderer + `rgb_array` satisfies REQS.md's visual output requirement.

---

### 5.4 `env/wrappers.py` — Optional Wrappers

```python
"""Optional wrappers for MarketCanvasEnv."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np


class DenseRewardWrapper(gymnasium.Wrapper):
    """Provides intermediate reward at every step, not just terminal.

    Computes the reward delta between consecutive states.
    Useful for training algorithms that benefit from dense reward signals.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self._last_reward: float = -1.0  # Worst possible at start

    def reset(self, **kwargs: Any) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        # Compute baseline reward for initial (empty) state
        reward, _ = self.unwrapped.compute_reward()
        self._last_reward = reward
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            # Terminal step: use the actual terminal reward
            self._last_reward = reward
            return obs, reward, terminated, truncated, info

        # Mid-episode: compute current reward and return delta
        current_reward, breakdown = self.unwrapped.compute_reward()
        delta = current_reward - self._last_reward
        self._last_reward = current_reward
        info["reward_breakdown"] = breakdown
        return obs, delta, terminated, truncated, info


class PixelObservationWrapper(gymnasium.ObservationWrapper):
    """Adds pixel rendering to the observation space.

    Replaces the Dict observation with one that includes a "pixels" key
    containing the rendered RGB array.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        canvas_h = self.unwrapped._canvas_height
        canvas_w = self.unwrapped._canvas_width

        # Add "pixels" to the observation space
        new_spaces = dict(self.observation_space.spaces)
        new_spaces["pixels"] = gymnasium.spaces.Box(
            low=0, high=255,
            shape=(canvas_h, canvas_w, 3),
            dtype=np.uint8,
        )
        self.observation_space = gymnasium.spaces.Dict(new_spaces)

    def observation(self, obs: dict) -> dict:
        pixels = self.unwrapped._renderer.render_to_array(self.unwrapped._canvas)
        return {**obs, "pixels": pixels}


class FlatActionWrapper(gymnasium.ActionWrapper):
    """Flattens Dict action space to MultiDiscrete.

    Useful for RL algorithms that don't handle Dict action spaces
    (e.g., some stable-baselines3 implementations).

    The MultiDiscrete space is:
    [action_type, element_idx, x, y, width, height, color_idx, content_idx]
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        action_dict = self.env.action_space

        self._keys = list(action_dict.spaces.keys())
        nvec = [action_dict.spaces[k].n for k in self._keys]
        self.action_space = gymnasium.spaces.MultiDiscrete(nvec)

    def action(self, action: np.ndarray) -> dict[str, Any]:
        """Convert MultiDiscrete array back to Dict."""
        return {k: int(action[i]) for i, k in enumerate(self._keys)}
```

#### 5.4.1 Design Notes

- **`DenseRewardWrapper`**: Uses `self.unwrapped` to reach the base env's reward calculator. Returns reward *delta* each step. This is the Minigrid `ActionBonus` pattern — the base env is sparse, wrappers add density.
- **`PixelObservationWrapper`**: Uses dict spread `{**obs, "pixels": pixels}` to add the pixel key without touching other observation keys. This is exactly the Minigrid `RGBImgObsWrapper` pattern.
- **`FlatActionWrapper`**: Converts Dict(Discrete, Discrete, ...) → MultiDiscrete for RL library compatibility. The reverse conversion in `action()` is trivial.
- **These wrappers are optional**: The base env works without them. They exist for training convenience.

---

## 6. Gymnasium Registration

The env should be registerable but NOT auto-registered at import time. Registration happens in `env/__init__.py` only when explicitly called:

```python
# In env/__init__.py, add:
def register_envs() -> None:
    """Register MarketCanvas environments with Gymnasium."""
    from gymnasium.envs.registration import register

    register(
        id="MarketCanvas-v0",
        entry_point="env.market_canvas_env:MarketCanvasEnv",
        max_episode_steps=50,
    )
```

Usage: `import env; env.register_envs(); gymnasium.make("MarketCanvas-v0")`

Or direct instantiation: `from env import MarketCanvasEnv; env = MarketCanvasEnv()`

Both patterns must work.

---

## 7. Tests

### 7.1 `tests/test_rewards.py`

```python
"""Tests for the reward engine.

Test structure:
- TestAccessibilityChecker: WCAG contrast ratio + scoring
- TestAestheticsScorer: overlap, alignment, margin, spacing
- TestConstraintChecker: has_element, element_color, min_elements
- TestRewardCalculator: orchestration, clamping, empty canvas
"""

import pytest
from engine.canvas import Canvas
from engine.types import CanvasConfig, ElementType
from rewards.accessibility import AccessibilityChecker, contrast_ratio, relative_luminance
from rewards.aesthetics import AestheticsScorer
from rewards.calculator import RewardCalculator
from rewards.constraints import ConstraintChecker, _color_distance
from rewards.prompts import ConstraintType, PromptBank, PromptConstraint, TargetPrompt


# ═══════════════════════════════════════════════════════════════════════════
#  Accessibility Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAccessibility:
    def test_relative_luminance_black(self):
        assert relative_luminance("#000000") == pytest.approx(0.0)

    def test_relative_luminance_white(self):
        assert relative_luminance("#FFFFFF") == pytest.approx(1.0)

    def test_contrast_ratio_black_white(self):
        ratio = contrast_ratio("#000000", "#FFFFFF")
        assert ratio == pytest.approx(21.0, abs=0.1)

    def test_contrast_ratio_same_color(self):
        assert contrast_ratio("#FF0000", "#FF0000") == pytest.approx(1.0)

    def test_contrast_ratio_is_symmetric(self):
        r1 = contrast_ratio("#FF0000", "#0000FF")
        r2 = contrast_ratio("#0000FF", "#FF0000")
        assert r1 == pytest.approx(r2)

    def test_score_no_text_elements(self):
        canvas = Canvas()
        canvas.add_element(ElementType.IMAGE, x=0, y=0, width=100, height=100)
        checker = AccessibilityChecker()
        assert checker.score(canvas) == 1.0

    def test_score_empty_canvas(self):
        canvas = Canvas()
        checker = AccessibilityChecker()
        assert checker.score(canvas) == 1.0

    def test_score_black_text_on_white_bg(self):
        """Black text on white canvas background — should pass WCAG."""
        canvas = Canvas(CanvasConfig(background_color="#FFFFFF"))
        canvas.add_element(
            ElementType.TEXT, x=10, y=10, width=100, height=30,
            text_color="#000000", content="Hello",
        )
        checker = AccessibilityChecker()
        assert checker.score(canvas) == 1.0

    def test_score_white_text_on_white_bg_fails(self):
        """White text on white background — fails WCAG."""
        canvas = Canvas(CanvasConfig(background_color="#FFFFFF"))
        canvas.add_element(
            ElementType.TEXT, x=10, y=10, width=100, height=30,
            text_color="#FFFFFF", content="Invisible",
        )
        checker = AccessibilityChecker()
        assert checker.score(canvas) == 0.0

    def test_score_shape_button_contrast(self):
        """SHAPE with content — bg is the shape's own color."""
        canvas = Canvas()
        # White text on dark blue button — should pass
        canvas.add_element(
            ElementType.SHAPE, x=10, y=10, width=200, height=60,
            color="#000080", text_color="#FFFFFF", content="Buy Now",
        )
        checker = AccessibilityChecker()
        assert checker.score(canvas) == 1.0

    def test_score_shape_bad_contrast(self):
        """Yellow text on yellow button — fails."""
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE, x=10, y=10, width=200, height=60,
            color="#FFD700", text_color="#FFFF00", content="Buy Now",
        )
        checker = AccessibilityChecker()
        assert checker.score(canvas) == 0.0

    def test_text_on_shape_background(self):
        """TEXT element placed on top of a SHAPE — bg is the shape color."""
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE, x=0, y=0, width=800, height=600,
            color="#000000",
        )
        canvas.add_element(
            ElementType.TEXT, x=100, y=100, width=200, height=50,
            text_color="#FFFFFF", content="White on Black",
        )
        checker = AccessibilityChecker()
        assert checker.score(canvas) == 1.0

    def test_large_text_lower_threshold(self):
        """Large text (≥ 18px) uses 3.0 threshold instead of 4.5."""
        canvas = Canvas(CanvasConfig(background_color="#FFFFFF"))
        # Gray that passes 3.0 but not 4.5
        canvas.add_element(
            ElementType.TEXT, x=10, y=10, width=200, height=50,
            text_color="#767676", content="Large", font_size=18,
        )
        checker = AccessibilityChecker()
        # #767676 on white has ratio ≈ 4.54 — passes both, need a lighter gray
        # #959595 on white has ratio ≈ 2.85 — fails both
        # #808080 on white has ratio ≈ 3.95 — passes 3.0 but fails 4.5
        canvas2 = Canvas(CanvasConfig(background_color="#FFFFFF"))
        canvas2.add_element(
            ElementType.TEXT, x=10, y=10, width=200, height=50,
            text_color="#808080", content="Large", font_size=18,
        )
        canvas2.add_element(
            ElementType.TEXT, x=10, y=80, width=200, height=50,
            text_color="#808080", content="Small", font_size=14,
        )
        checker2 = AccessibilityChecker()
        score = checker2.score(canvas2)
        # Large text passes (3.95 ≥ 3.0), small text fails (3.95 < 4.5)
        assert score == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════
#  Aesthetics Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAesthetics:
    def test_empty_canvas(self):
        canvas = Canvas()
        scorer = AestheticsScorer()
        assert scorer.score(canvas) == 0.0

    def test_no_overlap(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=100)
        canvas.add_element(ElementType.SHAPE, x=200, y=200, width=100, height=100)
        scorer = AestheticsScorer()
        overlap = scorer._overlap_score(canvas)
        assert overlap == 1.0

    def test_full_overlap(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=100)
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=100)
        scorer = AestheticsScorer()
        overlap = scorer._overlap_score(canvas)
        assert overlap < 1.0

    def test_alignment_perfect_center_x(self):
        canvas = Canvas()
        # All elements centered at x=400
        canvas.add_element(ElementType.SHAPE, x=300, y=0, width=200, height=50)
        canvas.add_element(ElementType.SHAPE, x=300, y=100, width=200, height=50)
        canvas.add_element(ElementType.SHAPE, x=300, y=200, width=200, height=50)
        scorer = AestheticsScorer()
        alignment = scorer._alignment_score(canvas)
        assert alignment == 1.0

    def test_margins_respected(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=30, y=30, width=100, height=50)
        scorer = AestheticsScorer()
        margin = scorer._margin_score(canvas)
        assert margin == 1.0

    def test_margins_violated(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=5, y=5, width=100, height=50)
        scorer = AestheticsScorer()
        margin = scorer._margin_score(canvas)
        assert margin == 0.0

    def test_even_spacing(self):
        canvas = Canvas()
        # Three elements at y=100, y=300, y=500 — even gaps of 200
        canvas.add_element(ElementType.SHAPE, x=100, y=75, width=100, height=50)
        canvas.add_element(ElementType.SHAPE, x=100, y=275, width=100, height=50)
        canvas.add_element(ElementType.SHAPE, x=100, y=475, width=100, height=50)
        scorer = AestheticsScorer()
        spacing = scorer._spacing_score(canvas)
        assert spacing == 1.0

    def test_single_element(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=100, y=100, width=200, height=100)
        scorer = AestheticsScorer()
        # Single element: only margin applies
        score = scorer.score(canvas)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Constraint Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraints:
    def test_empty_constraints(self):
        canvas = Canvas()
        checker = ConstraintChecker()
        assert checker.score(canvas, ()) == 1.0

    def test_has_element_by_type(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=100, height=50, content="Hello")
        checker = ConstraintChecker()
        c = PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "TEXT"})
        assert checker.score(canvas, (c,)) == 1.0

    def test_has_element_missing(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=50)
        checker = ConstraintChecker()
        c = PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "TEXT"})
        assert checker.score(canvas, (c,)) == 0.0

    def test_has_element_with_keyword(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.TEXT, x=0, y=0, width=100, height=50,
            content="Summer Sale"
        )
        checker = ConstraintChecker()
        c = PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "TEXT", "keywords": ["sale"]})
        assert checker.score(canvas, (c,)) == 1.0

    def test_has_element_keyword_no_match(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.TEXT, x=0, y=0, width=100, height=50,
            content="Hello World"
        )
        checker = ConstraintChecker()
        c = PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "TEXT", "keywords": ["sale"]})
        assert checker.score(canvas, (c,)) == 0.0

    def test_element_color_match(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE, x=0, y=0, width=100, height=50,
            color="#FFD700", content="Shop Now"
        )
        checker = ConstraintChecker()
        c = PromptConstraint(ConstraintType.ELEMENT_COLOR, {
            "type": "SHAPE", "keywords": ["shop"],
            "target_color": "#FFD700", "tolerance": 80,
        })
        assert checker.score(canvas, (c,)) == 1.0

    def test_element_color_within_tolerance(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE, x=0, y=0, width=100, height=50,
            color="#FFCC00", content="Shop Now"  # Close to gold
        )
        checker = ConstraintChecker()
        c = PromptConstraint(ConstraintType.ELEMENT_COLOR, {
            "type": "SHAPE", "keywords": ["shop"],
            "target_color": "#FFD700", "tolerance": 80,
        })
        assert checker.score(canvas, (c,)) == 1.0

    def test_min_elements(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=100, height=50)
        canvas.add_element(ElementType.SHAPE, x=0, y=100, width=100, height=50)
        checker = ConstraintChecker()
        c = PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 2})
        assert checker.score(canvas, (c,)) == 1.0

    def test_min_elements_not_met(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=100, height=50)
        checker = ConstraintChecker()
        c = PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 3})
        assert checker.score(canvas, (c,)) == 0.0

    def test_partial_constraint_satisfaction(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.TEXT, x=0, y=0, width=100, height=50,
            content="Summer Sale"
        )
        checker = ConstraintChecker()
        constraints = (
            PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "TEXT", "keywords": ["sale"]}),
            PromptConstraint(ConstraintType.HAS_ELEMENT, {"type": "SHAPE", "keywords": ["buy"]}),
        )
        assert checker.score(canvas, constraints) == pytest.approx(0.5)

    def test_color_distance(self):
        assert _color_distance("#000000", "#000000") == 0.0
        assert _color_distance("#FFFFFF", "#000000") == pytest.approx(441.67, abs=1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Calculator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRewardCalculator:
    def _make_prompt(self) -> TargetPrompt:
        return TargetPrompt(
            text="Test prompt",
            constraints=(PromptConstraint(ConstraintType.MIN_ELEMENTS, {"count": 1}),),
        )

    def test_empty_canvas_negative_reward(self):
        canvas = Canvas()
        calc = RewardCalculator()
        reward, breakdown = calc.calculate(canvas, self._make_prompt(), 0, 50)
        assert reward < 0.0  # Empty canvas should be penalized
        assert -1.0 <= reward <= 1.0

    def test_reward_clamped(self):
        canvas = Canvas()
        calc = RewardCalculator()
        reward, _ = calc.calculate(canvas, self._make_prompt(), 50, 50)
        assert -1.0 <= reward <= 1.0

    def test_breakdown_has_all_keys(self):
        canvas = Canvas()
        calc = RewardCalculator()
        _, breakdown = calc.calculate(canvas, self._make_prompt(), 10, 50)
        expected_keys = {"constraint", "aesthetics", "accessibility", "coverage", "efficiency"}
        assert set(breakdown.keys()) == expected_keys

    def test_all_sub_scores_in_range(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=100, y=100, width=200, height=50, content="Sale")
        calc = RewardCalculator()
        _, breakdown = calc.calculate(canvas, self._make_prompt(), 10, 50)
        for k, v in breakdown.items():
            assert 0.0 <= v <= 1.0, f"{k} out of range: {v}"

    def test_coverage_peak(self):
        """Canvas with ~40% element coverage should have high coverage score."""
        canvas = Canvas()
        # 800x600 = 480,000 sq px. 40% = 192,000 sq px. ~438x438
        canvas.add_element(ElementType.SHAPE, x=100, y=50, width=438, height=438)
        calc = RewardCalculator()
        _, breakdown = calc.calculate(canvas, self._make_prompt(), 1, 50)
        assert breakdown["coverage"] > 0.9

    def test_coverage_empty(self):
        canvas = Canvas()
        calc = RewardCalculator()
        _, breakdown = calc.calculate(canvas, self._make_prompt(), 0, 50)
        assert breakdown["coverage"] == 0.0

    def test_efficiency_decreases_with_steps(self):
        canvas = Canvas()
        calc = RewardCalculator()
        _, b1 = calc.calculate(canvas, self._make_prompt(), 10, 50)
        _, b2 = calc.calculate(canvas, self._make_prompt(), 40, 50)
        assert b1["efficiency"] > b2["efficiency"]

    def test_custom_weights(self):
        canvas = Canvas()
        weights = {
            "constraint": 1.0,
            "aesthetics": 0.0,
            "accessibility": 0.0,
            "coverage": 0.0,
            "efficiency": 0.0,
        }
        calc = RewardCalculator(weights=weights)
        reward, _ = calc.calculate(canvas, self._make_prompt(), 0, 50)
        # Constraint score is 0 (empty canvas, needs 1 element)
        # raw = 0.0, reward = 2*0 - 1 = -1.0
        assert reward == pytest.approx(-1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Bank Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptBank:
    def test_sample_returns_target_prompt(self):
        import numpy as np
        bank = PromptBank()
        rng = np.random.default_rng(42)
        prompt = bank.sample(rng)
        assert isinstance(prompt, TargetPrompt)
        assert len(prompt.text) > 0
        assert len(prompt.constraints) > 0

    def test_sample_is_deterministic_with_seed(self):
        import numpy as np
        bank = PromptBank()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        assert bank.sample(rng1).text == bank.sample(rng2).text

    def test_all_prompts_have_constraints(self):
        bank = PromptBank()
        for prompt in bank.PROMPTS:
            assert len(prompt.constraints) > 0
            assert len(prompt.text) > 0
```

### 7.2 `tests/test_env.py`

```python
"""Tests for the Gymnasium RL environment.

Test structure:
- TestEnvCreation: construction, spaces
- TestEnvReset: reset API, seeding
- TestEnvStep: action execution, termination, truncation
- TestEnvCheckEnv: Gymnasium compliance via check_env
- TestWrappers: DenseRewardWrapper, PixelObservationWrapper, FlatActionWrapper
"""

import pytest
import numpy as np
import gymnasium

from env.market_canvas_env import MarketCanvasEnv
from env.spaces import (
    ACTION_ADD_TEXT,
    ACTION_ADD_SHAPE,
    ACTION_DONE,
    ACTION_MOVE,
    ACTION_RECOLOR,
    ACTION_REMOVE,
    NUM_ELEMENT_FEATURES,
)
from env.wrappers import DenseRewardWrapper, FlatActionWrapper, PixelObservationWrapper


# ═══════════════════════════════════════════════════════════════════════════
#  Environment Creation
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvCreation:
    def test_default_construction(self):
        env = MarketCanvasEnv()
        assert env.observation_space is not None
        assert env.action_space is not None
        env.close()

    def test_custom_params(self):
        env = MarketCanvasEnv(
            canvas_width=400, canvas_height=300,
            max_steps=25, max_elements=10,
        )
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


# ═══════════════════════════════════════════════════════════════════════════
#  Reset
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvReset:
    def test_reset_returns_obs_and_info(self):
        env = MarketCanvasEnv()
        obs, info = env.reset(seed=42)
        assert "elements" in obs
        assert "element_mask" in obs
        assert "step_fraction" in obs
        assert "prompt" in info
        assert len(info["prompt"]) > 0
        env.close()

    def test_reset_clears_canvas(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        # Add an element
        action = env.action_space.sample()
        action["action_type"] = ACTION_ADD_TEXT
        env.step(action)
        # Reset should clear
        obs, info = env.reset(seed=42)
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


# ═══════════════════════════════════════════════════════════════════════════
#  Step
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvStep:
    def _make_add_action(self, action_type=ACTION_ADD_TEXT, **overrides):
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
        action = self._make_add_action()
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["element_count"] == 1
        assert info["action_result"]["success"] is True
        assert reward == 0.0  # Mid-episode
        assert not terminated
        env.close()

    def test_done_terminates(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        action = self._make_add_action(action_type=ACTION_DONE)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert truncated is False
        assert reward != 0.0  # Terminal reward computed
        env.close()

    def test_truncation_at_max_steps(self):
        env = MarketCanvasEnv(max_steps=3)
        env.reset(seed=42)
        for i in range(3):
            action = self._make_add_action()
            obs, reward, terminated, truncated, info = env.step(action)
        assert truncated is True
        assert reward != 0.0
        env.close()

    def test_move_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        env.step(self._make_add_action())
        move_action = self._make_add_action(
            action_type=ACTION_MOVE, element_idx=0, x=300, y=300
        )
        _, _, _, _, info = env.step(move_action)
        assert info["action_result"]["success"] is True
        env.close()

    def test_move_nonexistent_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        move_action = self._make_add_action(
            action_type=ACTION_MOVE, element_idx=5, x=300, y=300
        )
        _, _, _, _, info = env.step(move_action)
        assert info["action_result"]["success"] is False
        env.close()

    def test_recolor_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        env.step(self._make_add_action())
        recolor = self._make_add_action(
            action_type=ACTION_RECOLOR, element_idx=0, color_idx=2  # Red
        )
        _, _, _, _, info = env.step(recolor)
        assert info["action_result"]["success"] is True
        env.close()

    def test_remove_element(self):
        env = MarketCanvasEnv()
        env.reset(seed=42)
        env.step(self._make_add_action())
        assert env._canvas.element_count == 1
        remove = self._make_add_action(action_type=ACTION_REMOVE, element_idx=0)
        _, _, _, _, info = env.step(remove)
        assert info["action_result"]["success"] is True
        assert env._canvas.element_count == 0
        env.close()

    def test_random_actions_dont_crash(self):
        env = MarketCanvasEnv(max_steps=20)
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        env.close()

    def test_observation_always_in_space(self):
        env = MarketCanvasEnv(max_steps=10)
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        env.close()

    def test_reward_in_range(self):
        env = MarketCanvasEnv(max_steps=5)
        env.reset(seed=42)
        for _ in range(5):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            assert -1.0 <= reward <= 1.0
            if terminated or truncated:
                break
        env.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  Semantic State
# ═══════════════════════════════════════════════════════════════════════════

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
        env.close()


# ═══════════════════════════════════════════════════════════════════════════
#  check_env Compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckEnv:
    def test_check_env_passes(self):
        """The environment must pass Gymnasium's built-in compliance check."""
        env = MarketCanvasEnv()
        from gymnasium.utils.env_checker import check_env
        # check_env raises if anything is wrong
        check_env(env.unwrapped)
        env.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Wrappers
# ═══════════════════════════════════════════════════════════════════════════

class TestWrappers:
    def test_dense_reward_wrapper(self):
        env = DenseRewardWrapper(MarketCanvasEnv(max_steps=10))
        obs, info = env.reset(seed=42)
        # Mid-episode should get non-zero reward (delta)
        action = env.action_space.sample()
        action["action_type"] = ACTION_ADD_TEXT
        obs, reward, terminated, truncated, info = env.step(action)
        # Reward can be any float (delta), not necessarily 0
        assert isinstance(reward, float)
        env.close()

    def test_pixel_observation_wrapper(self):
        env = PixelObservationWrapper(MarketCanvasEnv())
        obs, _ = env.reset(seed=42)
        assert "pixels" in obs
        assert obs["pixels"].shape == (600, 800, 3)
        assert obs["pixels"].dtype == np.uint8
        assert env.observation_space.contains(obs)
        env.close()

    def test_flat_action_wrapper(self):
        env = FlatActionWrapper(MarketCanvasEnv())
        assert isinstance(env.action_space, gymnasium.spaces.MultiDiscrete)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        env.close()
```

---

## 8. Edge Cases & Behavioral Notes

| Scenario | Expected Behavior | Rationale |
|---|---|---|
| Agent adds element when canvas is full | `add_element()` returns None, action_result success=False | Canvas engine handles this already |
| Agent moves/recolors/removes a nonexistent element | action_result success=False, canvas unchanged | `_idx_to_element_id` returns None |
| Agent sends DONE on step 1 | Episode ends, terminal reward computed on near-empty canvas | Legal. Reward will be low (~-0.8) |
| All sub-rewards are 0 | raw=0.0, reward=-1.0 | Empty canvas + failed constraints + 100% steps used |
| All sub-rewards are 1 | raw=1.0, reward=1.0 | Perfect design on first step |
| Prompt has no constraints | R_constraint=1.0 (vacuously true) | Correct. Agent can still score from other sub-rewards |
| Canvas has text but no content string | Not counted as text-bearing for WCAG | Element must have content to be accessibility-relevant |
| Element partially off-canvas | Legal. Aesthetics margin_score penalizes it. Coverage uses element.area | Engine accepts it, reward penalizes it |
| width=0 or height=0 in ADD action | Mapped to width=1 or height=1 | `max(1, ...)` in action handler |
| Random action sampling | Many no-ops (e.g., MOVE with invalid element_idx) | Expected. check_env requires valid samples |

---

## 9. Performance Characteristics

| Operation | Complexity | Notes |
|---|---|---|
| `ConstraintChecker.score()` | O(C × N) | C constraints × N elements, both small |
| `AccessibilityChecker.score()` | O(N²) | For each text element, get_element_behind is O(N) |
| `AestheticsScorer.score()` | O(N²) | Overlap pairs + alignment clustering |
| `RewardCalculator.calculate()` | O(N²) | Dominated by aesthetics/accessibility |
| `MarketCanvasEnv.step()` | O(N²) + O(N) | Reward (terminal only) + observation building |
| `MarketCanvasEnv._get_obs()` | O(N) | Linear scan of elements, normalize + pad |
| `PixelObservationWrapper` | O(N) + PIL | Rendering is the bottleneck (~5ms) |

With max_elements=20, all operations are effectively O(1) in practice.

---

## 10. How to Verify

```bash
# From project root (mini-canva/)
# 1. Install dependencies
uv pip install -e ".[dev]" gymnasium

# 2. Run reward engine tests
pytest tests/test_rewards.py -v

# 3. Run environment tests
pytest tests/test_env.py -v

# 4. Run all tests (including existing engine tests)
pytest -v

# 5. Quick smoke test
python -c "
from env import MarketCanvasEnv
env = MarketCanvasEnv()
obs, info = env.reset(seed=42)
print(f'Prompt: {info[\"prompt\"]}')
for i in range(5):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f'Step {i+1}: reward={reward:.3f}, elements={info[\"element_count\"]}')
    if term or trunc:
        print(f'Episode ended. Final reward: {reward:.3f}')
        break
env.close()
"
```

---

## 11. What's NOT in This Spec (Phase 3)

1. **MCP Server** (`mcp/server.py`) — FastMCP wrapper around MarketCanvasEnv
2. **Demo Script** (`demo.py`) — Required deliverable: programmatic + random actions
3. **WRITEUP.md** — Required deliverable: design reasoning + scaling analysis
4. **Gymnasium registration via entry points** — Currently manual; could be automated in pyproject.toml
5. **Low-level action space** — REQS.md says "one or both"; we implement high-level only, discuss trade-offs in WRITEUP.md
