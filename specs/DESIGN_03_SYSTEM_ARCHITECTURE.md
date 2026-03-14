# Design Document 03: End-to-End System Architecture — MarketCanvas-Env

> The complete architecture for building MarketCanvas-Env from scratch, unifying the canvas engine, RL environment, MCP server, and demo script into a single, coherent system.

---

## 1. System Overview

MarketCanvas-Env is a system with four layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server Layer                          │
│  (FastMCP — exposes canvas operations as LLM-callable tools)│
│                                                             │
│  Tools: get_canvas_state, execute_action, get_current_reward│
├─────────────────────────────────────────────────────────────┤
│                  RL Environment Layer                        │
│  (Gymnasium Env — MarketCanvasEnv)                          │
│                                                             │
│  API: reset(), step(), render(), close()                    │
│  Spaces: observation_space, action_space                    │
├─────────────────────────────────────────────────────────────┤
│               Reward Engine Layer                            │
│  (Heuristic reward calculator)                              │
│                                                             │
│  Components: ConstraintChecker, AestheticsScorer,           │
│              AccessibilityChecker, CoverageScorer           │
├─────────────────────────────────────────────────────────────┤
│                Canvas Engine Layer                           │
│  (Pure Python 2D canvas simulator)                          │
│                                                             │
│  Core: CanvasState, Element, CanvasRenderer                 │
│  Operations: add, remove, move, recolor, serialize          │
└─────────────────────────────────────────────────────────────┘
```

**Key principle**: Each layer depends only on the layer below it. The canvas engine knows nothing about RL. The RL env knows nothing about MCP. Dependencies flow strictly downward.

---

## 2. Project Structure

```
mini-canva/
├── pyproject.toml                    # Project metadata, dependencies
├── README.md
├── specs/
│   ├── REQS.md                      # Original requirements
│   ├── DESIGN_01_CANVAS_ARCHITECTURE.md
│   ├── DESIGN_02_RL_ENVIRONMENT.md
│   └── DESIGN_03_SYSTEM_ARCHITECTURE.md  (this document)
│
├── market_canvas/                   # Main package
│   ├── __init__.py
│   │
│   ├── engine/                      # Layer 1: Canvas Engine
│   │   ├── __init__.py
│   │   ├── types.py                 # Element, ElementType, CanvasConfig
│   │   ├── canvas.py                # CanvasState (the scene graph)
│   │   └── renderer.py              # PIL-based rendering
│   │
│   ├── rewards/                     # Layer 2: Reward Engine
│   │   ├── __init__.py
│   │   ├── calculator.py            # RewardCalculator (orchestrator)
│   │   ├── constraints.py           # ConstraintChecker
│   │   ├── aesthetics.py            # AestheticsScorer
│   │   ├── accessibility.py         # AccessibilityChecker (WCAG)
│   │   └── prompts.py              # PromptBank + constraint parsing
│   │
│   ├── env/                         # Layer 3: RL Environment
│   │   ├── __init__.py
│   │   ├── market_canvas_env.py     # MarketCanvasEnv (gymnasium.Env)
│   │   ├── spaces.py               # Observation/action space builders
│   │   └── wrappers.py             # Optional wrappers (pixel obs, etc.)
│   │
│   └── mcp/                         # Layer 4: MCP Server
│       ├── __init__.py
│       └── server.py               # FastMCP server wrapping the env
│
├── demo.py                          # Demo script (required deliverable)
├── WRITEUP.md                       # Write-up (required deliverable)
└── tests/
    ├── test_engine.py               # Canvas engine unit tests
    ├── test_rewards.py              # Reward calculation tests
    ├── test_env.py                  # Gymnasium env compliance tests
    └── test_mcp.py                  # MCP server integration tests
```

---

## 3. Layer 1: Canvas Engine (Detailed Design)

### 3.1 `types.py` — Data Types

```python
# Core data types (using dataclasses for simplicity + JSON serialization)

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

class ElementType(str, Enum):
    TEXT = "TEXT"
    SHAPE = "SHAPE"
    IMAGE = "IMAGE"

@dataclass
class Element:
    id: str
    type: ElementType
    x: int
    y: int
    width: int
    height: int
    z_index: int = 0
    color: str = "#CCCCCC"         # Fill / background color
    text_color: str = "#000000"    # Text color
    content: str = ""              # Text content or label
    font_size: int = 16            # Font size in pixels
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Returns (left, top, right, bottom)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

@dataclass
class CanvasConfig:
    width: int = 800
    height: int = 600
    background_color: str = "#FFFFFF"
    max_elements: int = 20
```

### 3.2 `canvas.py` — CanvasState (The Scene Graph)

The central class. All mutations go through this class. It IS the state.

```python
class CanvasState:
    """
    Pure data container representing the canvas.
    No rendering, no RL logic, no side effects.
    All methods return mutation results, making the state trivially serializable.
    """
    
    def __init__(self, config: CanvasConfig):
        self.config = config
        self._elements: dict[str, Element] = {}  # id → Element (O(1) lookup)
        self._next_id: int = 0
    
    # === Queries ===
    def get_element(self, element_id: str) -> Optional[Element]
    def get_all_elements(self) -> list[Element]   # sorted by z_index
    def element_count(self) -> int
    
    # === Mutations ===
    def add_element(self, type: ElementType, x, y, w, h, **kwargs) -> Element
    def remove_element(self, element_id: str) -> bool
    def move_element(self, element_id: str, new_x: int, new_y: int) -> bool
    def resize_element(self, element_id: str, new_w: int, new_h: int) -> bool
    def change_color(self, element_id: str, color: str) -> bool
    def change_text_color(self, element_id: str, text_color: str) -> bool
    def set_content(self, element_id: str, content: str) -> bool
    
    # === Serialization ===
    def to_dict(self) -> dict        # Full JSON-serializable state
    def clear(self) -> None          # Reset to empty canvas
    
    # === Spatial queries (for reward computation) ===
    def get_overlapping_pairs(self) -> list[tuple[Element, Element, float]]
    def get_elements_in_bounds(self, l, t, r, b) -> list[Element]
```

**Design decisions**:
- **Dict storage** (`_elements`): `O(1)` add/remove/lookup by ID. List view sorted by z_index on demand.
- **No validation in mutations**: The canvas engine accepts any valid element. The reward function handles "bad" designs.
- **Immutable-friendly**: `to_dict()` produces a snapshot. Future vectorized envs can share state cheaply.

### 3.3 `renderer.py` — PIL-Based Rendering

```python
class CanvasRenderer:
    """
    Renders a CanvasState to a PIL Image or numpy array.
    Stateless — takes a CanvasState, returns an image.
    """
    
    def render(self, state: CanvasState) -> PIL.Image.Image:
        """Renders the canvas state to a PIL Image."""
        # 1. Create blank image with background_color
        # 2. Sort elements by z_index (ascending)
        # 3. For each element:
        #    - SHAPE: draw filled rectangle, then text overlay if content
        #    - TEXT: draw text at position with font_size
        #    - IMAGE: draw colored rectangle (placeholder)
        # 4. Return PIL Image
    
    def render_to_array(self, state: CanvasState) -> np.ndarray:
        """Returns (height, width, 3) uint8 numpy array."""
        return np.array(self.render(state))
    
    def save(self, state: CanvasState, path: str) -> None:
        """Saves canvas as PNG."""
        self.render(state).save(path)
```

---

## 4. Layer 2: Reward Engine (Detailed Design)

### 4.1 Architecture

```
                    ┌──────────────────┐
                    │ RewardCalculator │
                    │  (orchestrator)  │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
  ┌───────▼───────┐  ┌──────▼──────┐  ┌────────▼────────┐
  │ Constraint    │  │ Aesthetics  │  │ Accessibility   │
  │ Checker       │  │ Scorer      │  │ Checker (WCAG)  │
  └───────────────┘  └─────────────┘  └─────────────────┘
```

### 4.2 `prompts.py` — Prompt Bank & Constraint Definitions

```python
@dataclass
class PromptConstraint:
    """A single verifiable constraint extracted from a prompt."""
    type: str          # "has_element", "element_color", "contrast", "min_elements"
    params: dict       # Type-specific parameters

@dataclass 
class TargetPrompt:
    """A design task with its natural language description and machine-checkable constraints."""
    text: str                           # Human-readable prompt
    constraints: list[PromptConstraint] # Machine-checkable constraints
    difficulty: str = "easy"            # "easy", "medium", "hard"

class PromptBank:
    """Repository of target prompts for episode initialization."""
    
    PROMPTS: list[TargetPrompt] = [
        TargetPrompt(
            text="Create a Summer Sale email banner with a headline, yellow CTA button, and good contrast",
            constraints=[
                PromptConstraint("has_element", {"type": "TEXT", "role": "headline", "keywords": ["sale", "summer"]}),
                PromptConstraint("has_element", {"type": "SHAPE", "role": "button", "keywords": ["shop", "buy", "cta"]}),
                PromptConstraint("element_color", {"role": "button", "target_color": "#FFD700", "tolerance": 60}),
                PromptConstraint("contrast", {"min_ratio": 4.5}),
            ],
            difficulty="easy"
        ),
        TargetPrompt(
            text="Design a product launch announcement with a hero image, product name, and launch date",
            constraints=[
                PromptConstraint("has_element", {"type": "IMAGE", "role": "hero"}),
                PromptConstraint("has_element", {"type": "TEXT", "role": "title", "keywords": ["product", "launch"]}),
                PromptConstraint("has_element", {"type": "TEXT", "role": "date"}),
                PromptConstraint("min_elements", {"count": 3}),
            ],
            difficulty="easy"
        ),
        # ... additional prompts for variety
    ]
    
    def sample(self, rng, difficulty=None) -> TargetPrompt:
        """Sample a random prompt, optionally filtered by difficulty."""
        pool = self.PROMPTS if difficulty is None else [p for p in self.PROMPTS if p.difficulty == difficulty]
        return rng.choice(pool)
```

### 4.3 `constraints.py` — Constraint Checker

```python
class ConstraintChecker:
    """Checks if a canvas state satisfies a list of constraints."""
    
    def check(self, state: CanvasState, constraints: list[PromptConstraint]) -> float:
        """Returns fraction of satisfied constraints [0.0, 1.0]."""
        if not constraints:
            return 1.0
        satisfied = sum(1 for c in constraints if self._check_one(state, c))
        return satisfied / len(constraints)
    
    def _check_one(self, state: CanvasState, constraint: PromptConstraint) -> bool:
        # Dispatches to specific checker based on constraint.type
        # "has_element" → check if element of given type exists with matching keywords
        # "element_color" → check if element color is within tolerance of target
        # "contrast" → check if all text elements meet min contrast ratio
        # "min_elements" → check if element count >= threshold
```

### 4.4 `aesthetics.py` — Aesthetics Scorer

```python
class AestheticsScorer:
    """Scores visual quality of element layout."""
    
    def score(self, state: CanvasState) -> float:
        """Returns aesthetics score [0.0, 1.0]."""
        if state.element_count() == 0:
            return 0.0
        
        scores = [
            self._overlap_score(state),    # Penalize overlapping elements
            self._alignment_score(state),  # Reward aligned elements  
            self._margin_score(state),     # Reward proper margins from edges
            self._spacing_score(state),    # Reward even vertical spacing
        ]
        return sum(scores) / len(scores)
    
    def _overlap_score(self, state: CanvasState) -> float:
        """1.0 = no overlaps, 0.0 = complete overlap."""
        # For each pair of elements, compute intersection area
        # Return 1.0 - (total_overlap / total_element_area)
    
    def _alignment_score(self, state: CanvasState) -> float:
        """Score based on how many elements share common alignment axes."""
        # Check center-X alignment, center-Y alignment, left-edge alignment
        # Use tolerance window of ±5 pixels
        # Score = max alignment fraction across all axes
    
    def _margin_score(self, state: CanvasState) -> float:
        """Score based on elements having proper margins from canvas edges."""
        # MIN_MARGIN = 20px
        # fraction of elements with all edges >= MIN_MARGIN from canvas edges
    
    def _spacing_score(self, state: CanvasState) -> float:
        """Score based on regularity of vertical gaps between elements."""
        # Sort elements by center Y
        # Compute gaps between consecutive elements
        # Score = 1.0 - normalized_stddev(gaps)
```

### 4.5 `accessibility.py` — WCAG Checker

```python
class AccessibilityChecker:
    """Checks WCAG 2.1 AA contrast compliance."""
    
    def score(self, state: CanvasState) -> float:
        """Returns fraction of text elements meeting WCAG AA contrast [0.0, 1.0]."""
        text_elements = [e for e in state.get_all_elements() if e.type in (ElementType.TEXT, ElementType.SHAPE) and e.content]
        if not text_elements:
            return 1.0  # No text = no violations
        
        passing = sum(1 for e in text_elements if self._check_contrast(e, state))
        return passing / len(text_elements)
    
    def _check_contrast(self, element: Element, state: CanvasState) -> bool:
        """Check if element's text contrast meets WCAG AA."""
        bg_color = self._get_effective_background(element, state)
        ratio = self._contrast_ratio(element.text_color, bg_color)
        threshold = 3.0 if element.font_size >= 18 else 4.5  # Large text = lower threshold
        return ratio >= threshold
    
    def _get_effective_background(self, element: Element, state: CanvasState) -> str:
        """Determine what color is behind this element."""
        # If element is TEXT type, check for SHAPE elements behind it (lower z-index, overlapping bounds)
        # If none found, use canvas background_color
        # If element is SHAPE type, use its own color as background for its text content
    
    @staticmethod
    def _contrast_ratio(color1: str, color2: str) -> float:
        """WCAG 2.1 contrast ratio calculation."""
        l1 = AccessibilityChecker._relative_luminance(color1)
        l2 = AccessibilityChecker._relative_luminance(color2)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)
    
    @staticmethod
    def _relative_luminance(hex_color: str) -> float:
        """Convert hex color to relative luminance per WCAG 2.1."""
        r, g, b = int(hex_color[1:3], 16)/255, int(hex_color[3:5], 16)/255, int(hex_color[5:7], 16)/255
        # Apply gamma correction
        r = r/12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g/12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b/12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
```

### 4.6 `calculator.py` — Reward Orchestrator

```python
class RewardCalculator:
    """Combines all sub-reward components into a final scalar reward."""
    
    DEFAULT_WEIGHTS = {
        "constraint": 0.35,
        "aesthetics": 0.25,
        "accessibility": 0.20,
        "coverage": 0.10,
        "efficiency": 0.10,
    }
    
    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.constraint_checker = ConstraintChecker()
        self.aesthetics_scorer = AestheticsScorer()
        self.accessibility_checker = AccessibilityChecker()
    
    def calculate(self, state: CanvasState, prompt: TargetPrompt, 
                  steps_taken: int, max_steps: int) -> tuple[float, dict]:
        """
        Returns (total_reward, breakdown).
        total_reward is clamped to [-1.0, 1.0].
        breakdown is a dict of sub-scores for debugging.
        """
        breakdown = {
            "constraint": self.constraint_checker.check(state, prompt.constraints),
            "aesthetics": self.aesthetics_scorer.score(state),
            "accessibility": self.accessibility_checker.score(state),
            "coverage": self._coverage_score(state),
            "efficiency": max(0, 1.0 - steps_taken / max_steps),
        }
        
        total = sum(self.weights[k] * breakdown[k] for k in breakdown)
        # Shift to [-1, 1] range: 0 score → -1, 1.0 score → 1.0
        total = 2.0 * total - 1.0
        total = max(-1.0, min(1.0, total))
        
        return total, breakdown
    
    def _coverage_score(self, state: CanvasState) -> float:
        """Score canvas area utilization."""
        canvas_area = state.config.width * state.config.height
        element_area = sum(e.width * e.height for e in state.get_all_elements())
        ratio = min(element_area / canvas_area, 1.0)
        # Peak at ~40% coverage
        return 1.0 - abs(ratio - 0.4) / 0.4 if ratio <= 0.8 else 0.0
```

---

## 5. Layer 3: RL Environment (Detailed Design)

### 5.1 `market_canvas_env.py` — The Gymnasium Environment

```python
class MarketCanvasEnv(gymnasium.Env):
    """
    MarketCanvas-Env: A minimalist 2D design canvas RL environment.
    
    The agent's goal is to design a marketing asset (banner, poster)
    that satisfies a target prompt's constraints while maintaining
    good aesthetics and accessibility.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, canvas_width=800, canvas_height=600,
                 max_steps=50, max_elements=20, action_mode="high_level",
                 reward_weights=None):
        super().__init__()
        
        self.config = CanvasConfig(width=canvas_width, height=canvas_height, 
                                    max_elements=max_elements)
        self.canvas = CanvasState(self.config)
        self.renderer = CanvasRenderer()
        self.reward_calc = RewardCalculator(weights=reward_weights)
        self.prompt_bank = PromptBank()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_mode = action_mode
        
        # Define spaces (via spaces.py helper)
        self.observation_space = build_observation_space(max_elements, canvas_width, canvas_height)
        self.action_space = build_action_space(action_mode, max_elements, canvas_width, canvas_height)
        
        # Episode state
        self._current_prompt = None
        self._step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.canvas.clear()
        self._step_count = 0
        self._current_prompt = self.prompt_bank.sample(self.np_random)
        
        observation = self._get_observation()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        # 1. Parse and execute action on canvas
        action_result = self._execute_action(action)
        self._step_count += 1
        
        # 2. Check termination
        done_action = self._is_done_action(action)
        terminated = done_action
        truncated = self._step_count >= self.max_steps
        
        # 3. Compute reward (only at episode end for efficiency, or every step)
        if terminated or truncated:
            reward, reward_breakdown = self.reward_calc.calculate(
                self.canvas, self._current_prompt, self._step_count, self.max_steps
            )
        else:
            reward = 0.0
            reward_breakdown = {}
        
        # 4. Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = reward_breakdown
        info["action_result"] = action_result
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self.renderer.render_to_array(self.canvas)
        elif self.render_mode == "human":
            # Display using matplotlib or pygame
            pass
    
    def get_semantic_state(self) -> dict:
        """Returns full JSON state for MCP/LLM agents."""
        state = self.canvas.to_dict()
        state["target_prompt"] = self._current_prompt.text if self._current_prompt else ""
        state["step_count"] = self._step_count
        state["max_steps"] = self.max_steps
        return state
    
    def _execute_action(self, action) -> dict:
        """Parse action and apply mutation to canvas. Returns result metadata."""
        # Dispatch based on action_type:
        # ADD_TEXT → canvas.add_element(ElementType.TEXT, ...)
        # ADD_SHAPE → canvas.add_element(ElementType.SHAPE, ...)
        # ADD_IMAGE → canvas.add_element(ElementType.IMAGE, ...)
        # MOVE_ELEMENT → canvas.move_element(id, x, y)
        # CHANGE_COLOR → canvas.change_color(id, color)
        # DONE → no-op (signals episode end)
    
    def _get_observation(self) -> dict:
        """Build gymnasium-compatible observation dict."""
        # Convert canvas elements to fixed-shape numpy arrays
        # Normalize coordinates to [0, 1]
        # Pad unused element slots with zeros
        # Include element_mask, prompt_embedding, step_fraction
    
    def _get_info(self) -> dict:
        """Build info dict with human-readable debug data."""
        return {
            "element_count": self.canvas.element_count(),
            "step_count": self._step_count,
            "prompt": self._current_prompt.text if self._current_prompt else "",
            "semantic_state": self.get_semantic_state(),
        }
```

### 5.2 `spaces.py` — Space Builders

```python
def build_observation_space(max_elements, canvas_w, canvas_h):
    """Builds the gymnasium observation space."""
    NUM_FEATURES = 10  # x, y, w, h, z, type_onehot(3), has_content
    return spaces.Dict({
        "elements": spaces.Box(low=0, high=1, shape=(max_elements, NUM_FEATURES), dtype=np.float32),
        "element_mask": spaces.MultiBinary(max_elements),
        "step_fraction": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
    })

def build_action_space(mode, max_elements, canvas_w, canvas_h):
    """Builds the gymnasium action space."""
    if mode == "high_level":
        return spaces.Dict({
            "action_type": spaces.Discrete(6),        # ADD_TEXT, ADD_SHAPE, ADD_IMAGE, MOVE, RECOLOR, DONE
            "element_idx": spaces.Discrete(max_elements),
            "x": spaces.Discrete(canvas_w),
            "y": spaces.Discrete(canvas_h),
            "width": spaces.Discrete(canvas_w),
            "height": spaces.Discrete(canvas_h),
            "color_idx": spaces.Discrete(16),          # Palette of 16 colors
            "content_idx": spaces.Discrete(20),        # 20 content templates
        })
    else:
        raise NotImplementedError("Low-level action space not yet supported")
```

### 5.3 `wrappers.py` — Optional Wrappers

```python
class PixelObservationWrapper(gymnasium.ObservationWrapper):
    """Adds pixel rendering to the observation space."""
    # Wraps observation to include "pixels" key with RGB array

class FlattenActionWrapper(gymnasium.ActionWrapper):
    """Flattens Dict action space to MultiDiscrete for simpler RL algorithms."""

class DenseRewardWrapper(gymnasium.Wrapper):
    """Provides intermediate reward at every step (not just terminal)."""
    # Computes reward delta between consecutive states
```

---

## 6. Layer 4: MCP Server (Detailed Design)

### 6.1 `server.py` — FastMCP Integration

```python
from fastmcp import FastMCP
from market_canvas.env import MarketCanvasEnv

mcp = FastMCP("MarketCanvas-MCP")

# Global environment instance (single session)
env = None

@mcp.tool
def initialize_env(canvas_width: int = 800, canvas_height: int = 600,
                   max_steps: int = 50) -> dict:
    """Initialize a new canvas environment session."""
    global env
    env = MarketCanvasEnv(canvas_width=canvas_width, canvas_height=canvas_height,
                          max_steps=max_steps)
    obs, info = env.reset()
    return {
        "status": "initialized",
        "canvas_state": env.get_semantic_state(),
        "prompt": info["prompt"],
    }

@mcp.tool
def get_canvas_state() -> dict:
    """Get the current state of the canvas as a JSON object."""
    return env.get_semantic_state()

@mcp.tool
def execute_action(action_type: str, element_id: str = None,
                   x: int = None, y: int = None, 
                   width: int = None, height: int = None,
                   color: str = None, content: str = None) -> dict:
    """
    Execute a design action on the canvas.
    
    action_type: One of "add_text", "add_shape", "add_image", "move_element", 
                 "change_color", "done"
    """
    # Convert string action to internal action format
    # Call env.step(action)
    # Return new state + reward + done status
    action = _parse_mcp_action(action_type, element_id, x, y, width, height, color, content)
    obs, reward, terminated, truncated, info = env.step(action)
    
    return {
        "canvas_state": env.get_semantic_state(),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "reward_breakdown": info.get("reward_breakdown", {}),
        "action_result": info.get("action_result", {}),
    }

@mcp.tool
def get_current_reward() -> dict:
    """Calculate and return the current reward without taking an action."""
    reward, breakdown = env.reward_calc.calculate(
        env.canvas, env._current_prompt, env._step_count, env.max_steps
    )
    return {"reward": reward, "breakdown": breakdown}

@mcp.tool  
def save_canvas(filepath: str = "canvas_output.png") -> dict:
    """Save the current canvas state as a PNG image."""
    env.renderer.save(env.canvas, filepath)
    return {"status": "saved", "path": filepath}

@mcp.resource("canvas://state")
def canvas_state_resource() -> dict:
    """MCP resource providing the current canvas state."""
    return env.get_semantic_state() if env else {"error": "Environment not initialized"}

if __name__ == "__main__":
    mcp.run()
```

### 6.2 MCP ↔ Gymnasium Bridge

The MCP server wraps the Gymnasium environment but adapts the interface:

| Gymnasium Method | MCP Tool | Adaptation |
|-----------------|----------|------------|
| `env.reset()` | `initialize_env()` | Returns JSON instead of tensors |
| `env.step(action)` | `execute_action(...)` | String-based action params instead of array |
| `render()` | `save_canvas()` | Saves to file instead of returning array |
| `get_semantic_state()` | `get_canvas_state()` | Direct passthrough |
| (custom) | `get_current_reward()` | Read-only reward peek |

---

## 7. Demo Script Design

### 7.1 `demo.py`

```python
"""
Demo script for MarketCanvas-Env.
Shows: initialization, programmatic steps, state inspection, reward calculation, and visual output.
"""

def demo_programmatic():
    """Run a scripted sequence of actions to demonstrate the environment."""
    env = gymnasium.make("MarketCanvas-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    print(f"Target prompt: {info['prompt']}")
    print(f"Initial state: {info['semantic_state']}")
    
    # Execute a sequence of design actions
    actions = [
        {"action_type": 0, "x": 150, "y": 50, "width": 500, "height": 80, ...},   # Add headline
        {"action_type": 1, "x": 250, "y": 300, "width": 200, "height": 60, ...},   # Add CTA button
        {"action_type": 4, "element_idx": 1, "color_idx": 3, ...},                  # Recolor button
        {"action_type": 5},                                                          # DONE
    ]
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, elements={info['element_count']}")
        if terminated or truncated:
            print(f"Episode ended. Final reward: {reward:.3f}")
            print(f"Reward breakdown: {info['reward_breakdown']}")
            break
    
    # Save visual output
    img = env.render()
    PIL.Image.fromarray(img).save("demo_output.png")
    print("Canvas saved to demo_output.png")
    env.close()

def demo_random():
    """Run random actions to show the environment handles arbitrary input."""
    env = gymnasium.make("MarketCanvas-v0")
    obs, info = env.reset(seed=0)
    
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    demo_programmatic()
    demo_random()
```

---

## 8. Dependencies

```toml
[project]
name = "mini-canva"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "gymnasium>=1.0.0",        # RL environment framework
    "numpy>=1.24.0",           # Array operations
    "Pillow>=10.0.0",          # Image rendering
    "fastmcp>=2.0.0",          # MCP server framework
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",             # Testing
    "pygame>=2.5",             # Optional: human-mode rendering
]
```

**Design principle**: Minimal dependencies. No ML frameworks, no browser automation, no GPU libraries. The environment is pure Python + PIL.

---

## 9. Scaling Considerations (PPO with 10,000 Parallel Rollouts)

### 9.1 Bottleneck Analysis

| Bottleneck | Severity | Cause | Mitigation |
|-----------|----------|-------|------------|
| **PIL rendering** | High | `render()` is the slowest operation (~5ms per frame) | Don't render during training. Only render for evaluation/logging |
| **Python GIL** | High | `AsyncVectorEnv` uses threads, limited by GIL | Use `SubprocVectorEnv` (multiprocessing) or Cython/Rust canvas engine |
| **State serialization** | Medium | `to_dict()` creates JSON copies at every step | Use shared memory or numpy-native state representation |
| **Reward computation** | Medium | WCAG luminance calculation per text element | Cache luminance values, only recompute on color changes |
| **VLM inference** | Critical | If using visual observations, VLM forward pass dominates | Decouple VLM inference from env stepping (async architecture) |
| **Memory per env** | Low | Each env holds ~20 elements × ~50 bytes = 1KB | 10K envs = 10MB — trivial |

### 9.2 Redesign for Scale

```
At 10,000 parallel rollouts:

1. ENVIRONMENT VECTORIZATION
   - Replace Python CanvasState with NumPy-backed state (fixed-shape arrays)
   - State = numpy array of shape (10000, MAX_ELEMENTS, NUM_FEATURES)
   - Step function operates on entire batch via vectorized numpy ops
   - No Python loops over individual environments

2. RENDERING DECOUPLING
   - Render only a sampled subset (e.g., 1% of envs per batch)
   - Use a separate rendering worker pool (async, non-blocking)

3. REWARD COMPUTATION
   - Vectorize overlap detection: batch pairwise intersection via numpy broadcasting
   - Pre-compute color luminance lookup table (256³ entries, ~16MB)
   - WCAG check becomes a single array lookup, not per-element computation

4. INFRASTRUCTURE
   - Use Ray or Envpool for distributed environment management
   - gRPC-based env server if env runs on different nodes than the learner
   - Shared memory for obs/action buffers between env workers and learner

5. VLM-SPECIFIC
   - Batch visual observations and run VLM inference on GPU asynchronously
   - Use vLLM or TensorRT-LLM for efficient batched VLM forward passes
   - Pipeline: env.step() → queue → VLM batch inference → queue → env.step()
```

### 9.3 Performance Budget (per step, per env)

| Component | Target | Notes |
|-----------|--------|-------|
| State mutation | < 0.01ms | Dict insertion/update |
| Observation construction | < 0.05ms | numpy array fill |
| Reward computation | < 0.1ms | Cached, vectorized |
| Rendering (when needed) | < 5ms | PIL, only for eval |
| **Total step time** | **< 0.2ms** | **5000 steps/sec/env** |

At 10K envs × 5000 steps/sec = 50M steps/sec aggregate throughput (before VLM bottleneck).

---

## 10. Testing Strategy

### 10.1 Test Matrix

| Layer | Test Type | What to Test | Tool |
|-------|-----------|-------------|------|
| Canvas Engine | Unit | Element CRUD, serialization, spatial queries | pytest |
| Reward Engine | Unit | Each sub-reward independently, edge cases | pytest |
| RL Environment | Integration | `check_env()`, episode lifecycle, space compliance | gymnasium.utils |
| MCP Server | Integration | Tool calls, state consistency, error handling | FastMCP test client |
| Demo Script | Smoke | Runs without errors, produces output file | subprocess |

### 10.2 Key Test Cases

**Canvas Engine**:
- Add/remove element, verify state consistency
- Element outside bounds — accepted (no validation in engine)
- `to_dict()` → JSON serializable
- `get_overlapping_pairs()` correctly detects overlaps

**Reward Engine**:
- Empty canvas → constraint score = 0
- Perfect layout → aesthetics score ≈ 1.0
- Black text on white background → contrast passes WCAG
- White text on white background → contrast fails WCAG
- 100% canvas coverage → coverage score < 1.0 (over-utilized)

**RL Environment**:
- `gymnasium.utils.env_checker.check_env(env)` passes
- Episode terminates on DONE action
- Episode truncates at max_steps
- Observation matches observation_space
- Action space sampling produces valid actions

**MCP Server**:
- `initialize_env()` returns valid state
- `execute_action()` modifies state
- `get_current_reward()` returns consistent reward
- Invalid actions return error, don't crash
