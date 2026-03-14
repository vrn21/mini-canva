# Design Document 02: RL Environment Design for Canvas Interactions

> How to formulate the 2D canvas design problem as a Markov Decision Process and build a Gymnasium-compliant environment, informed by MiniWoB++, GUI-interaction RL research, and Farama Foundation best practices.

---

## 1. Prior Art: How RL Environments for UI Interaction Are Built

### 1.1 MiniWoB++ (Farama Foundation)

MiniWoB++ is the gold standard for browser-interaction RL. Its architecture reveals key design patterns:

| Aspect | MiniWoB++ Design | Lessons for MarketCanvas |
|--------|-----------------|--------------------------|
| **Observation** | DOM tree (structured) + pixel screenshot (visual) | We need both: JSON state (primary) + RGB array (optional) |
| **Action space** | Click(element), Type(string), Drag(coords) | Maps to both low-level and high-level actions |
| **Reward** | Sparse: +1 success, -1 failure | Too sparse for design. We need dense, heuristic rewards |
| **Episode** | Goal-directed with timeout | Same: "design a banner matching this prompt" |
| **State** | DOM elements with attributes | Our JSON scene graph IS the DOM equivalent |
| **Interface** | Gymnasium API (reset, step, render) | Same |

**Key takeaway**: MiniWoB++ succeeds because the DOM tree gives the agent *structured, semantically rich* observations. Our JSON scene graph serves the exact same role.

### 1.2 GUI Automation Environments (WebRLED, AndroidEnv)

These environments teach us:
- **Pixel-only observations are insufficient** for complex tasks (agents need semantic structure)
- **Large action spaces** require careful factorization (don't enumerate all possible pixel coordinates)
- **Sparse rewards** need augmentation with intermediate signals

### 1.3 PosterCopilot (Layout RL)

Recent research on RL for poster/banner layout:
- Uses RL to learn geometric understanding and aesthetic reasoning
- Reward functions combine rule-based heuristics (alignment, spacing) with learned aesthetic models
- Validates that heuristic reward functions CAN guide layout learning effectively

---

## 2. MDP Formulation

### 2.1 The Markov Decision Process

The canvas design task as an MDP `(S, A, T, R, γ)`:

```
S (State Space):  The canvas state — all elements, their properties, and the target prompt
A (Action Space): Add/modify/delete elements on the canvas
T (Transition):   Deterministic — applying action a to state s yields exactly one next state s'
R (Reward):       Heuristic scalar measuring design quality against the target prompt
γ (Discount):     1.0 (undiscounted, episodic task — reward at terminal step matters most)
```

### 2.2 Why This Is a Good MDP

1. **Markov property holds**: The current canvas state + target prompt fully determines the optimal next action. No history dependency.
2. **Deterministic transitions**: `T(s, a) → s'` is a pure function. No stochasticity in the engine.
3. **Finite episodes**: Episodes terminate after max steps or when the agent signals "done."
4. **State is fully observable**: The JSON scene graph contains complete information.

---

## 3. Observation Space Design

### 3.1 The Dual-Observation Architecture

Following MiniWoB++ and research consensus, we support two observation modes:

```
┌──────────────────────────────────────────────┐
│              Observation Space                │
├────────────────────┬─────────────────────────┤
│   Semantic State   │    Visual State          │
│   (primary)        │    (optional/bonus)      │
├────────────────────┼─────────────────────────┤
│ JSON dict of all   │ RGB numpy array          │
│ elements + prompt  │ (800, 600, 3) uint8      │
├────────────────────┼─────────────────────────┤
│ Used by: text/API  │ Used by: vision models   │
│ agents (LLMs)      │ (VLMs, CNNs)             │
└────────────────────┴─────────────────────────┘
```

### 3.2 Semantic State Structure

```json
{
  "canvas": {
    "width": 800,
    "height": 600,
    "background_color": "#FFFFFF"
  },
  "elements": [
    {
      "id": "element_0",
      "type": "TEXT",
      "x": 200, "y": 100,
      "width": 400, "height": 60,
      "z_index": 1,
      "color": "#000000",
      "text_color": "#FFFFFF",
      "content": "Summer Sale",
      "font_size": 48
    },
    {
      "id": "element_1",
      "type": "SHAPE",
      "x": 250, "y": 300,
      "width": 200, "height": 60,
      "z_index": 2,
      "color": "#FFD700",
      "text_color": "#000000",
      "content": "Shop Now",
      "font_size": 24
    }
  ],
  "target_prompt": "Create a Summer Sale email banner with a headline, a yellow CTA button, and good contrast",
  "step_count": 5,
  "max_steps": 50
}
```

### 3.3 Gymnasium Observation Space Definition

```python
# Semantic observation: variable-length JSON → use gymnasium.spaces.Text or Dict
# The challenge: variable number of elements → can't use fixed-shape Box

# Approach 1: Fixed-capacity element array (recommended for RL training)
observation_space = spaces.Dict({
    "elements": spaces.Box(
        low=0, high=1,
        shape=(MAX_ELEMENTS, NUM_FEATURES),  # e.g., (20, 10)
        dtype=np.float32
    ),  # Normalized, padded
    "element_mask": spaces.MultiBinary(MAX_ELEMENTS),  # Which slots are active
    "prompt_embedding": spaces.Box(
        low=-1, high=1,
        shape=(EMBEDDING_DIM,),
        dtype=np.float32
    ),  # Pre-computed text embedding of target prompt
    "step_fraction": spaces.Box(
        low=0, high=1, shape=(1,), dtype=np.float32
    ),  # step_count / max_steps
})

# Approach 2: Raw JSON (for LLM/MCP agents — not for tensor-based RL)
# Exposed via the info dict or MCP tools, not via observation_space
```

**Design decision**: We support BOTH. The Gymnasium observation space uses Approach 1 (fixed-shape tensors) for RL training compatibility. The raw JSON is available via `env.get_semantic_state()` and MCP tools for LLM agents.

### 3.4 Visual State

```python
# Visual observation: RGB pixel array
visual_observation_space = spaces.Box(
    low=0, high=255,
    shape=(600, 800, 3),  # height × width × channels
    dtype=np.uint8
)
```

Generated on-demand via PIL rendering. **NOT included in default observation to save compute** — accessible via `render(mode="rgb_array")` or a wrapper.

---

## 4. Action Space Design

### 4.1 The Two Action Paradigms

The REQS.md presents two approaches. Here is a principled analysis:

| Dimension | Low-Level (Computer Use) | High-Level (Semantic UI) |
|-----------|------------------------|--------------------------|
| **Actions** | `mouse_move(x,y)`, `click()`, `drag(x1,y1,x2,y2)`, `type(str)` | `add_element(type, props)`, `move_element(id, x, y)`, `change_color(id, hex)` |
| **Action space size** | Continuous (800×600 coords) | Discrete + parameterized |
| **Learning difficulty** | Very hard (credit assignment over pixel coords) | Moderate (direct semantic manipulation) |
| **Real-world alignment** | Mimics human computer use | Mimics API-driven design |
| **For LLM agents** | Requires vision + pixel grounding | Natural fit for tool-calling LLMs |
| **Reward shaping** | Needs dense intermediate rewards | Can use sparser rewards |
| **Implementation complexity** | Needs hit-testing, text input cursor, selection logic | Direct state mutation |

### 4.2 Recommended Primary Action Space: **High-Level Semantic**

**Rationale**: 
1. The assignment's goal is to train agents that "autonomously operate design software." LLM tool-calling IS the production interface.
2. Low-level mouse actions require solving *two* problems simultaneously: (a) spatial reasoning about coordinates and (b) design quality. Separating these makes learning tractable.
3. MCP integration maps 1:1 to high-level actions (each action = one tool call).

### 4.3 High-Level Action Space Definition

```python
# Factored discrete action space
action_space = spaces.Dict({
    "action_type": spaces.Discrete(6),
    # 0: ADD_TEXT
    # 1: ADD_SHAPE  
    # 2: ADD_IMAGE
    # 3: MOVE_ELEMENT
    # 4: CHANGE_COLOR
    # 5: DONE (terminate episode)
    
    "target_element_idx": spaces.Discrete(MAX_ELEMENTS),  # Which element to modify
    "x": spaces.Discrete(CANVAS_WIDTH),       # X position
    "y": spaces.Discrete(CANVAS_HEIGHT),      # Y position  
    "width": spaces.Discrete(CANVAS_WIDTH),   # Element width
    "height": spaces.Discrete(CANVAS_HEIGHT), # Element height
    "color_idx": spaces.Discrete(NUM_COLORS), # Index into color palette
    "content_idx": spaces.Discrete(NUM_CONTENT_TEMPLATES),  # Index into content bank
})
```

**For MCP / LLM agents (string-based):**
```python
# Actions are tool calls with JSON arguments:
# execute_action({"action": "add_element", "type": "TEXT", "content": "Summer Sale", ...})
# execute_action({"action": "move_element", "id": "element_0", "x": 200, "y": 100})
# execute_action({"action": "change_color", "id": "element_0", "color": "#FFD700"})
```

### 4.4 Low-Level Action Space (Optional, for Bonus)

If implemented, use a factored multi-discrete space:

```python
low_level_action_space = spaces.Dict({
    "action_type": spaces.Discrete(4),  # move, click, drag, type
    "x": spaces.Box(low=0, high=799, shape=(1,), dtype=np.int32),
    "y": spaces.Box(low=0, high=599, shape=(1,), dtype=np.int32),
    "drag_end_x": spaces.Box(low=0, high=799, shape=(1,), dtype=np.int32),
    "drag_end_y": spaces.Box(low=0, high=599, shape=(1,), dtype=np.int32),
    "text": spaces.Text(max_length=50),
})
```

This requires a full UI interaction layer (hit-testing, selection, toolbar) — significantly more complex.

### 4.5 Action Space Trade-Offs Summary

```
                    ┌─────────────────┐
                    │   Design Space   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐        ┌──────────▼──────────┐
    │   High-Level       │        │    Low-Level         │
    │   (Semantic)       │        │    (Computer Use)    │
    ├────────────────────┤        ├─────────────────────┤
    │ ✅ Tractable        │        │ ❌ Intractable       │
    │ ✅ MCP-native       │        │ ✅ Human-like        │
    │ ✅ Small action     │        │ ❌ Huge action       │
    │    space            │        │    space             │
    │ ❌ Not human-like   │        │ ❌ Needs vision      │
    │ ✅ Fast to train    │        │ ❌ Slow to train     │
    └────────────────────┘        └─────────────────────┘
```

---

## 5. Reward Function Design (The Core Challenge)

### 5.1 Design Philosophy

The reward function must balance three tensions:

1. **Informative vs. Hackable**: Dense rewards guide learning but create more opportunities for reward hacking
2. **Rule-based vs. Learned**: Heuristic rules are interpretable but may not capture true design quality
3. **Task-specific vs. General**: Prompt-specific rewards are more useful but harder to design

**Our approach**: Weighted sum of decomposed heuristic sub-rewards, each addressing a specific design quality dimension. Total reward clamped to `[-1.0, 1.0]`.

### 5.2 Reward Decomposition

```
R_total = clamp(
    w₁ · R_constraint +      # Are required elements present?
    w₂ · R_aesthetics +      # Layout quality (alignment, overlap, margins)
    w₃ · R_accessibility +   # WCAG contrast compliance
    w₄ · R_coverage +        # Canvas utilization
    w₅ · R_efficiency,       # Step efficiency penalty
    -1.0, 1.0
)

Default weights: w₁=0.35, w₂=0.25, w₃=0.20, w₄=0.10, w₅=0.10
```

### 5.3 Sub-Reward: Constraint Satisfaction (`R_constraint`)

Given a target prompt like "Create a Summer Sale banner with a headline, a yellow CTA button, and good contrast", we parse it into required constraints:

```
Constraints for example prompt:
  - has_text_element(content_matches="headline|title|summer sale")  → bool
  - has_shape_element(role="CTA|button")                           → bool
  - shape_has_color(role="CTA", color_close_to="#FFD700")          → bool (yellow)
```

```
R_constraint = (num_satisfied_constraints / total_constraints)
# Range: [0.0, 1.0]
```

**Prompt parsing**: For initial implementation, use a simple keyword-matching system against a predefined constraint schema. The system defines what constraints to look for based on prompt templates. No NLP needed.

### 5.4 Sub-Reward: Aesthetics (`R_aesthetics`)

Four components, equally weighted:

#### 5.4.1 Overlap Penalty
```
overlap_score = 1.0 - (total_overlap_area / total_element_area)
```
For each pair of elements, compute intersection rectangle area. Penalize overlaps proportional to their area.

#### 5.4.2 Alignment Score
```
alignment_score = max(
    horizontal_center_alignment,   # How many elements share a center X
    vertical_center_alignment,     # How many elements share a center Y  
    left_edge_alignment,           # How many elements share a left edge X
)
```
Check if elements are aligned to common axes. Use tolerance window (±5 pixels).

#### 5.4.3 Margin Compliance
```
margin_score = fraction of elements with >= MIN_MARGIN px from canvas edges
```
Elements too close to edges look unprofessional. `MIN_MARGIN = 20px`.

#### 5.4.4 Spacing Regularity
```
spacing_score = 1.0 - normalized_stddev(vertical_gaps_between_elements)
```
Even spacing between stacked elements is aesthetically pleasing.

```
R_aesthetics = 0.25 * overlap + 0.25 * alignment + 0.25 * margin + 0.25 * spacing
# Range: [0.0, 1.0]
```

### 5.5 Sub-Reward: Accessibility (`R_accessibility`)

WCAG 2.1 AA requires minimum contrast ratio of **4.5:1** for normal text and **3:1** for large text (≥18pt or ≥14pt bold).

```
Contrast Ratio = (L1 + 0.05) / (L2 + 0.05)

where L = 0.2126 * R_linear + 0.7152 * G_linear + 0.0722 * B_linear
      R_linear = (R_sRGB / 12.92)          if R_sRGB ≤ 0.03928
                 ((R_sRGB + 0.055) / 1.055)^2.4  otherwise
```

For each text element:
1. Determine background color (what's behind it — canvas bg or overlapping shape)
2. Calculate contrast ratio between text_color and background
3. Compare against WCAG threshold

```
R_accessibility = fraction of text elements meeting WCAG AA contrast
# Range: [0.0, 1.0]
```

### 5.6 Sub-Reward: Canvas Coverage (`R_coverage`)

Empty canvases score 0. Canvases where elements use 20-80% of the area score highest.

```
coverage_ratio = total_element_area / canvas_area
R_coverage = 1.0 - 2 * |coverage_ratio - 0.5|  # Peak at 50% coverage
# Alternative: gaussian centered at 0.4
# Range: [0.0, 1.0]
```

### 5.7 Sub-Reward: Step Efficiency (`R_efficiency`)

Encourage the agent to solve tasks in fewer steps.

```
R_efficiency = max(0, 1.0 - (steps_taken / max_steps))
# Range: [0.0, 1.0]
```

### 5.8 Reward Hacking Risks and Mitigations

| Hack | Description | Mitigation |
|------|-------------|------------|
| **Empty canvas** | Agent does nothing → no overlap, perfect spacing | R_constraint = 0 dominates (highest weight) |
| **Single huge element** | One element covering 50% → good coverage | R_aesthetics penalizes (no alignment variety) |
| **Off-canvas elements** | Agent places elements outside viewport | Don't count off-canvas elements for R_constraint |
| **Invisible text** | Text same color as background → "present" but unreadable | R_accessibility catches this (contrast = 1:1) |
| **Micro-elements** | Tiny elements pass all checks | Add minimum size threshold in constraint parser |
| **Color-only optimization** | Perfect contrast but terrible layout | Multi-objective weighting prevents over-optimization on one axis |

---

## 6. Episode Structure

### 6.1 Episode Lifecycle

```
┌─────────┐      ┌──────┐      ┌──────────┐
│  reset() │─────▶│step()│─────▶│terminated│
│          │      │      │──┐   │    or     │
│ • sample │      │ • act│  │   │ truncated │
│   prompt │      │ • obs│  │   │           │
│ • clear  │      │ • rew│  └──▶│ • final   │
│   canvas │      │ • info│     │   reward  │
└─────────┘      └──────┘      └──────────┘
                  ▲     │
                  └─────┘
              repeat up to max_steps
```

### 6.2 Termination Conditions

- **`terminated = True`**: Agent takes `DONE` action → episode ends normally
- **`truncated = True`**: `step_count >= max_steps` → episode ends due to time limit
- **Reward timing**: Full reward computed at episode end (terminal reward). Optional: intermediate reward signal at each step (for dense RL).

### 6.3 Target Prompts

Prompts are sampled from a bank during `reset()`:

```python
PROMPT_BANK = [
    {
        "text": "Create a Summer Sale email banner with a headline, a yellow CTA button, and good contrast",
        "constraints": [
            {"type": "has_element", "element_type": "TEXT", "role": "headline"},
            {"type": "has_element", "element_type": "SHAPE", "role": "cta_button"},
            {"type": "element_color", "role": "cta_button", "color": "#FFD700", "tolerance": 60},
            {"type": "contrast", "min_ratio": 4.5},
        ]
    },
    # ... more prompts
]
```

---

## 7. Gymnasium Environment API

### 7.1 Class Structure

```python
class MarketCanvasEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, canvas_width=800, canvas_height=600, 
                 max_steps=50, max_elements=20, action_mode="high_level"):
        super().__init__()
        self.observation_space = ...  # As defined in Section 3.3
        self.action_space = ...       # As defined in Section 4.3
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 1. Clear canvas
        # 2. Sample target prompt
        # 3. Return initial observation + info
        return observation, info
    
    def step(self, action):
        # 1. Parse action → mutation on canvas state
        # 2. Apply mutation (add/move/recolor element)
        # 3. Increment step counter
        # 4. Compute reward (terminal or intermediate)
        # 5. Check termination/truncation
        # 6. Return (obs, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info
    
    def render(self):
        # PIL-based rendering to RGB array or Pygame display
        pass
    
    def get_semantic_state(self):
        # Returns full JSON state dict (for MCP/LLM agents)
        pass
```

### 7.2 Registration

```python
gymnasium.register(
    id="MarketCanvas-v0",
    entry_point="market_canvas.env:MarketCanvasEnv",
    max_episode_steps=50,
)
```

### 7.3 Vectorized Environment Support

For PPO with parallel rollouts:

```python
envs = gymnasium.vector.AsyncVectorEnv([
    lambda: gymnasium.make("MarketCanvas-v0")
    for _ in range(num_parallel)
])
```

Since our environment is pure Python with no external dependencies (no browser, no GPU), it vectorizes trivially.

---

## 8. Lessons from OSS RL Environments

### 8.1 What Makes a Good RL Environment (Patterns)

From studying Gymnasium built-in environments, MiniWoB++, and research:

| Pattern | Description | Application |
|---------|-------------|-------------|
| **Deterministic core** | Same state + action → same next state | Our canvas engine is pure function |
| **Seeded randomness** | Only in reset(), controlled by seed | Prompt sampling uses np_random |
| **Cheap step()** | < 1ms per step enables 10k+ parallel envs | No rendering in step() by default |
| **Rich info dict** | Debug data in info, not observation | Sub-rewards, element count, etc. |
| **check_env compliance** | Pass gymnasium.utils.env_checker.check_env | Must validate spaces, API contract |
| **Wrapper-friendly** | Core env is minimal; wrappers add features | TimeLimit, PixelObservation as wrappers |
