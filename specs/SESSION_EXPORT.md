# Session Export — MarketCanvas-Env Design & Canvas Engine QA

> **Date**: 2026-03-15  
> **Conversation ID**: `1c343340-12c2-45e1-ad73-60ff83693456`  
> **Purpose**: Full context export to resume work in a new conversation.

---

## 1. Project Overview

**MarketCanvas-Env** is a minimalist 2D design canvas simulator for reinforcement learning, located at `/Users/vrn21/Developer/verita/mini-canva/`. The system has four layers:

1. **Canvas Engine** (✅ designed + implemented + QA'd)
2. **Reward Engine** (✅ designed, not yet implemented)
3. **RL Environment** (✅ designed, not yet implemented)
4. **MCP Server** (✅ designed, not yet implemented)

The project requirements live in `specs/REQS.md`.

---

## 2. What Was Done in This Session

### Phase 1: Deep Research
- Researched real-world canvas architecture (Canva's scene graph, Easel design system, Fabric.js, Figma's C++ engine, WebAssembly, constraint-based layout)
- Researched Gymnasium custom environment best practices (Farama Foundation docs)
- Researched MiniWoB++ and GUI-interaction RL environments
- Researched FastMCP server Python implementation
- Researched WCAG 2.1 contrast ratio calculation algorithm
- Researched heuristic reward function design, PPO scaling bottlenecks with VLMs

### Phase 2: Three Architecture Design Docs Written

#### `specs/DESIGN_01_CANVAS_ARCHITECTURE.md`
- Real-world Canva/Figma architecture analysis (scene graph, rendering engines, design systems)
- Gap analysis: REQS.md vs real-world features
- Mini-Canva data model: flat element list (not tree), `CanvasState`, `Element` dataclass, PIL renderer
- Key decision: **flat list over tree hierarchy** — no grouping/nesting needed for RL

#### `specs/DESIGN_02_RL_ENVIRONMENT.md`
- MDP formulation (state, action, transition, reward, episode)
- Dual observation space: JSON semantic (primary) + RGB pixel (optional)
- High-level vs low-level action space analysis → **recommends high-level semantic actions**
- Decomposed reward function with 5 sub-components:
  - `R_constraint` (0.35 weight): required elements present per prompt
  - `R_aesthetics` (0.25): overlap, alignment, margin, spacing
  - `R_accessibility` (0.20): WCAG contrast ratio compliance
  - `R_coverage` (0.10): canvas area utilization (peaks at ~40%)
  - `R_efficiency` (0.10): step penalty
- Reward clamped to [-1.0, 1.0]
- Episode: terminates on DONE action or truncates at max_steps

#### `specs/DESIGN_03_SYSTEM_ARCHITECTURE.md`
- 4-layer architecture with strict downward dependencies
- Full project structure (all files/directories)
- Detailed class/method pseudo-designs for every module
- FastMCP server implementation (tools: `initialize_env`, `get_canvas_state`, `execute_action`, `get_current_reward`, `save_canvas`)
- Demo script design (`demo_programmatic` + `demo_random`)
- Dependencies: `gymnasium>=1.0.0`, `numpy>=1.24.0`, `Pillow>=10.0.0`, `fastmcp>=2.0.0`
- PPO scaling analysis for 10K parallel rollouts (bottleneck table, performance budget)
- Testing strategy (unit + integration + smoke tests)

### Phase 3: Canvas Engine Implementation Spec

#### `specs/CANVAS_ENGINE_SPEC.md`
- Detailed, code-first implementation spec for the `engine/` package
- Complete production-ready code for `types.py`, `canvas.py`, `renderer.py`
- Comprehensive test spec (`test_engine.py` with 30+ test cases)
- Edge case documentation table
- Performance characteristics table

### Phase 4: Implementation (by another agent, with user-approved improvements)

The implementing agent made 9 improvements over my original spec (all accepted by user). These are documented in the **Implementation Changelog** at the top of `CANVAS_ENGINE_SPEC.md`:

| Change | Original | Implementation |
|--------|----------|----------------|
| Package structure | `market_canvas/engine/` (nested) | `engine/` (flat) |
| Z-order storage | `z_index` field + dict + `sorted()` | List position IS z-order + sidecar dict index |
| Element dataclass | `@dataclass` | `@dataclass(slots=True)` |
| `Element.to_dict()` | `dataclasses.asdict()` + enum patch | Manual dict literal (~5x faster) |
| `max_elements` | Fixed at 20 | `Optional[int]`, defaults to `None` (unlimited) |
| Z-order manipulation | Mutate `z_index` field | `reorder_element()`, `bring_to_front()`, `send_to_back()` |
| RL export | Only `to_dict()` | Added `to_numpy()` returning `(features, mask)` arrays |
| Episode reset | Only `clear()` | Added `snapshot()` / `restore()` |
| Font handling | Load font on every call | Font cache `dict[int, Font]` keyed by size |

### Phase 5: QA Review (done by me, this session)

**Result: 59/59 tests pass in 0.32s. Verdict: ship-quality.**

5 minor (non-blocking) notes:
1. `CanvasConfig` docstring says "Immutable" but isn't frozen — cosmetic
2. `update_element` uses `__slots__` introspection — `dataclasses.fields()` would be more robust
3. `_hex_to_rgb` assumes valid hex — add a comment
4. `to_numpy()` features are unnormalized — correct (RL layer's job), add docstring note
5. `get_elements_at()` reverses in-place — purely stylistic

---

## 3. Current File Structure

```
mini-canva/
├── pyproject.toml                    # deps: Pillow, numpy, pytest
├── engine/                           # ✅ IMPLEMENTED + QA'd
│   ├── __init__.py                   # Re-exports: Canvas, Element, ElementType, etc.
│   ├── types.py                      # ElementType(str, Enum), Element(@dataclass slots=True), CanvasConfig
│   ├── canvas.py                     # Canvas class (list + sidecar dict, CRUD, spatial queries, to_numpy, snapshot/restore)
│   └── renderer.py                   # CanvasRenderer (PIL-based, font cache, image placeholder cross pattern)
├── tests/
│   └── test_engine.py                # 59 tests (CRUD, z-order, spatial, serialization, numpy, snapshot, renderer)
├── specs/
│   ├── REQS.md                       # Original requirements
│   ├── DESIGN_01_CANVAS_ARCHITECTURE.md
│   ├── DESIGN_02_RL_ENVIRONMENT.md
│   ├── DESIGN_03_SYSTEM_ARCHITECTURE.md
│   ├── CANVAS_ENGINE_SPEC.md         # Implementation spec (with changelog)
│   └── SESSION_EXPORT.md             # This file
└── main.py                           # Placeholder (unused)
```

---

## 4. Key Design Decisions (Approved by User)

1. **Flat element list, not tree** — No grouping/nesting. RL agent operates on individual elements.
2. **List position = z-order** — No stored `z_index` field. Derived during serialization.
3. **Engine accepts "bad" designs** — Off-canvas elements, overlaps are legal. Reward function penalizes.
4. **`add_element` returns `None` on failure** — Not exceptions. RL step function checks result.
5. **Semantic JSON state is primary observation** — Pixel rendering is optional/secondary.
6. **High-level semantic actions** — `add_text`, `move`, `recolor`, `done` — not pixel-level mouse/keyboard.
7. **Terminal-only reward** — Reward computed only when episode ends (DONE or max_steps). 0 reward mid-episode.
8. **Each layer depends only on the layer below** — Canvas knows nothing about RL. RL knows nothing about MCP.

---

## 5. What's Next (Not Yet Done)

1. **Reward Engine** (`rewards/`) — `ConstraintChecker`, `AestheticsScorer`, `AccessibilityChecker`, `RewardCalculator`
   - Design is in `DESIGN_02_RL_ENVIRONMENT.md` §4 and `DESIGN_03_SYSTEM_ARCHITECTURE.md` §4
   - WCAG contrast algorithm is fully specified
   - Prompt bank with `TargetPrompt` + `PromptConstraint` dataclasses

2. **RL Environment** (`env/`) — `MarketCanvasEnv(gymnasium.Env)`, `spaces.py`, `wrappers.py`
   - Design is in `DESIGN_02_RL_ENVIRONMENT.md` and `DESIGN_03_SYSTEM_ARCHITECTURE.md` §5
   - Observation space: `Dict(elements, element_mask, step_fraction)`
   - Action space: `Dict(action_type, element_idx, x, y, width, height, color_idx, content_idx)`

3. **MCP Server** (`mcp/`) — FastMCP wrapper
   - Design is in `DESIGN_03_SYSTEM_ARCHITECTURE.md` §6

4. **Demo Script** (`demo.py`) — Required deliverable
   - Design is in `DESIGN_03_SYSTEM_ARCHITECTURE.md` §7

5. **WRITEUP.md** — Required deliverable

---

## 6. User's Role & Preferences

- User is the **architect/designer** — writes design docs, does NOT implement
- Another agent implements based on the design docs
- User reviews and accepts/modifies suggestions from the implementing agent
- User values: **scalability, maintainability, simplicity, no over-engineering, first-principles thinking**
- User approved all 9 implementation changelog improvements without pushback
