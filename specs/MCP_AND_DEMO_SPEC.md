# Phase 3 - MCP Server & Demo Script Implementation Specification

> **Audience**: Implementing agent. This document contains the complete specification for building the Phase 3 deliverables of MarketCanvas-Env: the FastMCP server and the `demo.py` script, plus the required dependency and test updates.
>
> **Depends on**: `engine/`, `rewards/`, and `env/` packages (implemented).
>
> **North star**: `specs/REQS.md` plus the already-approved architectural decisions in `specs/DESIGN_01_CANVAS_ARCHITECTURE.md`, `specs/DESIGN_02_RL_ENVIRONMENT.md`, `specs/DESIGN_03_SYSTEM_ARCHITECTURE.md`, `specs/CANVAS_ENGINE_SPEC.md`, and `specs/REWARD_AND_ENV_SPEC.md`.

---

## 0. Requirements Traceability

Every Phase 3 deliverable maps directly to REQS.md and the approved prior specs:

| Requirement / Prior Decision | Where It Is Implemented |
|---|---|
| "simultaneously act as a Model Context Protocol (MCP) server" | `server.py` |
| "Expose tools like get_canvas_state, execute_action, and get_current_reward" | `server.py` FastMCP tools |
| "Demo Script (demo.py)" | `demo.py` |
| "accepts a mock target prompt, takes a few programmatic (or random) steps, and prints the resulting state and reward" | `demo_programmatic()` + `demo_random()` |
| Flat package structure (`engine/`, `rewards/`, `env/`) | Phase 3 stays flat: top-level `server.py` and `demo.py` |
| Gymnasium-compliant environment remains the single source of episode logic | `server.py` delegates to `MarketCanvasEnv` instead of re-implementing step/reward rules |
| High-level semantic action space with 7 discrete action types | `server.py` exposes string-based semantic actions and translates them to the env's discrete action dict |
| Preserve Phase 2 fixes (`prompt_id`, dense reward delta fix, min element size 20px) | `server.py` surfaces `prompt_id`; demo/tests assert current env behavior rather than bypassing it |

---

## 1. Scope & Constraints

**What this spec covers**:

- `server.py` - FastMCP server
- `demo.py` - deterministic scripted demo plus random smoke demo
- `tests/test_mcp.py` - MCP integration tests
- `tests/test_demo.py` - demo smoke/unit tests
- `pyproject.toml` - dependency and test-runner updates

**What this spec does NOT cover**:

- Any new reward logic
- Any new Gymnasium space definitions
- Any new canvas primitives
- Any HTTP deployment, auth, or production hosting

**Hard constraints**:

1. Reuse the existing `MarketCanvasEnv` as the source of truth for reset, step, reward, and semantic state.
2. Do not introduce a local package named `mcp/`.
3. Keep the Phase 3 surface area simple: one server module, one demo module, focused tests.
4. Stay compatible with `fastmcp>=2.0.0`; do not require niche or v3-only features.
5. Do not over-engineer session management. One in-process environment per server process is enough.

---

## 2. Critical Design Decision: No Local `mcp/` Package

The preliminary architecture doc sketched a `mcp/server.py` layout. Do **not** implement that in this repository.

### 2.1 Why

FastMCP depends on the official Python package named `mcp`. If this repo adds a top-level local package named `mcp/`, Python import resolution can shadow the installed dependency when running from the project root. That is a fragile import collision and can break FastMCP at runtime.

### 2.2 Required Layout

Use top-level modules instead:

```text
mini-canva/
├── engine/
├── rewards/
├── env/
├── server.py               # NEW - FastMCP server
├── demo.py                 # NEW - demo script
├── tests/
│   ├── test_engine.py
│   ├── test_rewards.py
│   ├── test_env.py
│   ├── test_mcp.py         # NEW
│   └── test_demo.py        # NEW
└── pyproject.toml
```

If Phase 4 later needs a packaged server module, use a non-conflicting name like `marketcanvas_server/`, never `mcp/`.

---

## 3. External Patterns Incorporated

This spec intentionally follows the same patterns used by the upstream libraries and comparable RL projects:

### 3.1 FastMCP Patterns

- Create one `FastMCP(...)` instance in module scope.
- Register tools with `@mcp.tool`.
- Register a read-only JSON resource with `@mcp.resource(...)`.
- Start the server via `if __name__ == "__main__": mcp.run()`.
- Test the server in-process using the FastMCP client instead of standing up an external transport.

### 3.2 Gymnasium Demo Patterns

Following Minigrid manual-control/testing patterns and MiniWoB++ example usage:

- Seed reset explicitly for deterministic scripted runs.
- Use a small step loop that prints reward/termination state after each action.
- Always close the env in `finally`.
- Keep random demos as smoke tests, not benchmarks.
- Treat demo functions as callable Python functions first, CLI entrypoints second.

### 3.3 MCP Tool / Resource Best Practices Applied Here

- Read-only data retrieval stays in `get_canvas_state`, `get_current_reward`, and `canvas://state`.
- Mutations flow only through `initialize_env`, `execute_action`, and `save_canvas`.
- Server-side validation catches malformed tool calls early.
- Action failures that are valid but unsuccessful (e.g. bad `element_id`) return structured failure results instead of crashing the server.
- Resource output is stable JSON so clients can consume it repeatedly.

---

## 4. Dependency Changes

### 4.1 `pyproject.toml`

Update dependencies as follows:

```toml
[project]
dependencies = [
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "gymnasium>=1.0.0",
    "fastmcp>=2.0.0",        # NEW
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.23.0", # NEW - recommended by FastMCP testing docs
]

[tool.pytest.ini_options]
asyncio_mode = "auto"         # NEW
```

### 4.2 Packaging

Do **not** add a `mcp` package to `[tool.hatch.build.targets.wheel].packages`.

Keep:

```toml
[tool.hatch.build.targets.wheel]
packages = ["engine", "rewards", "env"]
```

Rationale:

- `server.py` and `demo.py` are repo-level entry modules, not packages.
- The assignment deliverable is the repository, not a published wheel.
- Avoid unnecessary packaging churn in Phase 3.

---

## 5. `server.py` - FastMCP Server

### 5.1 Responsibilities

`server.py` is a thin adapter layer. It must:

1. Hold the current `MarketCanvasEnv` session.
2. Translate MCP-friendly semantic actions into the env's discrete action dict.
3. Expose current semantic state and reward to LLM clients.
4. Save the rendered canvas to disk.
5. Expose the current canvas state as a read-only MCP resource.

It must **not**:

- Duplicate reward logic
- Maintain a second canvas state
- Re-implement episode transitions outside the env
- Add networking or auth abstractions

### 5.2 Public API

The server exposes exactly these tools and resources:

#### Tools

- `initialize_env`
- `get_canvas_state`
- `execute_action`
- `get_current_reward`
- `save_canvas`

#### Resource

- `canvas://state`

---

### 5.3 Action Semantics

The MCP-facing action names are:

| MCP action | Existing env action |
|---|---|
| `add_text` | `ACTION_ADD_TEXT` |
| `add_shape` | `ACTION_ADD_SHAPE` |
| `add_image` | `ACTION_ADD_IMAGE` |
| `move` | `ACTION_MOVE` |
| `recolor` | `ACTION_RECOLOR` |
| `remove` | `ACTION_REMOVE` |
| `done` | `ACTION_DONE` |

These names must be exposed as strings because they are clearer for tool-calling LLMs than integer action codes.

---

### 5.4 Important Bridge Design

The current env action space is intentionally RL-friendly, not LLM-friendly:

- colors are palette indices
- content is a template index
- elements are referenced by position (`element_idx`), not stable IDs

The MCP server must bridge that mismatch without changing the env API.

#### 5.4.1 Element Lookup

For `move`, `recolor`, and `remove`, the server accepts `element_id: str` and resolves it to the current z-order index via `env._canvas.get_all_elements()`.

If the ID does not exist:

- do **not** crash
- execute no mutation
- return a structured failure result with `success: false`

#### 5.4.2 Arbitrary Content

The env only supports `content_idx`, but MCP clients should be able to pass arbitrary `content: str`.

Required bridge behavior:

1. Choose the nearest env-compatible action by supplying a valid `content_idx`.
2. Call `env.step(action)` normally.
3. If the action created or targeted an element and the caller provided explicit `content`, patch the canvas element's `content` afterward via `env._canvas.update_element(...)`.

This keeps the env unchanged while making the MCP surface expressive enough for LLM use.

#### 5.4.3 Arbitrary Hex Colors

The env only supports palette indices, but MCP clients should be able to pass arbitrary `#RRGGBB` colors.

Required bridge behavior:

1. Map the requested color to the nearest palette entry for the `env.step(...)` call.
2. After the step, patch the exact element color on the canvas if the caller supplied a color not already equal to the palette value.

Special case:

- For `TEXT` elements, treat recoloring as updating `text_color`, not the unused `color` field.
- For `SHAPE` elements with text content, keep `text_color` contrasting against the final fill color.
- For `IMAGE` elements, recolor updates `color`.

This is a deliberate MCP-layer usability improvement over the raw RL action space.

#### 5.4.4 Minimum Element Size

Do not bypass the environment's anti-reward-hacking fix.

For add actions:

- if `width` or `height` is omitted, use sensible defaults
- if provided below `20`, clamp to `20`

The server must preserve the env's minimum substantive element size behavior.

---

### 5.5 `server.py` Structure

Use this module structure:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from engine.types import ElementType
from env.market_canvas_env import MarketCanvasEnv
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
)
from rewards.accessibility import relative_luminance
```

#### 5.5.1 Server State Container

Use a tiny dataclass instead of loose globals:

```python
@dataclass
class ServerSession:
    env: MarketCanvasEnv | None = None
    seed: int | None = None


SESSION = ServerSession()
mcp = FastMCP("MarketCanvas-MCP")
```

This is enough. Do not add a session registry, locks, or multi-tenant abstractions.

---

### 5.6 Helper Functions

Implement the following private helpers.

#### `_require_env() -> MarketCanvasEnv`

- Returns the current env if initialized.
- Raises `RuntimeError("Environment not initialized. Call initialize_env first.")` otherwise.

Let FastMCP convert the exception to an MCP tool error.

#### `_semantic_state(env: MarketCanvasEnv) -> dict[str, Any]`

- Returns `env.get_semantic_state()`
- Injects `prompt_id`
- Injects `initialized: True`

Reason: `env.get_semantic_state()` currently does not include `prompt_id`, but Phase 2 explicitly added `prompt_id` to preserve the Markov property. The server should expose that identifier too.

#### `_element_id_to_idx(env, element_id) -> int | None`

- Resolve stable element IDs to the current positional index expected by the env action dict.

#### `_nearest_color_index(hex_color: str) -> int`

- Compute Euclidean RGB distance against `COLOR_PALETTE`.
- Return the nearest palette index.

#### `_content_index(content: str | None) -> int`

- If `content` exactly matches an entry in `CONTENT_TEMPLATES`, return its index.
- Else return a reasonable fallback index based on action type:
  - text -> `"Summer Sale"` or `"Product Name"`
  - shape -> `"Shop Now"`
  - image -> `"Product Name"`

The exact fallback should be deterministic.

#### `_shape_text_color(fill_color: str) -> str`

- Use `relative_luminance(fill_color)` to keep shape label text legible.
- Mirror the env's threshold logic (`> 0.179 -> black else white`).

#### `_default_box(action_type: str) -> tuple[int, int, int, int]`

Provide minimal default geometry so an LLM can call add-actions without filling every field:

- `add_text` -> `(120, 80, 520, 80)`
- `add_shape` -> `(300, 320, 200, 60)`
- `add_image` -> `(80, 140, 260, 220)`

These defaults are intentionally banner-like and deterministic.

---

### 5.7 Tool Specifications

#### 5.7.1 `initialize_env`

```python
@mcp.tool
def initialize_env(
    canvas_width: int = 800,
    canvas_height: int = 600,
    max_steps: int = 50,
    max_elements: int = 20,
    seed: int | None = None,
) -> dict[str, Any]:
    ...
```

**Behavior**:

1. Create a fresh `MarketCanvasEnv`.
2. Call `reset(seed=seed)`.
3. Store the env in `SESSION`.
4. Return a structured result:

```python
{
    "status": "initialized",
    "prompt": info["prompt"],
    "prompt_id": int(obs["prompt_id"]),
    "canvas_state": _semantic_state(env),
    "element_count": info["element_count"],
    "step_count": info["step_count"],
}
```

**Validation**:

- `canvas_width > 0`
- `canvas_height > 0`
- `max_steps > 0`
- `max_elements > 0`

Invalid values raise `ValueError`.

**Notes**:

- Re-initialization replaces the prior in-memory env.
- If a prior env exists, call `SESSION.env.close()` before replacing it.
- No separate `close` tool is needed in Phase 3.

---

#### 5.7.2 `get_canvas_state`

```python
@mcp.tool
def get_canvas_state() -> dict[str, Any]:
    ...
```

**Behavior**:

- Return `_semantic_state(_require_env())`.

This is a pure read tool.

---

#### 5.7.3 `execute_action`

```python
ActionName = Literal[
    "add_text",
    "add_shape",
    "add_image",
    "move",
    "recolor",
    "remove",
    "done",
]


@mcp.tool
def execute_action(
    action_type: ActionName,
    element_id: str | None = None,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    color: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    ...
```

**Core rule**: `execute_action` always delegates to `env.step(...)`; it never mutates episode counters or termination flags on its own.

##### Supported Inputs

- `add_text`: optional `x`, `y`, `width`, `height`, `color`, `content`
- `add_shape`: optional `x`, `y`, `width`, `height`, `color`, `content`
- `add_image`: optional `x`, `y`, `width`, `height`, `color`, `content`
- `move`: required `element_id`, `x`, `y`
- `recolor`: required `element_id`, `color`
- `remove`: required `element_id`
- `done`: no other arguments used

##### Validation Rules

Malformed calls raise `ValueError` before stepping:

- `move` without `element_id`, `x`, or `y`
- `recolor` without `element_id` or `color`
- `remove` without `element_id`
- malformed color string not matching `#RRGGBB`

##### Translation Rules

Build a complete env action dict with all fields present:

```python
{
    "action_type": ...,
    "element_idx": ...,
    "x": ...,
    "y": ...,
    "width": ...,
    "height": ...,
    "color_idx": ...,
    "content_idx": ...,
}
```

Use defaults for irrelevant fields so the dict always satisfies the env action schema.

##### Post-Step Patching

After `obs, step_reward, terminated, truncated, info = env.step(action)`:

1. Inspect `info["action_result"]`.
2. If the action successfully created or modified an element and the caller supplied richer MCP data, patch the actual canvas element:
   - exact `content`
   - exact `text_color` for text recolor
   - exact `color` for shape/image recolor
   - exact `color` plus contrasting `text_color` for labeled shapes
3. If `terminated` or `truncated`, recompute reward with `env.compute_reward()` so the returned final reward reflects the post-patched state.

For non-terminal steps:

- `reward` remains the env's step reward (`0.0` today)
- additionally return `current_reward` and `current_reward_breakdown` computed from `env.compute_reward()`

This gives MCP clients immediate progress visibility without changing env semantics.

##### Return Shape

Return:

```python
{
    "canvas_state": _semantic_state(env),
    "reward": float(...),                   # step reward, or recomputed final reward on terminal steps
    "current_reward": float(...),           # reward of the current state after patching
    "terminated": terminated,
    "truncated": truncated,
    "reward_breakdown": {...},              # final breakdown if terminal, else {}
    "current_reward_breakdown": {...},      # always populated
    "action_result": info.get("action_result", {}),
}
```

##### Failure Semantics

If the action is well-formed but references a missing `element_id`:

- do not raise
- return `action_result.success == False`
- still return the unchanged state and current reward

That is an environment-level unsuccessful action, not a transport-level error.

---

#### 5.7.4 `get_current_reward`

```python
@mcp.tool
def get_current_reward() -> dict[str, Any]:
    ...
```

**Behavior**:

- Call `env.compute_reward()`
- Return:

```python
{
    "reward": reward,
    "breakdown": breakdown,
    "step_count": env._step_count,
    "max_steps": env.max_steps,
    "prompt": env._current_prompt.text,
    "prompt_id": env._current_prompt_id,
}
```

This tool is read-only and does not advance the episode.

---

#### 5.7.5 `save_canvas`

```python
@mcp.tool
def save_canvas(filepath: str = "canvas_output.png") -> dict[str, Any]:
    ...
```

**Behavior**:

1. Resolve `filepath` with `Path(filepath).expanduser()`.
2. Create parent directories if needed.
3. Save via `env._renderer.save(env._canvas, path)`.
4. Return:

```python
{
    "status": "saved",
    "path": str(path.resolve()),
    "element_count": env._canvas.element_count,
}
```

**Notes**:

- Overwriting an existing file is acceptable in Phase 3.
- Use the env's renderer directly. Do not round-trip through `render()` unless necessary.

---

### 5.8 Resource Specification

#### `canvas://state`

```python
@mcp.resource("canvas://state")
def canvas_state_resource() -> str:
    ...
```

**Return type**: JSON string, not dict.

Why:

- FastMCP resources are read-oriented and simplest when returned as explicit JSON text.
- This keeps resource behavior stable across FastMCP 2.x versions.

**Behavior**:

- If no env has been initialized, return:

```json
{
  "initialized": false,
  "message": "Call initialize_env first."
}
```

- Else return `json.dumps(_semantic_state(env), indent=2, sort_keys=True)`.

This is the stable, read-only snapshot surface for MCP resource readers.

---

### 5.9 `server.py` Entrypoint

Finish the file with:

```python
if __name__ == "__main__":
    mcp.run()
```

Use stdio transport by default. Do not hardcode HTTP transport in Phase 3.

That matches the intended local MCP-client use case and FastMCP's recommended baseline pattern.

---

### 5.10 Full Behavioral Notes

1. The server is single-session and in-memory per process. That is correct for the assignment.
2. `initialize_env` is the only session reset entrypoint.
3. `get_canvas_state` and `canvas://state` must both expose `prompt_id`.
4. The server must preserve the env's actual step counter, termination, and truncation rules.
5. The server may patch content/colors after `env.step`, but it must not patch geometry, reward weights, or prompt selection behavior.
6. The server must not use private env internals more than necessary. Using `_canvas`, `_renderer`, `_current_prompt_id`, and `_step_count` is acceptable here because this is a first-party adapter layer inside the same repo.

---

## 6. `demo.py` - Scripted and Random Demonstrations

### 6.1 Responsibilities

`demo.py` exists to satisfy the assignment deliverable and to provide a human-readable smoke test of the environment.

It must contain:

- `demo_programmatic()`
- `demo_random()`
- `main()`

Both functions must be importable and callable from tests.

---

### 6.2 Design Goals

1. Demonstrate the env through the actual Gymnasium API.
2. Be deterministic when seeded.
3. Print useful output without being verbose.
4. Save at least one rendered PNG.
5. Never leave env instances unclosed.

---

### 6.3 Module Layout

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from PIL import Image

from env import register_envs
from env.market_canvas_env import MarketCanvasEnv
from env.spaces import (
    ACTION_ADD_IMAGE,
    ACTION_ADD_SHAPE,
    ACTION_ADD_TEXT,
    ACTION_DONE,
    ACTION_MOVE,
    ACTION_RECOLOR,
    COLOR_PALETTE,
    CONTENT_TEMPLATES,
)
```

`register_envs()` should be called inside `main()` and inside demo helpers before `gym.make(...)` so the script works when run directly.

---

### 6.4 Helper Functions

Implement these private helpers for readability.

#### `_color_idx(hex_color: str) -> int`

- Exact lookup into `COLOR_PALETTE`
- Raise `ValueError` if missing

Use exact palette colors in the demo. Do not add approximate color logic here.

#### `_content_idx(text: str) -> int`

- Exact lookup into `CONTENT_TEMPLATES`
- Raise `ValueError` if missing

#### `_base_action() -> dict[str, int]`

Return a fully-populated default env action dict:

```python
{
    "action_type": ACTION_ADD_TEXT,
    "element_idx": 0,
    "x": 100,
    "y": 100,
    "width": 200,
    "height": 60,
    "color_idx": 0,
    "content_idx": 0,
}
```

Each scripted action mutates a copy of this dict.

#### `_programmatic_actions(prompt: str) -> list[tuple[str, dict[str, int]]]`

Map the known prompt texts from `PromptBank` to a short deterministic action script.

This function is the key to keeping `demo_programmatic()` robust even though prompt sampling happens inside `env.reset()`.

Do not add NLP. Use exact or keyword-based matching on the current hardcoded prompt texts.

Required behavior:

- For the summer-sale prompt, add a headline, add a yellow CTA button, then `done`
- For product-launch, add image + title + date, then `done`
- For newsletter-signup, add heading + input-like shape + subscribe button, then `done`
- For holiday card, add festive image + greeting + decorative shape, then `done`
- For flash-sale, add headline + discount text + red CTA button, then `done`
- Fallback: add one text element, one shape, then `done`

---

### 6.5 `demo_programmatic`

```python
def demo_programmatic(
    seed: int = 42,
    output_path: str | Path = "outputs/demo_programmatic.png",
) -> dict[str, Any]:
    ...
```

**Required flow**:

1. Call `register_envs()`.
2. Create env via `gym.make("MarketCanvas-v0", render_mode="rgb_array")`.
3. Reset with `seed=seed`.
4. Print:
   - prompt text
   - prompt id
   - initial element count
5. Build the scripted actions from the sampled prompt.
6. Execute each action in order.
7. After each step, print one concise line:

```text
Step 1 | add headline | reward=0.000 | elements=1 | terminated=False | truncated=False
```

8. Stop when `terminated or truncated`.
9. Save the final rendered image to `output_path`.
10. Print final reward and reward breakdown.
11. Return a summary dict.

**Required return shape**:

```python
{
    "prompt": info["prompt"],
    "prompt_id": int(obs["prompt_id"]),
    "steps_executed": int,
    "terminated": bool,
    "truncated": bool,
    "final_reward": float,
    "reward_breakdown": dict[str, Any],
    "output_path": str(resolved_output_path),
    "semantic_state": env.unwrapped.get_semantic_state(),
}
```

**Resource management**:

Wrap env usage in `try/finally: env.close()`.

---

### 6.6 `demo_random`

```python
def demo_random(
    seed: int = 0,
    total_steps: int = 25,
) -> dict[str, Any]:
    ...
```

This is a smoke demo, not a benchmark.

**Required flow**:

1. Call `register_envs()`.
2. Create env via `gym.make("MarketCanvas-v0")`.
3. Reset with `seed=seed`.
4. Sample random actions from `env.action_space`.
5. Track:
   - total steps taken
   - episode count
   - how many episodes terminated/truncated
6. If an episode ends before `total_steps`, reset and continue.
7. Print short progress lines, not full states.
8. Return a summary dict.

**Required return shape**:

```python
{
    "total_steps": int,
    "episodes_started": int,
    "episodes_finished": int,
    "last_prompt": str,
    "last_element_count": int,
}
```

Again, always close the env in `finally`.

---

### 6.7 `main()`

```python
def main() -> None:
    register_envs()
    print("=== Programmatic Demo ===")
    demo_programmatic()
    print()
    print("=== Random Demo ===")
    demo_random()


if __name__ == "__main__":
    main()
```

Keep the CLI entrypoint minimal. No argparse is required in Phase 3.

---

### 6.8 Demo Design Notes

1. Use `gym.make(...)` instead of direct `MarketCanvasEnv(...)` construction so the demo exercises Gymnasium registration too.
2. The programmatic demo should be deterministic for a fixed seed.
3. The scripted demo should be prompt-aware, not hardcoded to one prompt.
4. The random demo exists to prove robustness under arbitrary valid actions.
5. The programmatic demo should save an image; the random demo does not need to.

---

## 7. Tests

Phase 3 needs two new test files.

---

### 7.1 `tests/test_mcp.py`

This file should be async and use the in-process FastMCP client.

#### 7.1.1 Fixture

Create a fixture like:

```python
import pytest
from fastmcp import Client

from server import SESSION, mcp


@pytest.fixture(autouse=True)
def reset_server_session():
    if SESSION.env is not None:
        SESSION.env.close()
    SESSION.env = None
    SESSION.seed = None
    yield
    if SESSION.env is not None:
        SESSION.env.close()
    SESSION.env = None
    SESSION.seed = None


@pytest.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client
```

No external transport or subprocess is needed.

#### 7.1.2 Test Cases

Implement these tests.

##### `TestServerInitialization`

- `test_list_tools_contains_expected_names`
- `test_initialize_env_returns_prompt_and_canvas_state`
- `test_initialize_env_seed_is_deterministic`

##### `TestStateAccess`

- `test_get_canvas_state_before_init_errors`
- `test_get_canvas_state_after_init_contains_prompt_id`
- `test_canvas_state_resource_before_init_returns_initialized_false`
- `test_canvas_state_resource_after_init_returns_json_snapshot`

##### `TestExecuteAction`

- `test_execute_action_add_text_updates_canvas`
- `test_execute_action_add_shape_updates_canvas`
- `test_execute_action_move_existing_element`
- `test_execute_action_remove_existing_element`
- `test_execute_action_done_terminates_episode`
- `test_execute_action_missing_required_fields_errors`
- `test_execute_action_unknown_element_returns_structured_failure`

##### `TestBridgeBehavior`

- `test_execute_action_accepts_custom_content_not_in_templates`
- `test_execute_action_accepts_custom_hex_color_not_in_palette`
- `test_recolor_text_element_updates_text_color`
- `test_recolor_shape_keeps_contrasting_text_color`
- `test_add_action_clamps_dimensions_to_minimum_size`

##### `TestRewardAndSave`

- `test_get_current_reward_returns_scalar_and_breakdown`
- `test_save_canvas_writes_png`
- `test_save_canvas_returns_absolute_path`

#### 7.1.3 Assertions to Include

- returned canvas state contains `prompt_id`
- resource output parses as valid JSON
- `save_canvas` output file exists and has `.png` suffix
- terminal `done` step returns non-empty reward breakdown
- custom content patching is reflected in the semantic state
- custom color patching is reflected in the semantic state

---

### 7.2 `tests/test_demo.py`

This file tests the Python-callable demo functions directly.

#### 7.2.1 Test Cases

##### `TestProgrammaticDemo`

- `test_demo_programmatic_returns_summary`
- `test_demo_programmatic_writes_output_image`
- `test_demo_programmatic_is_deterministic_for_fixed_seed`
- `test_demo_programmatic_semantic_state_has_target_prompt`

##### `TestRandomDemo`

- `test_demo_random_returns_summary`
- `test_demo_random_runs_without_crashing`
- `test_demo_random_can_cross_episode_boundaries`

##### `TestMain`

- `test_main_runs_both_demos`

Use `capsys` or `monkeypatch` only if needed. Keep these tests simple and behavior-oriented.

#### 7.2.2 Determinism Rule

For `demo_programmatic(seed=42, ...)`, the returned summary must be repeatable across runs:

- same prompt
- same step count
- same termination flags
- same final reward within floating-point tolerance

The saved image bytes do not need exact pixel-by-pixel assertions in Phase 3.

---

## 8. Edge Cases & Failure Modes

These are mandatory implementation behaviors.

| Scenario | Required Behavior |
|---|---|
| MCP tool called before `initialize_env` | raise helpful error |
| Resource read before `initialize_env` | return `initialized: false` JSON |
| `move` with nonexistent `element_id` | structured failure, no crash |
| `recolor` with invalid hex string | `ValueError` |
| `save_canvas` parent directory missing | create it |
| `initialize_env` called twice | replace prior env cleanly |
| `done` called immediately after reset | legal; returns terminal reward |
| custom content not in `CONTENT_TEMPLATES` | bridge via post-step patch |
| custom color not in `COLOR_PALETTE` | bridge via nearest palette + exact patch |
| random demo hits termination early | reset and continue until `total_steps` consumed |

---

## 9. Acceptance Checklist

Phase 3 is complete only when all of the following are true:

1. `server.py` runs as a valid FastMCP server via `python server.py`.
2. The server exposes the 5 required tools and `canvas://state`.
3. MCP actions operate through the existing `MarketCanvasEnv`.
4. `get_canvas_state` and `canvas://state` expose `prompt_id`.
5. `demo.py` contains both `demo_programmatic()` and `demo_random()`.
6. `demo_programmatic()` saves a PNG and prints final reward information.
7. `demo_random()` can run arbitrary sampled actions without crashing.
8. `tests/test_mcp.py` passes.
9. `tests/test_demo.py` passes.
10. Existing `tests/test_engine.py`, `tests/test_rewards.py`, and `tests/test_env.py` still pass.

---

## 10. Implementation Summary

The Phase 3 implementation should stay intentionally small:

- one top-level FastMCP server module
- one top-level demo module
- two focused new test files
- one dependency addition plus async-test support

The key engineering work is not boilerplate. It is the adapter logic that translates an LLM-friendly semantic interface into the existing RL-friendly env action space while preserving determinism, current reward logic, and the Phase 2 correctness fixes.
