# Bonus Low-Level Tools Design

> Purpose: propose an end-to-end design for the optional low-level computer-use tools from `specs/REQS.md`:
> `mouse_move(x, y)`, `mouse_click()`, `mouse_drag(x1, y1, x2, y2)`, and `keyboard_type(string)`.
>
> Scope of this document: research synthesis, repo-specific reasoning, API design, environment design, MCP design, compatibility rules, and a concrete implementation/test plan.
>
> Non-goal: this document does not implement the feature. Implementation should start only after approval.

---

## 1. Current Repo Reality

The current codebase is cleanly optimized for a **semantic** action path:

- `env/spaces.py` exposes a discrete/factored Gym action space for `add_text`, `add_shape`, `add_image`, `move`, `recolor`, `remove`, and `done`.
- [`/Users/vrn21/Developer/verita/mini-canva/env/market_canvas_env.py`](/Users/vrn21/Developer/verita/mini-canva/env/market_canvas_env.py) is the only episode/state transition authority.
- [`/Users/vrn21/Developer/verita/mini-canva/server.py`](/Users/vrn21/Developer/verita/mini-canva/server.py) is a thin MCP adapter over that semantic env.
- Tests only cover the semantic action path today.

That means the correct extension strategy is not "replace the env with a pixel UI simulator." The correct strategy is:

1. keep the semantic path as the default and stable interface,
2. add a separate optional low-level interaction layer,
3. route low-level gestures into the same canvas state and episode counters,
4. avoid breaking existing semantic tests, tools, or demos.

This matches the user's requirement that low-level tools should be available, but optional.

---

## 2. Research Summary: OSS RL Envs Using Low-Level Tools

I reviewed representative OSS environments where low-level computer-use actions are central or configurable.

### 2.1 MiniWoB++

MiniWoB++ exposes browser-task actions such as:

- coordinate click actions,
- text typing actions,
- focus/element interaction primitives.

Its docs explicitly define action classes like `CoordClick`, `CoordClickMulti`, `Type`, and `FocusAndType`, which shows a practical pattern: keep the action vocabulary small, but give each action a clear operational meaning.  
Source: [MiniWoB action API](https://miniwob.farama.org/content/actions/)

### 2.2 BrowserGym

BrowserGym is useful because it does not lock the benchmark to a single action interface. Its docs describe **action sets** that can be swapped or constrained, including a `HighLevelActionSet`, subsets, and strict Python action definitions. That is the strongest precedent for making low-level control **configurable rather than mandatory**.  
Source: [BrowserGym action sets](https://browsergym.readthedocs.io/latest/core/action_space/)

### 2.3 AndroidEnv

AndroidEnv is an RL environment for Android device interaction. The repo frames the environment around realistic device control and observation pipelines rather than domain-specific semantic actions. That is relevant here because it shows the infrastructure cost of treating low-level control as first-class: the environment needs an explicit interaction state machine, not just a set of direct semantic mutations.  
Source: [android_env repository](https://github.com/google-deepmind/android_env)

### 2.4 WebArena

WebArena is task-oriented rather than a raw action-space tutorial, but it is still useful because it evaluates agents in realistic interfaces where observation, grounding, and action execution are tightly coupled. The practical lesson is that once the interface becomes computer-use-like, the runtime must represent **interaction context** such as focused element, selected target, and current cursor position.  
Source: [WebArena](https://webarena.dev/)

### 2.5 OSWorld

OSWorld pushes further into desktop-like multimodal computer use. The relevant takeaway is not that Mini-Canva should copy OSWorld's complexity. The takeaway is that true low-level control introduces state beyond the task state itself: cursor, focus, drag intent, and mode transitions.  
Source: [OSWorld](https://os-world.github.io/)

---

## 3. What the Research Implies for This Repo

Three patterns are consistent across the OSS systems above:

### 3.1 Low-level control needs an interaction model, not just extra tool names

Adding `mouse_move`, `mouse_click`, `mouse_drag`, and `keyboard_type` as direct aliases to semantic actions would be incorrect. Low-level environments work because they define how gestures map to state transitions through:

- hit testing,
- focus,
- selection,
- drag semantics,
- typing targets,
- no-op/failure handling.

This repo currently has none of that. It must be introduced explicitly.

### 3.2 Action-set configurability is the right fit

BrowserGym's configurable action sets are the strongest architectural match for this project. The semantic action path is already implemented and stable. Low-level actions should therefore be:

- available only when requested,
- isolated from the semantic default,
- internally mapped into the same underlying canvas state.

This gives us bonus functionality without damaging the benchmark's current tractability.

### 3.3 The low-level layer should stay minimalist

MiniWoB++, AndroidEnv, and OSWorld all show how large the state/action surface can become. For this repo, copying full desktop-style interaction would be overkill. The correct version is a **minimal deterministic canvas UI model** that supports only the gestures required by `REQS.md`.

---

## 4. Repo-Specific Design Goals

The design for Mini-Canva should satisfy these goals:

1. Preserve the current semantic env and MCP server behavior by default.
2. Make low-level tools opt-in at env initialization time.
3. Keep all state deterministic and serializable.
4. Avoid duplicating reward logic.
5. Keep the current canvas engine as the source of truth for element storage.
6. Make low-level tool failures structured and debuggable.
7. Expose enough interaction state to let LLMs/RL agents use the low-level interface effectively.

---

## 5. Proposed High-Level Architecture

I recommend adding a thin interaction/controller layer between gestures and canvas mutations.

```text
Low-level MCP tools / low-level Gym actions
                  |
                  v
      Interaction Controller (new)
      - cursor state
      - selection
      - focus
      - drag intent
      - tool mode
                  |
                  v
       Canvas mutations via existing Canvas API
                  |
                  v
      MarketCanvasEnv step counters / reward / termination
```

The important point is separation of concerns:

- `engine/` stays a pure canvas engine.
- `env/market_canvas_env.py` stays the episode authority.
- a new controller translates low-level gestures into deterministic mutations.
- `server.py` exposes low-level MCP tools only when configured.

---

## 6. Core Design Decision: Optional Interaction Mode

### 6.1 New mode flag

Add an explicit action-interface setting:

- `action_interface="semantic"` as the default
- `action_interface="low_level"`
- `action_interface="hybrid"` optionally, if we want both in the same session

My recommendation for implementation order is:

1. support `semantic` and `low_level`,
2. optionally support `hybrid` only if it remains simple.

### 6.2 Why default must remain semantic

The current env, tests, and MCP tooling are all semantic-first. Making low-level the default would:

- break the current contract,
- make simple tool-calling workflows harder,
- inflate agent difficulty dramatically,
- add interaction state to every episode even when not needed.

So low-level support should be enabled only by explicit initialization.

---

## 7. Proposed Low-Level Interaction Model

The bonus tools in `REQS.md` are underspecified. To make them usable, the environment needs a minimal UI grammar.

### 7.1 Interaction state to add

Per env session, maintain:

- `cursor_x`, `cursor_y`
- `selected_element_id | None`
- `focused_element_id | None`
- `active_tool`
- `pending_text_buffer | None` only if needed

I recommend `active_tool` values:

- `select`
- `text`
- `shape`
- `image`

This is the smallest useful set that allows low-level actions to create the three supported element types.

### 7.2 Why tool mode is necessary

Without a tool mode, `mouse_drag` cannot deterministically answer:

- does dragging move an existing element?
- create a shape?
- create an image?
- select a region?

Real software solves this with toolbar mode. Mini-Canva should too.

### 7.3 How agents switch tools

Because `REQS.md` only names four low-level tools, we should avoid inventing many extra tools. The cleanest answer is:

- keep the existing semantic `execute_action` tool available in low-level sessions only for **tool selection and explicit done**, or
- add one minimal MCP tool: `set_active_tool(tool)` plus `done`.

I recommend the second option because it keeps the low-level surface coherent and explicit.

Minimal added MCP helpers:

- `set_active_tool(tool: Literal["select", "text", "shape", "image"])`
- `submit_episode()` or continue using semantic `done`

This is the one place where I would intentionally extend beyond the four literal bonus gestures. Without tool selection, low-level creation is ambiguous.

---

## 8. Exact Semantics of Each Low-Level Tool

### 8.1 `mouse_move(x, y)`

Behavior:

- clamps cursor to canvas bounds,
- updates cursor position,
- does not mutate canvas elements,
- does not select or focus anything by itself.

Step accounting:

- counts as one env step when invoked through MCP or low-level env step,
- returns current observation and current reward snapshot,
- never terminates by itself.

### 8.2 `mouse_click()`

Behavior depends on `active_tool`.

If `active_tool == "select"`:

- hit-test topmost element under cursor,
- if hit: select that element and focus it if it is text-editable,
- if no hit: clear selection and focus.

If `active_tool == "text"`:

- create a default text element at cursor position,
- select and focus the new text element.

If `active_tool == "shape"`:

- create a default button/rectangle centered at cursor,
- select the new shape.

If `active_tool == "image"`:

- create a default image placeholder centered at cursor,
- select the new image element.

Why this choice:

- single-click creation with default sizes is standard in UI tools,
- it avoids requiring drag for every insertion,
- it keeps the action usable for LLM tool-calling.

### 8.3 `mouse_drag(x1, y1, x2, y2)`

This tool should be treated as a complete gesture, not dependent on current cursor state.

Recommended semantics:

- move cursor to `(x1, y1)`,
- interpret drag using `active_tool`,
- end with cursor at `(x2, y2)`.

If `active_tool == "select"`:

- hit-test at `(x1, y1)`,
- if an element is hit, drag moves that element by delta `(x2 - x1, y2 - y1)`,
- dragged element becomes selected,
- if no element is hit, return structured no-op failure.

If `active_tool == "text"`:

- create a text element with bounding box defined by drag rect,
- focus/select the new text element.

If `active_tool == "shape"`:

- create a shape with bounding box defined by drag rect.

If `active_tool == "image"`:

- create an image placeholder with bounding box defined by drag rect.

Why not support resize in `select` mode initially:

- resize needs handles and more UI state,
- move is the dominant useful gesture,
- it keeps the first version deterministic and testable.

### 8.4 `keyboard_type(text)`

Behavior:

- requires `focused_element_id` pointing to a text-capable element,
- text-capable elements are `TEXT` and `SHAPE`,
- updates the focused element's `content`,
- for `TEXT`, content affects the rendered text directly,
- for `SHAPE`, content updates the button label,
- for `IMAGE`, return structured failure.

Failure cases:

- no focused element,
- focused element removed,
- text too long if we choose to cap length.

I recommend **replace-content semantics**, not append semantics, for v1:

- easier for RL credit assignment,
- deterministic,
- avoids needing cursor position within text,
- matches the repo's existing model where content is a whole-field property.

---

## 9. Observation Changes Needed

Low-level control is not usable unless the observation exposes the interaction state.

### 9.1 Semantic state additions

Extend the semantic state payload with:

- `interaction.action_interface`
- `interaction.active_tool`
- `interaction.cursor`
- `interaction.selected_element_id`
- `interaction.focused_element_id`

Example:

```json
{
  "interaction": {
    "action_interface": "low_level",
    "active_tool": "text",
    "cursor": {"x": 120, "y": 90},
    "selected_element_id": "element_3",
    "focused_element_id": "element_3"
  }
}
```

This must appear in both:

- `get_semantic_state()`
- MCP responses based on semantic state

### 9.2 Gym observation space additions

If low-level mode is enabled for Gym usage, the tensor observation should also include compact interaction features:

- normalized cursor position,
- selected slot index or sentinel,
- focused slot index or sentinel,
- active tool id.

This avoids a mismatch where MCP sees interaction state but RL observations do not.

---

## 10. Gym Environment Design

### 10.1 Do not overload the current semantic action space

The existing semantic discrete action dict should stay unchanged.

Instead, add a separate action-space builder for low-level mode, for example:

- `build_low_level_action_space(...)`

Proposed structure:

```python
spaces.Dict(
    {
        "action_type": spaces.Discrete(5),  # move, click, drag, type, done
        "x": spaces.Discrete(canvas_width),
        "y": spaces.Discrete(canvas_height),
        "x2": spaces.Discrete(canvas_width),
        "y2": spaces.Discrete(canvas_height),
        "tool": spaces.Discrete(4),         # optional if embedded in step, but not preferred
        "content_idx": spaces.Discrete(len(CONTENT_TEMPLATES)),
    }
)
```

But for env cleanliness, I do **not** recommend encoding tool switching into every step payload. Tool switching should be its own explicit action or helper method.

So the stronger design is:

- low-level env action types: `move`, `click`, `drag`, `type`, `set_tool`, `done`
- low-level MCP tools: same conceptual surface

### 10.2 Env internals

`MarketCanvasEnv.step()` should branch on `self.action_interface`:

- semantic interface -> current `_execute_action`
- low-level interface -> new `_execute_low_level_action`

Reward, truncation, and termination behavior should remain shared.

This preserves all reward semantics and keeps one env class.

---

## 11. MCP Server Design

### 11.1 Initialization

Extend `initialize_env(...)` with:

- `action_interface: Literal["semantic", "low_level"] = "semantic"`

Potentially later:

- `enable_semantic_tools: bool = True`

My recommendation:

- in `semantic` mode, expose the current tool behavior as-is,
- in `low_level` mode, keep semantic read tools but use low-level mutation tools,
- optionally still allow `execute_action` for backward compatibility only if explicitly enabled.

### 11.2 MCP tool surface

Current read tools can stay:

- `initialize_env`
- `get_canvas_state`
- `get_observation`
- `get_current_reward`
- `save_canvas`

New optional low-level mutation tools:

- `set_active_tool`
- `mouse_move`
- `mouse_click`
- `mouse_drag`
- `keyboard_type`
- `submit_episode` or reuse `execute_action(..., action_type="done")`

### 11.3 Why separate tools instead of overloading `execute_action`

`REQS.md` explicitly describes low-level interaction as separate tool-like operations. Separate MCP tools are better because:

- they mirror real computer-use APIs,
- they are easier for tool-calling LLMs to chain,
- they make logs and tests clearer,
- they avoid a polymorphic mega-tool with weak validation.

---

## 12. Mapping Low-Level Tools to Canvas Operations

The low-level layer should still end in the same canvas operations the repo already supports.

### 12.1 Creation mappings

- text click/drag -> `Canvas.add_element(ElementType.TEXT, ...)`
- shape click/drag -> `Canvas.add_element(ElementType.SHAPE, ...)`
- image click/drag -> `Canvas.add_element(ElementType.IMAGE, ...)`

### 12.2 Selection and move mappings

- hit-test uses `Canvas.get_elements_at(x, y)`
- movement uses `Canvas.move_element(...)`

### 12.3 Typing mappings

- typing uses `Canvas.update_element(element_id, content=text)`

### 12.4 Color behavior

The four bonus low-level tools do not include recoloring. That is fine. The low-level feature can ship without low-level recolor support and still satisfy the bonus requirement. Recolor can remain semantic-only unless we later add a palette/fill tool model.

This is an important scope boundary.

---

## 13. Proposed File-Level Changes

If approved, I would implement roughly this layout:

- `env/spaces.py`
  - add `ACTION_INTERFACE_*` constants
  - add low-level action constants
  - add `build_low_level_action_space()`
  - extend observation space for interaction features when needed

- `env/interaction.py` or `env/low_level_controller.py`
  - new interaction-state dataclass
  - hit testing helpers
  - low-level gesture interpreter

- [`/Users/vrn21/Developer/verita/mini-canva/env/market_canvas_env.py`](/Users/vrn21/Developer/verita/mini-canva/env/market_canvas_env.py)
  - accept `action_interface`
  - initialize/reset interaction state
  - branch semantic vs low-level stepping
  - include interaction data in semantic state and info/observation

- [`/Users/vrn21/Developer/verita/mini-canva/server.py`](/Users/vrn21/Developer/verita/mini-canva/server.py)
  - accept `action_interface` in `initialize_env`
  - register low-level tools
  - validate tool availability by interface
  - return interaction state in payloads

- `tests/test_env.py`
  - add low-level env stepping tests
  - add interaction-state reset tests

- `tests/test_mcp.py`
  - add low-level MCP integration tests
  - verify semantic mode remains unchanged

- `demo.py`
  - optionally add a low-level scripted demo path

---

## 14. Failure Model and Determinism Rules

Low-level tools must not silently mutate unpredictably.

### 14.1 Structured failures

Each tool should return:

- attempted action,
- success boolean,
- relevant `element_id` if any,
- current interaction state,
- current reward snapshot.

Examples of valid structured failures:

- `mouse_click` in select mode on empty canvas,
- `mouse_drag` starting on empty space in select mode,
- `keyboard_type` with no focused element,
- tool call used in semantic-only session.

### 14.2 Deterministic defaults

Default sizes and placements must be fixed:

- text click insert size is fixed,
- shape click insert size is fixed,
- image click insert size is fixed.

This is critical for RL reproducibility and testability.

---

## 15. Testing Strategy

### 15.1 Env tests

I would add tests for:

- low-level mode builds a different action space
- reset initializes cursor/tool/selection state deterministically
- `set_tool(text)` then `click` creates a text element
- `set_tool(shape)` then `drag` creates a shape with dragged bounds
- `set_tool(select)` then `drag` moves the topmost hit element
- `keyboard_type` updates focused text content
- typing without focus returns structured failure
- done/termination still computes reward correctly

### 15.2 MCP tests

I would add tests for:

- `initialize_env(action_interface="low_level")`
- low-level tool names are available
- `mouse_move` updates cursor
- `mouse_click` in text mode creates a text element
- `mouse_drag` in shape mode creates a shape
- `mouse_drag` in select mode moves an element
- `keyboard_type` edits focused text
- calling low-level tools in semantic mode errors cleanly
- semantic tools still behave identically in semantic mode

### 15.3 Regression coverage

The most important regression tests are:

- all existing semantic tests still pass,
- reward behavior is unchanged for existing semantic episodes,
- `get_canvas_state` and `get_observation` remain backward-compatible except for additive interaction fields.

---

## 16. Phased Implementation Plan

If approved, I would implement in this order:

1. add action-interface config and interaction state model
2. implement low-level env stepping and semantic state extensions
3. expose MCP low-level tools
4. add env and MCP tests
5. add or update demo/docs
6. run full test suite via `uv`

This ordering minimizes risk because the env/controller behavior is validated before MCP exposure.

---

## 17. Trade-Offs and Constraints

### 17.1 What this design intentionally does not do

- no full toolbar UI rendering,
- no resize handles,
- no text cursor positions,
- no append/backspace editing model,
- no low-level recolor flow,
- no hybrid multi-pointer or transient UI animation system.

That is deliberate. The goal is to satisfy the bonus requirement with a deterministic, trainable, minimal computer-use layer.

### 17.2 Why this is still sufficient

The requirement asks for low-level tools, not a full Canva clone. This design provides:

- true coordinate-based interaction,
- true click/drag/type semantics,
- deterministic task execution,
- compatibility with the existing semantic benchmark.

---

## 18. Recommendation

I recommend approving implementation with these exact boundaries:

1. keep semantic mode as the default,
2. add `action_interface="low_level"` as an opt-in mode,
3. add a minimal controller with cursor, selection, focus, and active tool,
4. expose dedicated MCP low-level tools,
5. keep recolor and other advanced editing semantic-only for now.

This gives the project a credible low-level computer-use path without turning the repo into a full GUI simulator.

---

## 19. References

- [REQS.md](./REQS.md)
- [MiniWoB action API](https://miniwob.farama.org/content/actions/)
- [BrowserGym action sets](https://browsergym.readthedocs.io/latest/core/action_space/)
- [android_env repository](https://github.com/google-deepmind/android_env)
- [WebArena](https://webarena.dev/)
- [OSWorld](https://os-world.github.io/)
