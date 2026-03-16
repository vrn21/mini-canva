# WRITEUP: MarketCanvas-Env Architecture & Design Decisions

**TL;DR:** A minimalist RL environment and MCP server for design agents. Features dual semantic/visual states, semantic/low-level action spaces, and a heuristic-based design reward.

## 1. Environment Formulation

The `Canvas` engine is a pure-data state container managing `Text`, `Shape`, and `Image` elements. Elements possess core physical and styling properties (`x`, `y`, `width`, `height`, `z-index`, `color`, `content`). They are strictly managed in a back-to-front list (dictating z-order compliance) paired with a dictionary index for ID lookups.

### 1.1 State Space Mapping

The agent perceives the canvas through two distinct state representations:

#### Semantic State (DOM-like JSON)

The environment emits a complete property graph exposing every element and spatial relationship. For LLM agents interacting via MCP, this structured JSON is their only perception.

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
      "type": "SHAPE",
      "x": 0,
      "y": 0,
      "width": 800,
      "height": 600,
      "z_index": 0,
      "color": "#FF8C00",
      "text_color": "#000000",
      "content": "rectangle",
      "font_size": 16
    },
    {
      "id": "element_1",
      "type": "TEXT",
      "x": 200,
      "y": 150,
      "width": 400,
      "height": 100,
      "z_index": 1,
      "color": "#FFFFFF",
      "text_color": "#FFFFFF",
      "content": "Summer Sale!",
      "font_size": 24
    }
  ],
  "element_count": 2,
  "target_prompt": "Design a holiday greeting card with a festive image, a greeting message, and a decorative border",
  "step_count": 2,
  "max_steps": 10,
  "spatial_relationships": [
    {
      "element_a": "element_0",
      "element_b": "element_1",
      "a_left_of_b": false,
      "a_right_of_b": false,
      "a_above_b": false,
      "a_below_b": false,
      "overlaps": true,
      "overlap_area": 40000,
      "a_contains_b": true,
      "b_contains_a": false,
      "center_dx": 0,
      "center_dy": -100
    }
  ],
  "prompt_id": 3,
  "initialized": true,
  "session_id": "44a51bd2cb3f481e84268d7db09b66fa"
}

```

#### Visual State (RGB Array)

For Vision-Language Models (VLMs), the canvas is rasterized to a configured NumPy RGB array (e.g., `128x96x3`). This is essential for evaluating overlapping semantics and relative contrast that raw bounding boxes obscure.

### 1.2 Action Space Grounding

The action space enforces one of two disjoint interaction interfaces, dictating how the agent manipulates the state.

#### High-Level Semantic UI (Default)

The agent acts as an omnipotent API client using a parameterized discrete action space `Dict(action_type, element_idx, x, y, width, height, color_idx, content_idx)`.The available tools are:

* `add_text` / `add_shape` / `add_image`: Instantiates elements using a constrained palette of 16 hex colors and 20 text content templates.
* `move`: Translates an existing element to an absolute `(x, y)` coordinate.
* `recolor`: Updates the `color` property (and contrasts `text_color` if it's a shape).
* `remove`: Deletes an element.
* `done`: Terminates the episode for evaluation.

#### Low-Level Computer Use

The agent acts as a human proxy using a continuous mouse/keyboard interface `Dict(action_type, x, y, x2, y2, tool, text)`. This exponentially compounds credit assignment difficulty but eliminates the sim-to-real gap, yielding a policy technically capable of operating native Figma via OS accessibility APIs. The available tools are:

* `mouse_move`: Repositions the virtual cursor.
* `mouse_click`: Clicks the active coordinate to select or interact.
* `mouse_drag`: Drags from `(x, y)` to `(x2, y2)` to draw or move elements.
* `keyboard_type`: Injects raw string content (up to 64 chars).
* `set_tool`: Switches the active cursor mode (`SELECT`, `TEXT`, `SHAPE`, `IMAGE`).

## 2. The Reward Engine

The reward engine maps subjective visual design quality to a strict scalar `$R \in [-1.0, 1.0]$`. The final reward is a weighted linear combination of five normalized sub-components: Constraints (0.35), Aesthetics (0.25), Accessibility (0.20), Coverage (0.10), and Efficiency (0.10).

### 2.1 Heuristic Calculation Mechanics

Each sub-component evaluates a distinct dimension of the state graph:

* **Constraints (`ConstraintChecker`):** Validates prompt satisfaction. It dispatches to specific checkers (`HAS_ELEMENT`, `ELEMENT_COLOR`, `MIN_ELEMENTS`) that evaluate the canvas via queries.
* **Aesthetics (`AestheticsScorer`):** Scores layout geometry. It averages four sub-scores: Overlap (penalizes bounding box intersections), Alignment (fraction of elements sharing $X/Y$ center or left-edge axes within a 10px tolerance), Margin (fraction of elements staying 20px away from canvas boundaries), and Spacing (minimizing the normalized standard deviation of vertical gaps between elements).
* **Accessibility (`AccessibilityChecker`):** Enforces WCAG 2.1 AA text contrast. It queries the z-ordered canvas to find the effective background color directly behind text, calculating the relative luminance ratio. Text must clear 4.5:1 (or 3.0:1 for large fonts).
* **Coverage:** Penalizes abnormally small or massive canvases. It calculates the ratio of combined element bounding-box areas to the total canvas area, peaking at an optimal $40$-$80\%$ coverage ratio.
* **Efficiency:** A linear decay based on `steps_taken / max_steps` to incentivize shorter episodes.

### 2.2 Attack Vectors & Reward Hacking

Agents exploit heuristics by shrinking elements to 1-pixel widths or hiding shapes beneath identical foregrounds. 

From a successful MDP score perspective, a highly rewarded image is one that satisfies the constraints, has good aesthetics, and is accessible; but from an artist's perspective, the image looks poor. This discrepancy shows mathematical proxies fail to capture visual aesthetics.

## 3. Scaling to 10,000 Parallel PPO Rollouts

### 3.1 Bottlenecks

1. **PIL rendering is Python-bound.** Every `step()` in VLM mode triggers `CanvasRenderer.render()` — a Python `for` loop over elements with per-call GIL acquire/release. 500k render calls per epoch (10k envs × 50 steps).
2. **Repeated list copies.** `get_all_elements()` returns `list(self._elements)` (full copy), called 5–8 times per `step()` across obs/reward/spatial-relationship paths.
3. **$O(n^2)$ Python loops.** `get_overlapping_pairs()` and `_build_spatial_relationships()` both do pairwise Python iteration on every reward computation and state query.
4. **Global singleton session lock.** One `threading.RLock` around one `MarketCanvasEnv`. Serializes all MCP calls — 10k rollouts need 10k isolated instances.
5. **Scalar observation fill.** `_get_semantic_obs()` allocates + fills `np.zeros((20,15))` element-by-element in Python every step.
6. **Double reward computation.** `env.step()` and `_step_payload()` both call `compute_reward()` at terminal steps.

### 3.2 Addressing Each Bottleneck

1. **PIL rendering → batched rasterization.** Two options: (a) Brax-style — replace `Canvas` with a JAX pytree of shape `[num_envs, max_elements, ...]`, render via vectorized array ops on GPU (`jax.vmap` over a pure-function rasterizer). (b) Rust — port `CanvasRenderer` to `tiny-skia`, expose via PyO3; $N$ core-pinned OS threads render in parallel, no GIL.
2. **List copies → struct-of-arrays.** Store element properties as contiguous arrays (`x`, `y`, `width`, `height` as `[num_envs, max_elements]` tensors + a `mask` array). Eliminates per-call `list()` copies — obs, reward, and spatial queries all read the same backing arrays via slicing.
3. **$O(n^2)$ Python → batched tensor ops.** Pairwise overlap becomes a single broadcasted intersection: `jnp.maximum(0, min(rights[:,:,None], rights[:,None,:]) - max(lefts[:,:,None], lefts[:,None,:]))`. One fused kernel for all 10k envs × all element pairs.
4. **Singleton lock → per-rollout isolation.** Two paths: (a) SampleFactory-style — $W$ worker processes each owning $10000/W$ independent env instances, no shared state, no locks. (b) EnvPool-style — lock-free action queue (crossbeam `ArrayQueue` / `AtomicUsize` slot allocation) dispatches to Rust worker threads, async `send`/`recv` decouples action dispatch from observation collection.
5. **Scalar fill → vectorized construction.** With struct-of-arrays, `_get_semantic_obs()` becomes a slice + normalize: `obs = state_arrays[:, :max_elements] / norm_constants`. No per-element Python loop, no per-step allocation.
6. **Double reward → cache terminal result.** `compute_reward()` result is computed once in `env.step()` at terminal steps and stored in `info`; `_step_payload()` reads from `info` instead of recomputing.

## 4. Trade-offs and Limitations

* **Constrained Action Palette:** 16 colors and 20 templates prevent synthesis of novel content, limiting agents to discrete composition.
* **Heuristic Reward Limits:** Pure mathematical proxies (contrast, alignment) are gameable and fail to capture subjective design quality available via human preference signals (RLHF/DPO).
* **2D Sim-to-Real Gap:** Rectangular fills ignore gradients, shadows, and vector paths (Figma/Canva features), requiring heavy domain adaptation for production transfer.
* **Global Session Lock:** `ServerSession` uses a threading lock, restricting performance to single-client interactions. Scaling requires session pooling or per-request instantiation.
* **Static Prompting:** 5 hardcoded prompts limit training diversity. Production use requires procedural prompt generation or larger dataset integration.

## 5. Conclusion

MarketCanvas-Env is a minimal but complete RL environment that covers the full MDP loop: a pure-data canvas engine, dual observation modalities (semantic JSON + RGB pixels), two disjoint action interfaces (semantic API + low-level computer use), and a multi-component heuristic reward. It simultaneously functions as an MCP server for direct LLM tool-calling interaction. The primary open question is whether heuristic rewards can bootstrap useful design behavior, or whether human preference signals are required from the start.
