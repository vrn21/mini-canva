# Bonus Research: Visual State for Mini-Canva


## 1. Executive Summary

The bonus requirement is already partly feasible in this codebase:

- `engine/renderer.py` can render the canvas to a PIL image or `np.ndarray`
- `env/market_canvas_env.py` already exposes `render_mode="rgb_array"`
- `env/wrappers.py` already has a `PixelObservationWrapper` that appends a `pixels` key

The real design question is not "can Mini-Canva render pixels?" but:

1. What exact visual observation contract should the environment expose?
2. Should pixels be primary, optional, or dual with semantic state?
3. What resolution is realistic for RL training?
4. How do we keep it deterministic and scalable?

The strongest recommendation from surveying established OSS RL environments is:

- Keep semantic state as the default observation.
- Expose RGB as an optional wrapper or mode, not as the default base observation.
- Return pixels as `uint8` RGB arrays.
- Keep two visual tiers:
  - native render for demos/export/debugging
  - downscaled training render for RL/VLM pipelines
- Support both:
  - `semantic + pixels`
  - `pixels_only`

For Mini-Canva specifically, a good contract is:

- default env observation: current semantic observation only
- optional wrapper:
  - `pixels_key="pixels"`
  - `size=(128, 96)` or `(256, 192)`
  - `include_semantic=True|False`
  - `dtype=np.uint8`
  - output shape `(H, W, 3)` in HWC layout

This matches how popular environments handle the same problem: symbolic-first envs usually add pixels through wrappers, while vision-first envs keep pixel observations compact and standardized.

## 2. What Popular OSS RL Environments Actually Do

### 2.1 Gymnasium Core Pattern: Pixels as an Optional Wrapper

[Gymnasium](https://gymnasium.farama.org/main/api/wrappers/observation_wrappers/) exposes rendered observations through `AddRenderObservation`, which replaced the older `PixelObservationWrapper`. It supports two important modes:

- `render_only=True`: replace the observation with pixels
- `render_only=False`: append pixels to the original observation dict

That is exactly the split Mini-Canva should support.

Important detail from Gymnasium docs:

- rendered observations are attached under a configurable key such as `pixels`
- resizing is handled by a separate wrapper like `ResizeObservation`
- Gymnasium explicitly notes that `AddRenderObservation` has no vector version

Implication for Mini-Canva:

- optional pixels are standard
- keeping resize separate is a common design
- naive pixel wrappers do not automatically solve large-scale training throughput

### 2.2 dm_control Pattern: `pixels_only` vs `state + pixels`

The DeepMind Control Suite has a dedicated pixel wrapper in [`dm_control/suite/wrappers/pixels.py`](https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/suite/wrappers/pixels.py). Its interface is very clean:

- `pixels_only=True`: discard low-dimensional state and keep only pixels
- `pixels_only=False`: preserve state and add pixels under a configurable key
- `render_kwargs`: control the renderer

This is one of the cleanest precedents for Mini-Canva because the environment is conceptually the same type of problem: the simulator has a native internal state, but some agents should train from rendered images instead.

Also relevant: the [`dm_control` README](https://github.com/google-deepmind/dm_control) spends real attention on rendering backends, including headless EGL and software fallbacks. That is a signal that once pixels matter for training, rendering infrastructure becomes part of the environment contract, not just a convenience method.

Implication for Mini-Canva:

- add a `pixels_only` option
- keep render configuration explicit
- treat rendering determinism and headless operation as first-class concerns

### 2.3 MiniGrid Pattern: Structured State by Default, RGB via Wrappers

[MiniGrid](https://minigrid.farama.org/api/wrapper/) is especially relevant because it is a lightweight, synthetic environment designed for speed and research control.

By default, MiniGrid observations are not RGB pixels. They are a compact symbolic encoding plus optional language fields. The docs explicitly say:

- default obs is a partially observable compact encoding, not pixels
- use `RGBImgPartialObsWrapper` or `RGBImgObsWrapper` to switch to RGB
- use `ImgObsWrapper` to strip away mission/language fields when training on image-only inputs

This is a strong precedent for Mini-Canva:

- symbolic or semantic state should stay the default because it is cheap and information-dense
- RGB should be layered on through wrappers
- image-only training should be available, but not forced on everyone

MiniGrid is also useful because it shows that one environment can support multiple observation families without confusing the RL API, as long as the wrappers are explicit.

### 2.4 MiniWorld Pattern: Pure RGB as the Native Observation

[MiniWorld](https://miniworld.farama.org/environments/hallway/) is a contrasting example. Its environments natively expose observations as RGB arrays such as `Box(0, 255, (60, 80, 3), uint8)`.

This works because:

- the task is fundamentally embodied and visual
- there is no richer low-dimensional state that is intended for the agent
- the observation size is already compact

Mini-Canva is not like MiniWorld. Mini-Canva already has an exact semantic scene graph. So pure-RGB-as-default would throw away high-quality information and make training harder for no real gain.

Implication for Mini-Canva:

- use MiniWorld as a reference for shape/dtype conventions
- do not copy MiniWorld's default observation choice

### 2.5 Procgen Pattern: Native Small RGB for Throughput

The [Procgen benchmark](https://github.com/openai/procgen) exposes native RGB observations with shape `(64, 64, 3)`. In its vectorized form, observations are returned under an `"rgb"` key.

The important lesson is not just "use pixels." It is:

- keep images small
- make the visual contract fixed and simple
- support vectorized execution cleanly

Procgen's observation size is intentionally compact. That is one reason it is usable as a large-scale benchmark.

Implication for Mini-Canva:

- if Mini-Canva ever targets large-scale visual RL, training observations should not stay at the full canvas size of `800x600`
- a dedicated lower-resolution render path is not optional if the environment is meant to scale

### 2.6 Atari / ALE Pattern: Aggressive Preprocessing + Native Vectorization

The [ALE docs](https://ale.farama.org/environments/) expose Atari observations in several modes:

- RGB: `Box(0, 255, (210, 160, 3), np.uint8)`
- grayscale
- RAM

The [ALE vector environment guide](https://ale.farama.org/vector-environment/) is even more relevant. It describes a high-performance vector environment equivalent to standard Atari preprocessing:

- frame skipping
- resizing
- grayscale conversion
- frame stacking
- native multi-threaded parallel execution

Typical vector output is shaped around stacked `84x84` frames.

Implication for Mini-Canva:

- if pixels are for serious training, preprocessing is part of the design
- resizing before the learner sees the observation is standard
- vectorized infrastructure matters far more than the wrapper itself once scale enters the picture

One useful difference: Mini-Canva is effectively static after each semantic action, unlike Atari. That means frame stacking is probably unnecessary unless Mini-Canva later grows cursor trajectories, animations, or transient UI states.

### 2.7 MineRL Pattern: Multimodal Dict Observations

[MineRL](https://minerl.readthedocs.io/en/v0.4.4/environments/handlers.html) uses `gym.spaces.Dict` observations made of multiple handlers. The visual `pov` field is an RGB `np.uint8` image, but it is not the whole observation. Environments may also include inventory, compass, and other structured channels.

This is one of the best precedents for Mini-Canva because it shows:

- pixels and structured state can coexist naturally in one observation dict
- agents with different capabilities can use different subsets
- multimodal training does not require abandoning structured fields

Implication for Mini-Canva:

- `{"elements": ..., "element_mask": ..., "pixels": ...}` is a normal RL environment shape
- a visual bonus does not need to replace the semantic observation

### 2.8 MiniWoB++ Pattern: Screenshot + DOM Together

[MiniWoB++](https://miniwob.farama.org/content/observation_space/) is directly relevant because it is a UI environment.

Its observation dict includes:

- `utterance`: task instruction
- `fields`: extracted task fields
- `screenshot`: RGB screenshot as `np.ndarray (H, W, 3)` with `uint8`
- `dom_elements`: structured visible DOM elements

This is almost the exact pattern Mini-Canva should follow if it wants to be useful for multimodal UI agents in the future:

- keep the screenshot
- keep the structured UI state
- do not force one modality to replace the other

### 2.9 BrowserGym / WorkArena Pattern: Multimodal Web-Agent Observations

[WorkArena](https://servicenow.github.io/WorkArena/) and [BrowserGym](https://github.com/ServiceNow/BrowserGym) push the same idea further. WorkArena explicitly describes BrowserGym as providing multimodal observations, including:

- AXTree / accessibility tree
- HTML
- screenshots

The ServiceNow write-up also explains why accessibility-tree style structure is valuable: raw HTML is too large, so they prefer structured interface representations that are easier for agents to act on.

This is highly relevant for Mini-Canva because your existing semantic state is already the environment's equivalent of a compact accessibility tree or scene graph.

Implication for Mini-Canva:

- semantic state should remain first-class even after visual state is added
- future multimodal training benefits most from paired structured + visual observations, not from pixels alone

## 3. Cross-Environment Patterns That Show Up Repeatedly

Across Gymnasium, dm_control, MiniGrid, MiniWoB, BrowserGym, MineRL, Procgen, and ALE, the same patterns keep appearing.

### Pattern A: Pixels are usually `uint8`, RGB, and unnormalized at the env boundary

Environments typically return raw `uint8` images and leave:

- float conversion
- normalization
- channel reordering
- batching

to the learner or training wrappers.

Recommendation for Mini-Canva:

- keep env output as `np.uint8`
- keep channel order as HWC `(H, W, 3)`
- let downstream code convert to CHW or float32 if needed

### Pattern B: Symbolic-first environments usually add pixels via wrappers

MiniGrid and dm_control are the cleanest examples. The base simulator keeps a compact state representation, then wrapper layers add pixels or replace the observation with pixels.

Recommendation for Mini-Canva:

- do not make pixels part of the default base env observation
- do make them available through a wrapper and/or observation mode

### Pattern C: UI-like environments keep visual and structural channels together

MiniWoB and BrowserGym do not choose between screenshot and structure. They keep both because they solve different failure modes:

- pixels capture visual layout, style, color, contrast, overlap, and appearance
- structure captures element identities, relationships, and action affordances

Recommendation for Mini-Canva:

- if the point is future multimodal training, the best training dataset is paired semantic state + rendered pixels

### Pattern D: Training pixels are usually smaller than demo pixels

Procgen uses `64x64`. ALE standardizes around `84x84`. MiniWorld uses compact camera images. Large raw frames are avoided unless the domain genuinely requires them.

Recommendation for Mini-Canva:

- support full-resolution renders for human inspection and export
- support reduced-resolution renders for learning

### Pattern E: Scale pressure quickly shifts from API design to systems design

At small scale, a wrapper is enough. At large scale, the bottlenecks are:

- rendering cost
- memory bandwidth
- vectorization strategy
- copying between CPU and learner
- serialization cost in rollouts or datasets

Mini-Canva's bonus feature is easy to prototype, but expensive to scale if implemented naively.

## 4. What This Means for Mini-Canva

Mini-Canva is unusual in a good way:

- it has a fully known symbolic canvas state
- it has a deterministic renderer
- the scene is simple: rectangles, text, image placeholders
- the visual state is valuable mostly for multimodal training, reward analysis, and debugging

That means Mini-Canva should not imitate Atari or MiniWorld blindly. It should take the multimodal pattern from MiniWoB, BrowserGym, MineRL, MiniGrid, and dm_control.

### Recommended Design Principle

Visual state should be:

- optional
- deterministic
- paired with semantic state by default
- configurable in resolution
- cheap to disable

## 5. Recommended Observation Contract

### 5.1 Base Environment

Keep the current base observation unchanged:

- `elements`
- `element_mask`
- `step_fraction`
- `prompt_id`

This remains the fastest and most information-dense observation for non-visual agents.

### 5.2 Add a Better Pixel Wrapper

The current `PixelObservationWrapper` is a good start, but it is too minimal for a future-facing visual contract. It should evolve toward:

```python
PixelObservationWrapper(
    env,
    size: tuple[int, int] | None = (128, 96),
    include_semantic: bool = True,
    pixels_key: str = "pixels",
    interpolation: str = "bilinear",
)
```

Behavior:

- if `include_semantic=True`:
  - return the original observation dict plus `pixels`
- if `include_semantic=False`:
  - return only the image observation

This mirrors Gymnasium and dm_control.

### 5.3 Keep HWC `uint8`

Recommended pixel tensor:

- shape: `(height, width, 3)`
- dtype: `np.uint8`
- color order: RGB

Reason:

- this is what Gymnasium, Procgen, MiniWoB, and most render APIs already expose
- it avoids premature coupling to a specific deep learning stack

### 5.4 Support Two Resolution Tiers

Mini-Canva should distinguish:

1. `native_pixels`
   - exact canvas resolution, currently `800x600`
   - used for debugging, PNG export, demos, qualitative inspection

2. `training_pixels`
   - downscaled view such as `256x192` or `128x96`
   - used in rollout buffers and visual learning pipelines

The environment should not force these to be the same thing.

### 5.5 Include Pixels in `reset()` and `step()`, Not Only `render()`

For RL, the observation must come back from `reset()` and `step()`. The `render()` method should remain available, but `render()` alone is not enough.

This is already partially solved by the wrapper; the missing part is making the wrapper configurable and intentionally designed rather than just attached.

## 6. Concrete Implementation Options

### Option A: Minimal Bonus Completion

Goal:

- satisfy the assignment cleanly
- keep code changes small

Implementation:

- keep `CanvasRenderer.render_to_array()`
- keep `MarketCanvasEnv.render()`
- extend `PixelObservationWrapper` with optional resizing
- document the shape/dtype contract
- add tests for:
  - key presence
  - shape
  - dtype
  - determinism for same canvas state

Pros:

- small change
- low risk
- enough for demos and basic visual-agent experiments

Cons:

- not optimized for scale
- no `pixels_only` mode
- no clean split between native export and training resolution

### Option B: Proper Multimodal RL Interface

Goal:

- make Mini-Canva genuinely useful for future multimodal training

Implementation:

- upgrade the wrapper to support:
  - `include_semantic`
  - `size`
  - `pixels_key`
- add a dedicated helper on the renderer:
  - `render_to_array(canvas, size=None)`
- optionally add `observation_mode`:
  - `"semantic"`
  - `"semantic+pixels"`
  - `"pixels"`
- add tests for all three modes

Pros:

- clean API
- directly aligned with Gymnasium + dm_control patterns
- works well for both structured and vision-first agents

Cons:

- slightly more API surface
- more tests and docs needed

### Option C: Scale-Oriented Visual Pipeline

Goal:

- prepare for large PPO-style parallel rollouts

Implementation:

- separate full-resolution renderer from training renderer
- batch/vectorize environments
- keep pixels disabled by default
- downscale aggressively for rollouts
- optionally pre-allocate buffers and reuse arrays
- consider a vector env that renders many canvases in one worker process

Pros:

- future-proof for serious experimentation

Cons:

- much more engineering than the bonus task itself
- probably overkill for this assignment repo right now

## 7. Resolution and Memory Tradeoffs

The raw canvas size matters a lot.

### 7.1 Full Resolution Is Expensive

At `800x600x3`, one observation is:

- `1,440,000` bytes
- about `1.37 MiB`

At `10,000` parallel environments, a single visual observation batch is:

- `14,400,000,000` bytes
- about `13.41 GiB`

That is before:

- frame stacking
- rollout storage
- learner copies
- model activations

So full-resolution pixels are not realistic as the default training observation.

### 7.2 More Practical Training Sizes

At `256x192x3`:

- one frame is `147,456` bytes
- `10,000` envs is about `1.37 GiB`

At `128x96x3`:

- one frame is `36,864` bytes
- `10,000` envs is about `351.6 MiB`

That is still substantial, but much more plausible.

### 7.3 Recommended Default

If you want one default training size, `128x96` is the safest starting point.

Why:

- preserves the `4:3` aspect ratio of `800x600`
- keeps text/buttons/layout visible enough for coarse design reasoning
- does not explode memory immediately

If qualitative fidelity matters more than throughput, use `256x192`.

## 8. Determinism Requirements

If visual state is meant for RL training, it must be reproducible.

Mini-Canva is already in good shape here because the renderer is stateless apart from the font cache, but there are still some details to lock down.

### 8.1 Font Determinism

Current renderer behavior:

- default PIL bitmap font unless a TTF path is provided
- default bitmap font does not support arbitrary font sizes well

For visual training, this is a real issue because:

- text rasterization may differ across environments if font selection changes
- a future VLM policy can become sensitive to tiny glyph differences

Recommendation:

- bundle one known TTF font with the repo
- make it the default renderer font
- test that the same canvas renders identically across resets on the same machine

### 8.2 Fixed Draw Order

Mini-Canva already uses list order as z-order, which is good. The render spec should explicitly guarantee:

- background first
- elements in z-order
- no hidden randomness in rendering

### 8.3 Resize Determinism

If downscaling is added, interpolation mode should be fixed in code and documented. Otherwise the same env can produce slightly different images depending on the library default.

## 9. Performance Bottlenecks to Expect

If this bonus evolves into real multimodal training, these are the likely bottlenecks.

### 9.1 CPU Rendering Cost

Current rendering is PIL-based and CPU-only. For a simple 2D scene this is fine at small scale, but at large rollout counts it will become expensive.

Most likely hotspot:

- rendering text repeatedly

Why it matters:

- every step may require a full raster pass even when only one object moved

Mitigations:

- keep visual state optional
- use lower training resolutions
- cache fonts aggressively
- consider dirty-region rendering later only if profiling shows it matters

### 9.2 Memory Bandwidth and Copies

Converting PIL images to NumPy arrays creates large memory movement. At scale, copies can dominate.

Mitigations:

- render only when pixels are requested
- avoid full-res rollout storage
- preallocate or reuse arrays if later profiling justifies it

### 9.3 Python-Level Vectorization Limits

Gymnasium's own render-observation wrapper has no vector version. ALE had to build native vector support to get serious throughput.

Implication:

- a per-env Python wrapper is fine for the bonus task
- it is not the final answer for 10k parallel visual rollouts

### 9.4 Serialization for Offline Data

If you later log both semantic state and pixels for every step, datasets will grow quickly. PNG compression is helpful for export, but rollout-time training wants raw arrays, not file writes.

Recommendation:

- keep export and training pathways separate

## 10. Recommendation for This Repo

### Recommended near-term choice

Implement the visual bonus as an optional multimodal observation path, not as a new default observation.

Concretely:

1. Keep the current semantic base env exactly as-is.
2. Upgrade `PixelObservationWrapper` to support:
   - `size`
   - `include_semantic`
   - `pixels_key`
3. Keep `render()` for native full-resolution RGB export.
4. Document one default training size, preferably `128x96`.
5. Add tests covering:
   - output dtype `uint8`
   - output shape
   - same state -> same pixels
   - wrapper mode with and without semantic state

### Recommended medium-term choice

If you later care about serious multimodal RL:

- add `pixels_only` mode
- bundle a deterministic TTF font
- add vectorized env support or a batched renderer
- keep rollout images smaller than export images

## 11. Why This Is the Right Fit for Mini-Canva

Mini-Canva is not a game benchmark where pixels are the only observable world state. It is closer to a UI environment with a perfect internal scene graph. That makes the best design much clearer:

- for reasoning efficiency: semantic state
- for visual grounding: RGB pixels
- for future multimodal training: both together

That is exactly the pattern used by modern UI/web-agent environments like MiniWoB and BrowserGym, and it is also compatible with wrapper-based designs from Gymnasium, MiniGrid, and dm_control.

## 12. Suggested Follow-Up Spec

If this should turn into an implementation task later, the next spec should define:

- wrapper constructor API
- resize policy and interpolation mode
- observation space definitions for:
  - semantic
  - semantic + pixels
  - pixels only
- deterministic font choice
- tests and benchmark targets

## 13. Sources

- [REQS.md in this repo](./REQS.md)
- [Gymnasium observation wrappers](https://gymnasium.farama.org/main/api/wrappers/observation_wrappers/)
- [dm_control repository README](https://github.com/google-deepmind/dm_control)
- [dm_control pixel wrapper source](https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/suite/wrappers/pixels.py)
- [MiniGrid wrapper docs](https://minigrid.farama.org/api/wrappers/)
- [MiniGrid training docs](https://minigrid.farama.org/main/content/training/)
- [MiniWorld Hallway docs](https://miniworld.farama.org/environments/hallway/)
- [Procgen repository](https://github.com/openai/procgen)
- [ALE environment docs](https://ale.farama.org/environments/)
- [ALE vector environment guide](https://ale.farama.org/vector-environment/)
- [MineRL environment handlers docs](https://minerl.readthedocs.io/en/v0.4.4/environments/handlers.html)
- [MiniWoB++ observation space docs](https://miniwob.farama.org/content/observation_space/)
- [BrowserGym repository](https://github.com/ServiceNow/BrowserGym)
- [WorkArena project page](https://servicenow.github.io/WorkArena/)
- [ServiceNow WorkArena blog post](https://www.servicenow.com/blogs/2024/introducing-workarena-benchmark)
