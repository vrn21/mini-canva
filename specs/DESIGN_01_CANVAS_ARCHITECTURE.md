# Design Document 01: Real-World Canvas Architecture vs. Mini-Canva

> How production design tools (Canva, Figma) are actually built, and how their core principles map to our minimalist simulation.

---

## 1. How Real-World Canvas Systems Work

### 1.1 The Scene Graph — The Heart of Every Canvas Tool

Every production canvas tool (Canva, Figma, Sketch, Adobe Illustrator) is built on a **scene graph**: a tree data structure where each node represents a visual element on the canvas.

```
Canvas (root)
├── Layer 0 (background)
│   └── Rectangle { fill: #FFFFFF, w: 800, h: 600 }
├── Layer 1
│   ├── Text { content: "Summer Sale", x: 200, y: 100, fontSize: 48 }
│   └── Group
│       ├── Rectangle { fill: #FFD700, x: 250, y: 300, w: 200, h: 60 }
│       └── Text { content: "Shop Now", x: 280, y: 315 }
└── Layer 2
    └── Image { src: "hero.png", x: 50, y: 50, w: 300, h: 200 }
```

**Key properties of a scene graph:**
- **Hierarchical**: Transformations propagate parent → children (move a group = move all children)
- **Order-dependent**: Z-order for rendering is determined by tree traversal order
- **Type-polymorphic**: All nodes share a common interface (`draw()`, `getBounds()`, `hitTest()`) but vary in type (text, shape, image, group)

### 1.2 Canva's Specific Architecture

Canva's engineering reveals several architectural patterns:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Rendering | HTML Canvas API + custom engine (CanvaSX) | GPU-accelerated 2D drawing |
| State Management | MobX + React | Reactive UI updates when scene graph changes |
| Event System | RxJS | Stream-based event handling (drag, resize, click) |
| Design System | "Easel" (8px grid) | Consistent spacing, alignment, component reuse |
| Extensions | Sandboxed iframes + message bus | Third-party apps run isolated from main canvas |

**Key insight for Mini-Canva**: Canva separates the **scene graph** (pure data) from the **renderer** (presentation). The scene graph is serializable JSON; rendering is a side-effect. This exact pattern is what makes it simulatable.

### 1.3 Figma's Complementary Pattern

Figma uses a compiled C++ engine (compiled to WebAssembly) with a custom scene graph. Relevant patterns:

- **Constraint-based layout**: Elements have constraints relative to parent frames (pin to edges, stretch, center)
- **Auto-layout**: Elements within frames can flow horizontally or vertically (like CSS flexbox)
- **Variants & components**: Reusable, parameterized element templates

### 1.4 Common Abstractions Across All Canvas Tools

Regardless of tool, these abstractions are universal:

1. **Element** — Atomic visual unit with spatial + visual properties
2. **Canvas** — The container/viewport that holds elements 
3. **Selection** — Currently active element(s) for manipulation
4. **Transform** — Position, rotation, scale applied to an element
5. **Style** — Visual properties (fill, stroke, opacity, shadow)
6. **Constraints** — Relational rules between elements (alignment, spacing)

---

## 2. Mapping Real-World Architecture to the REQS.md Spec

### 2.1 Requirements Gap Analysis

| Real-World Concept | REQS.md Coverage | Mini-Canva Scope |
|---|---|---|
| Scene graph (tree) | Implicit (elements with z-index) | **Flat list with z-index** — sufficient for a simulator. No deep nesting needed |
| Element types | Text, Shape (Rect/Button), Image | **3 types** — minimal but covers the key archetypes |
| Element properties | x, y, w, h, z-index, color, content | **Core spatial + visual** — missing rotation, opacity, font-size but adequate for RL |
| Rendering | Optional visual output via PIL/Pygame | **PIL-based rasterization** — deterministic, fast, no GPU dependency |
| DOM/Accessibility tree | Required as semantic state | **JSON serialization of scene graph** — this IS the accessibility tree |
| Undo/redo | Not required | Not needed for RL episodes |
| Real-time collaboration | Not required | Not needed — single agent per env |
| Groups/nesting | Not required | Not needed — flat element list is simpler for RL |
| Constraints/auto-layout | Not required | Not needed — agent must learn layout |

### 2.2 What the Spec Gets Right (and Why)

The REQS.md is well-designed for a simulation environment because:

1. **Flat element list** — No tree traversal complexity. `O(n)` operations on element count, trivially vectorizable.
2. **Minimal property set** — Every property is a number or string. No complex cascading styles.
3. **Deterministic rendering** — PIL produces identical output for identical state. No browser quirks.
4. **JSON state** — The semantic state IS the scene graph. No lossy conversion needed.

### 2.3 What to Add Beyond the Spec (Design Recommendations)

> [!IMPORTANT]
> The spec is intentionally minimal. The following recommendations add value without adding complexity.

#### 2.3.1 Element ID System
Every element needs a stable, unique identifier. Without it, action spaces like `move_element(id, x, y)` are impossible.

**Recommendation**: Use auto-incrementing integer IDs (`element_0`, `element_1`, ...). Simple, deterministic, and serializable.

#### 2.3.2 Canvas Background Color
The spec doesn't mention canvas background. This matters for:
- WCAG contrast calculations (text against what background?)
- Visual rendering (what color is the "empty" canvas?)

**Recommendation**: Canvas has a `background_color` property, default `#FFFFFF`.

#### 2.3.3 Font Size for Text Elements
`content` alone isn't enough for text layout.  Two texts with different font sizes occupy different bounding boxes.

**Recommendation**: Add `font_size` (int, pixels) to text elements. Default 16.

#### 2.3.4 Element Bounds Validation
Should elements be allowed outside the canvas? In real Canva, yes (off-canvas elements are common). But for RL:

**Recommendation**: Elements CAN extend beyond canvas bounds, but reward functions should penalize off-canvas elements. This gives the agent the freedom to explore but guides it toward valid layouts.

---

## 3. The Mini-Canva Data Model (First Principles Design)

### 3.1 Core Data Structures

The fundamental principle: **the scene graph IS the state, and the state must be fully serializable to JSON**.

```
CanvasState:
  width: int (800)                    # Canvas dimensions in pixels
  height: int (600)
  background_color: str ("#FFFFFF")   # Hex color
  elements: List[Element]             # Flat, ordered by z-index
  next_id: int                        # Auto-incrementing ID counter

Element:
  id: str ("element_0")              # Unique, stable identifier
  type: enum (TEXT | SHAPE | IMAGE)  # Element archetype
  x: int                             # Top-left X position
  y: int                             # Top-left Y position
  width: int                         # Bounding box width
  height: int                        # Bounding box height
  z_index: int                       # Drawing order (higher = on top)
  color: str ("#000000")             # Fill color (hex)
  text_color: str ("#000000")        # Text color for TEXT/SHAPE elements
  content: str                       # Text content / label / image placeholder name
  font_size: int (16)                # Font size in pixels (TEXT elements)
```

### 3.2 Why a Flat List, Not a Tree

In production Canva, the tree structure enables:
- Group transforms (move group → move children)
- Component instances (one definition, many usages)
- Constraint propagation (child pinned to parent edge)

**None of these are needed for RL training.** The agent operates on individual elements. A flat list means:
- `O(1)` element lookup by ID (use a dict internally, expose as list)
- No tree-traversal complexity in the step function
- Trivially serializable to JSON
- Easy to convert to gymnasium observation spaces

### 3.3 Rendering Pipeline

```
CanvasState (data) → PIL Image Draw → RGB numpy array (800×600×3)
```

The rendering order:
1. Fill canvas with `background_color`
2. Sort elements by `z_index` (ascending = back to front)
3. For each element:
   - SHAPE: Draw filled rectangle with `color`, text overlay with `text_color` if content exists
   - TEXT: Draw text with `text_color` at (x, y) with `font_size`
   - IMAGE: Draw colored bounding box with `color` (since real images aren't loaded)

This is intentionally trivial. The renderer exists only for optional visual output and pixel-space observation. **The primary observation is always the JSON semantic state.**

---

## 4. Design Principle Summary

| Principle | Application |
|-----------|------------|
| **Separation of concerns** | Scene graph (data) is strictly separated from rendering (presentation) |
| **Serialization as truth** | If it can't be serialized to JSON, it doesn't exist in the state |
| **Flat over deep** | Flat element list over tree hierarchy — simpler for RL, same expressiveness for this scope |
| **Determinism** | Same state + same action = same next state. No randomness in the engine itself |
| **Minimal but complete** | Every property needed for the reward function is in the state. Nothing more |

---

## 5. Comparison Table: Real Canva vs. Mini-Canva

| Dimension | Real Canva | Mini-Canva |
|-----------|-----------|------------|
| Scene graph | Deep tree with groups, frames, components | Flat list with z-index ordering |
| Rendering | GPU-accelerated Canvas API / WebGL | CPU-based PIL Image |
| Element types | 50+ (text, shape, image, video, chart, ...) | 3 (text, shape, image placeholder) |
| Properties per element | 100+ (position, style, effects, constraints, ...) | ~10 (position, size, color, content) |
| State format | Proprietary binary + JSON APIs | Pure JSON |
| Collaboration | Real-time multi-user CRDT | Single-agent, single-instance |
| Undo/Redo | Full command history | None (RL episodes are forward-only) |
| Performance target | 60fps interactive editing | < 1ms per step (pure Python computation) |
| Use case | Human designers | RL agent training |
