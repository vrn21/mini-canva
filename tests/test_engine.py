"""Tests for the canvas engine."""

import json

import numpy as np
import pytest

from engine import Canvas, CanvasConfig, CanvasRenderer, Element, ElementType

try:
    from PIL import Image
except ImportError:
    Image = None


# ═══════════════════════════════════════════════════════════════════════════
#  CanvasConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCanvasConfig:
    def test_default_values(self):
        config = CanvasConfig()
        assert config.width == 800
        assert config.height == 600
        assert config.background_color == "#FFFFFF"
        assert config.max_elements is None

    def test_custom_values(self):
        config = CanvasConfig(width=1920, height=1080, background_color="#000000", max_elements=50)
        assert config.width == 1920
        assert config.height == 1080
        assert config.max_elements == 50


# ═══════════════════════════════════════════════════════════════════════════
#  Element Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestElement:
    def test_bounds(self):
        el = Element(id="e0", type=ElementType.SHAPE, x=10, y=20, width=100, height=50)
        assert el.bounds == (10, 20, 110, 70)

    def test_center(self):
        el = Element(id="e0", type=ElementType.SHAPE, x=0, y=0, width=100, height=50)
        assert el.center == (50.0, 25.0)

    def test_area(self):
        el = Element(id="e0", type=ElementType.SHAPE, x=0, y=0, width=100, height=50)
        assert el.area == 5000

    def test_to_dict_is_json_serializable(self):
        el = Element(
            id="e0", type=ElementType.TEXT, x=10, y=20, width=100, height=50,
            content="Hello", color="#FF0000",
        )
        d = el.to_dict(z_index=3)
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["type"] == "TEXT"
        assert parsed["content"] == "Hello"
        assert parsed["color"] == "#FF0000"
        assert parsed["z_index"] == 3

    def test_to_dict_does_not_include_computed_properties(self):
        el = Element(id="e0", type=ElementType.SHAPE, x=10, y=20, width=100, height=50)
        d = el.to_dict()
        assert "bounds" not in d
        assert "center" not in d
        assert "area" not in d

    def test_slots_are_used(self):
        el = Element(id="e0", type=ElementType.SHAPE, x=0, y=0, width=10, height=10)
        assert hasattr(el, "__slots__")
        with pytest.raises(AttributeError):
            el.arbitrary_attr = "should fail"


# ═══════════════════════════════════════════════════════════════════════════
#  Canvas CRUD Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCanvasCRUD:
    def test_add_element(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=10, y=20, width=100, height=50, content="Hi")
        assert el is not None
        assert el.id == "element_0"
        assert el.type == ElementType.TEXT
        assert el.content == "Hi"
        assert canvas.element_count == 1

    def test_add_assigns_incrementing_ids(self):
        canvas = Canvas()
        e0 = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        e1 = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        e2 = canvas.add_element(ElementType.IMAGE, x=0, y=0, width=10, height=10)
        assert e0.id == "element_0"
        assert e1.id == "element_1"
        assert e2.id == "element_2"

    def test_add_appends_to_front(self):
        """New elements are added to the front (highest z-order)."""
        canvas = Canvas()
        e0 = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        e1 = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        elements = canvas.get_all_elements()
        assert elements[0].id == e0.id  # back
        assert elements[1].id == e1.id  # front

    def test_add_returns_none_when_full(self):
        canvas = Canvas(CanvasConfig(max_elements=2))
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        result = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        assert result is None
        assert canvas.element_count == 2

    def test_add_unlimited_when_no_max(self):
        canvas = Canvas()
        for i in range(100):
            el = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
            assert el is not None
        assert canvas.element_count == 100

    def test_add_returns_none_for_invalid_dimensions(self):
        canvas = Canvas()
        assert canvas.add_element(ElementType.TEXT, x=0, y=0, width=0, height=10) is None
        assert canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=-5) is None
        assert canvas.element_count == 0

    def test_remove_element(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        assert canvas.remove_element(el.id) is True
        assert canvas.element_count == 0

    def test_remove_nonexistent_returns_false(self):
        canvas = Canvas()
        assert canvas.remove_element("nonexistent") is False

    def test_remove_preserves_order(self):
        canvas = Canvas()
        e0 = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        e1 = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        e2 = canvas.add_element(ElementType.IMAGE, x=0, y=0, width=10, height=10)
        canvas.remove_element(e1.id)
        elements = canvas.get_all_elements()
        assert [e.id for e in elements] == [e0.id, e2.id]

    def test_remove_updates_index(self):
        canvas = Canvas()
        e0 = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        e1 = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        canvas.remove_element(e0.id)
        assert canvas.get_element(e1.id) is not None
        assert canvas.get_element(e0.id) is None

    def test_get_element(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=10, y=20, width=100, height=50)
        found = canvas.get_element(el.id)
        assert found is not None
        assert found.x == 10

    def test_get_nonexistent_returns_none(self):
        canvas = Canvas()
        assert canvas.get_element("nonexistent") is None

    def test_has_element(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        assert canvas.has_element(el.id) is True
        assert canvas.has_element("nonexistent") is False

    def test_move_element(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=10, y=20, width=100, height=50)
        assert canvas.move_element(el.id, 50, 60) is True
        assert canvas.get_element(el.id).x == 50
        assert canvas.get_element(el.id).y == 60

    def test_move_nonexistent_returns_false(self):
        canvas = Canvas()
        assert canvas.move_element("nonexistent", 0, 0) is False

    def test_move_allows_off_canvas(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        assert canvas.move_element(el.id, -100, 9999) is True
        assert canvas.get_element(el.id).x == -100

    def test_resize_element(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        assert canvas.resize_element(el.id, 200, 300) is True
        assert canvas.get_element(el.id).width == 200
        assert canvas.get_element(el.id).height == 300

    def test_resize_rejects_invalid_dimensions(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        assert canvas.resize_element(el.id, 0, 10) is False
        assert canvas.get_element(el.id).width == 10

    def test_update_element(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10, content="old")
        assert canvas.update_element(el.id, content="new", font_size=24) is True
        assert canvas.get_element(el.id).content == "new"
        assert canvas.get_element(el.id).font_size == 24

    def test_update_cannot_change_id_or_type(self):
        canvas = Canvas()
        el = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        canvas.update_element(el.id, id="hacked", type=ElementType.SHAPE)
        assert canvas.get_element(el.id).id == el.id
        assert canvas.get_element(el.id).type == ElementType.TEXT

    def test_clear(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        canvas.clear()
        assert canvas.element_count == 0
        el = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        assert el.id == "element_0"


# ═══════════════════════════════════════════════════════════════════════════
#  Canvas Z-Order Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCanvasZOrder:
    def test_reorder_element(self):
        canvas = Canvas()
        e0 = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        e1 = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        e2 = canvas.add_element(ElementType.IMAGE, x=0, y=0, width=10, height=10)
        canvas.reorder_element(e0.id, 2)
        elements = canvas.get_all_elements()
        assert [e.id for e in elements] == [e1.id, e2.id, e0.id]

    def test_bring_to_front(self):
        canvas = Canvas()
        e0 = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        e1 = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        canvas.bring_to_front(e0.id)
        elements = canvas.get_all_elements()
        assert elements[-1].id == e0.id

    def test_send_to_back(self):
        canvas = Canvas()
        e0 = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        e1 = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        canvas.send_to_back(e1.id)
        elements = canvas.get_all_elements()
        assert elements[0].id == e1.id

    def test_reorder_clamps_index(self):
        canvas = Canvas()
        e0 = canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        assert canvas.reorder_element(e0.id, 999) is True
        assert canvas.reorder_element(e0.id, -10) is True

    def test_reorder_nonexistent_returns_false(self):
        canvas = Canvas()
        assert canvas.reorder_element("nonexistent", 0) is False

    def test_z_index_derived_in_to_dict(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        canvas.add_element(ElementType.IMAGE, x=0, y=0, width=10, height=10)
        d = canvas.to_dict()
        z_indices = [e["z_index"] for e in d["elements"]]
        assert z_indices == [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════
#  Canvas Spatial Query Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCanvasQueries:
    def test_get_elements_by_type(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        texts = canvas.get_elements_by_type(ElementType.TEXT)
        assert len(texts) == 2
        assert all(e.type == ElementType.TEXT for e in texts)

    def test_get_elements_at_point(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=100)
        canvas.add_element(ElementType.SHAPE, x=50, y=50, width=100, height=100)
        hits = canvas.get_elements_at(75, 75)
        assert len(hits) == 2
        # Topmost (last in list) should be first in hits
        assert hits[0].id == "element_1"

    def test_get_elements_at_miss(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=50, height=50)
        hits = canvas.get_elements_at(100, 100)
        assert len(hits) == 0

    def test_overlapping_pairs(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=100, height=100)
        canvas.add_element(ElementType.SHAPE, x=50, y=50, width=100, height=100)
        pairs = canvas.get_overlapping_pairs()
        assert len(pairs) == 1
        assert pairs[0][2] == 50 * 50

    def test_non_overlapping_elements(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=50, height=50)
        canvas.add_element(ElementType.SHAPE, x=100, y=100, width=50, height=50)
        pairs = canvas.get_overlapping_pairs()
        assert len(pairs) == 0

    def test_get_element_behind(self):
        canvas = Canvas()
        bg = canvas.add_element(
            ElementType.SHAPE, x=0, y=0, width=800, height=600, color="#FF0000",
        )
        text = canvas.add_element(ElementType.TEXT, x=100, y=100, width=200, height=50)
        behind = canvas.get_element_behind(text.id)
        assert behind is not None
        assert behind.id == bg.id

    def test_get_element_behind_no_overlap(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=50, height=50)
        text = canvas.add_element(ElementType.TEXT, x=200, y=200, width=50, height=50)
        assert canvas.get_element_behind(text.id) is None

    def test_get_element_behind_picks_closest(self):
        """Should return the element directly behind, not the furthest back."""
        canvas = Canvas()
        far_back = canvas.add_element(ElementType.SHAPE, x=0, y=0, width=800, height=600)
        mid = canvas.add_element(ElementType.SHAPE, x=50, y=50, width=400, height=400)
        top = canvas.add_element(ElementType.TEXT, x=100, y=100, width=200, height=50)
        behind = canvas.get_element_behind(top.id)
        assert behind.id == mid.id


# ═══════════════════════════════════════════════════════════════════════════
#  Canvas Serialization Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCanvasSerialization:
    def test_to_dict_is_json_serializable(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=10, y=20, width=100, height=50, content="Hello")
        canvas.add_element(ElementType.SHAPE, x=50, y=100, width=200, height=60, color="#FFD700")
        d = canvas.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["element_count"] == 2
        assert parsed["canvas"]["width"] == 800
        assert len(parsed["elements"]) == 2

    def test_to_dict_elements_have_derived_z_index(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        d = canvas.to_dict()
        assert d["elements"][0]["z_index"] == 0
        assert d["elements"][1]["z_index"] == 1

    def test_empty_canvas_to_dict(self):
        canvas = Canvas()
        d = canvas.to_dict()
        assert d["element_count"] == 0
        assert d["elements"] == []
        assert d["canvas"]["background_color"] == "#FFFFFF"


# ═══════════════════════════════════════════════════════════════════════════
#  Canvas Numpy Export Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCanvasNumpy:
    def test_to_numpy_empty(self):
        canvas = Canvas()
        features, mask = canvas.to_numpy()
        assert features.shape == (0, 13)
        assert mask.shape == (0,)

    def test_to_numpy_shape(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=10, y=20, width=100, height=50)
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=50, height=50, color="#FF0000")
        features, mask = canvas.to_numpy()
        assert features.shape == (2, 13)
        assert features.dtype == np.float32
        assert mask.shape == (2,)
        assert mask.all()

    def test_to_numpy_values(self):
        canvas = Canvas()
        canvas.add_element(
            ElementType.SHAPE, x=10, y=20, width=100, height=50,
            color="#FF0000", text_color="#00FF00", font_size=24, content="btn",
        )
        features, _ = canvas.to_numpy()
        row = features[0]
        assert row[0] == 1  # SHAPE = 1
        assert row[1] == 10  # x
        assert row[2] == 20  # y
        assert row[3] == 100  # width
        assert row[4] == 50  # height
        assert row[5] == 255  # color_r
        assert row[6] == 0  # color_g
        assert row[7] == 0  # color_b
        assert row[8] == 0  # text_color_r
        assert row[9] == 255  # text_color_g
        assert row[10] == 0  # text_color_b
        assert row[11] == 24  # font_size
        assert row[12] == 3  # len("btn")


# ═══════════════════════════════════════════════════════════════════════════
#  Canvas Snapshot / Restore Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCanvasSnapshot:
    def test_snapshot_and_restore(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=10, y=20, width=100, height=50, content="Hello")
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=50, height=50)
        snap = canvas.snapshot()

        canvas.clear()
        assert canvas.element_count == 0

        canvas.restore(snap)
        assert canvas.element_count == 2
        assert canvas.get_element("element_0").content == "Hello"

    def test_restore_is_independent_copy(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=10, y=20, width=100, height=50, content="Hello")
        snap = canvas.snapshot()

        canvas.restore(snap)
        canvas.move_element("element_0", 999, 999)
        # Original snapshot should not be affected
        canvas.restore(snap)
        assert canvas.get_element("element_0").x == 10

    def test_restore_rebuilds_index(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        snap = canvas.snapshot()

        canvas.clear()
        canvas.restore(snap)
        assert canvas.has_element("element_0") is True
        assert canvas.get_element("element_0") is not None

    def test_snapshot_preserves_id_counter(self):
        canvas = Canvas()
        canvas.add_element(ElementType.TEXT, x=0, y=0, width=10, height=10)
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=10, height=10)
        snap = canvas.snapshot()

        canvas.clear()
        canvas.restore(snap)
        e = canvas.add_element(ElementType.IMAGE, x=0, y=0, width=10, height=10)
        assert e.id == "element_2"


# ═══════════════════════════════════════════════════════════════════════════
#  Renderer Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCanvasRenderer:
    def test_render_empty_canvas(self):
        canvas = Canvas()
        renderer = CanvasRenderer()
        img = renderer.render(canvas)
        assert img.size == (800, 600)
        assert img.mode == "RGB"
        assert img.getpixel((400, 300)) == (255, 255, 255)

    def test_render_with_shape(self):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=100, y=100, width=200, height=100, color="#FF0000")
        renderer = CanvasRenderer()
        img = renderer.render(canvas)
        assert img.getpixel((200, 150)) == (255, 0, 0)

    def test_render_to_array_shape(self):
        canvas = Canvas()
        renderer = CanvasRenderer()
        arr = renderer.render_to_array(canvas)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (600, 800, 3)
        assert arr.dtype == np.uint8

    def test_render_custom_background(self):
        canvas = Canvas(CanvasConfig(background_color="#000000"))
        renderer = CanvasRenderer()
        img = renderer.render(canvas)
        assert img.getpixel((400, 300)) == (0, 0, 0)

    def test_render_z_order(self):
        """Elements later in the list (higher z-order) render on top."""
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=0, y=0, width=200, height=200, color="#FF0000")
        canvas.add_element(ElementType.SHAPE, x=50, y=50, width=200, height=200, color="#0000FF")
        renderer = CanvasRenderer()
        img = renderer.render(canvas)
        # Point (100, 100) is covered by both, but blue (added second) is on top
        assert img.getpixel((100, 100)) == (0, 0, 255)

    def test_save_creates_file(self, tmp_path):
        canvas = Canvas()
        canvas.add_element(ElementType.SHAPE, x=10, y=10, width=50, height=50, color="#00FF00")
        renderer = CanvasRenderer()
        output_path = tmp_path / "test_output.png"
        renderer.save(canvas, output_path)
        assert output_path.exists()
        img = Image.open(output_path)
        assert img.size == (800, 600)
