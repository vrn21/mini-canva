"""Tests for the FastMCP server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

fastmcp = pytest.importorskip("fastmcp")
Client = fastmcp.Client

from server import SESSION, mcp


def _tool_data(result: Any) -> Any:
    """Extract structured data from a FastMCP tool result."""

    if hasattr(result, "data"):
        return result.data
    if hasattr(result, "structured_content"):
        return result.structured_content
    if hasattr(result, "structuredContent"):
        return result.structuredContent
    return result


def _tool_names(tools: Any) -> set[str]:
    """Extract tool names from list_tools output."""

    items = getattr(tools, "tools", tools)
    return {item.name for item in items}


def _resource_text(result: Any) -> str:
    """Extract text content from a FastMCP resource read result."""

    if isinstance(result, str):
        return result

    if hasattr(result, "contents"):
        contents = result.contents
    elif isinstance(result, (list, tuple)):
        contents = result
    else:
        contents = [result]

    first = contents[0]
    if isinstance(first, str):
        return first
    for attr in ("text", "content", "data"):
        value = getattr(first, attr, None)
        if isinstance(value, str):
            return value
    raise AssertionError(f"Unable to extract text from resource result: {result!r}")


async def _initialize(
    mcp_client,
    seed: int = 42,
    observation_mode: str = "semantic",
    action_interface: str = "semantic",
) -> dict[str, Any]:
    """Initialize the server and return the structured payload."""

    return _tool_data(
        await mcp_client.call_tool(
            "initialize_env",
            {
                "seed": seed,
                "observation_mode": observation_mode,
                "action_interface": action_interface,
            },
        )
    )


@pytest.fixture(autouse=True)
def reset_server_session() -> None:
    """Reset the singleton server session around each test."""

    if SESSION.env is not None:
        SESSION.env.close()
    SESSION.env = None
    SESSION.seed = None
    SESSION.session_id = None
    yield
    if SESSION.env is not None:
        SESSION.env.close()
    SESSION.env = None
    SESSION.seed = None
    SESSION.session_id = None


@pytest.fixture
async def mcp_client():
    """Yield an in-process FastMCP client for the server."""

    async with Client(transport=mcp) as client:
        yield client


class TestServerInitialization:
    async def test_list_tools_contains_expected_names(self, mcp_client):
        tools = await mcp_client.list_tools()
        assert _tool_names(tools) == {
            "initialize_env",
            "get_canvas_state",
            "get_observation",
            "execute_action",
            "get_current_reward",
            "save_canvas",
            "set_active_tool",
            "mouse_move",
            "mouse_click",
            "mouse_drag",
            "keyboard_type",
        }

    async def test_initialize_env_returns_prompt_and_canvas_state(self, mcp_client):
        data = await _initialize(mcp_client)

        assert data["status"] == "initialized"
        assert data["session_id"]
        assert data["prompt"]
        assert isinstance(data["prompt_id"], int)
        assert data["action_interface"] == "semantic"
        assert data["observation_mode"] == "semantic"
        assert data["observation"]["mode"] == "semantic"
        assert data["observation"]["semantic"] is not None
        assert data["observation"]["pixels"] is None
        assert data["canvas_state"]["initialized"] is True
        assert data["canvas_state"]["prompt_id"] == data["prompt_id"]
        assert data["canvas_state"]["session_id"] == data["session_id"]
        assert data["element_count"] == 0
        assert data["step_count"] == 0

    async def test_initialize_env_seed_is_deterministic(self, mcp_client):
        first = await _initialize(mcp_client, seed=7)
        second = await _initialize(mcp_client, seed=7)

        assert first["prompt"] == second["prompt"]
        assert first["prompt_id"] == second["prompt_id"]


class TestStateAccess:
    async def test_get_canvas_state_before_init_errors(self, mcp_client):
        with pytest.raises(Exception):
            await mcp_client.call_tool("get_canvas_state", {"session_id": "missing"})

    async def test_get_canvas_state_after_init_contains_prompt_id(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool("get_canvas_state", {"session_id": init["session_id"]})
        data = _tool_data(result)

        assert data["initialized"] is True
        assert isinstance(data["prompt_id"], int)
        assert data["target_prompt"]
        assert data["session_id"] == init["session_id"]

    async def test_get_observation_semantic_mode(self, mcp_client):
        init = await _initialize(mcp_client, observation_mode="semantic")
        result = await mcp_client.call_tool("get_observation", {"session_id": init["session_id"]})
        data = _tool_data(result)

        assert data["mode"] == "semantic"
        assert data["semantic"] is not None
        assert data["pixels"] is None

    async def test_get_observation_semantic_plus_pixels_mode(self, mcp_client):
        init = await _initialize(mcp_client, observation_mode="semantic+pixels")
        result = await mcp_client.call_tool("get_observation", {"session_id": init["session_id"]})
        data = _tool_data(result)

        assert data["mode"] == "semantic+pixels"
        assert data["semantic"] is not None
        assert data["pixels_shape"] == [96, 128, 3]
        assert data["pixels_dtype"] == "uint8"
        assert len(data["pixels"]) == 96
        assert len(data["pixels"][0]) == 128
        assert len(data["pixels"][0][0]) == 3

    async def test_get_observation_pixels_mode(self, mcp_client):
        init = await _initialize(mcp_client, observation_mode="pixels")
        result = await mcp_client.call_tool("get_observation", {"session_id": init["session_id"]})
        data = _tool_data(result)

        assert data["mode"] == "pixels"
        assert data["semantic"] is None
        assert data["pixels_shape"] == [96, 128, 3]
        assert data["pixels_dtype"] == "uint8"

    async def test_canvas_state_resource_before_init_returns_initialized_false(self, mcp_client):
        result = await mcp_client.read_resource("canvas://state")
        payload = json.loads(_resource_text(result))

        assert payload["initialized"] is False
        assert "initialize_env" in payload["message"]

    async def test_canvas_state_resource_after_init_returns_json_snapshot(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.read_resource("canvas://state")
        payload = json.loads(_resource_text(result))

        assert payload["initialized"] is True
        assert "canvas" in payload
        assert isinstance(payload["prompt_id"], int)
        assert payload["session_id"] == init["session_id"]

    async def test_stale_session_id_is_rejected(self, mcp_client):
        first = await _initialize(mcp_client, seed=1)
        await _initialize(mcp_client, seed=2)

        with pytest.raises(Exception):
            await mcp_client.call_tool("get_canvas_state", {"session_id": first["session_id"]})


class TestExecuteAction:
    async def test_execute_action_add_text_updates_canvas(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "execute_action",
            {
                "session_id": init["session_id"],
                "action_type": "add_text",
                "content": "Summer Sale",
            },
        )
        data = _tool_data(result)

        assert data["action_result"]["success"] is True
        assert data["observation_mode"] == "semantic"
        assert data["observation"]["mode"] == "semantic"
        assert data["canvas_state"]["element_count"] == 1
        assert data["canvas_state"]["elements"][0]["content"] == "Summer Sale"

    async def test_execute_action_returns_pixels_when_mode_enabled(self, mcp_client):
        init = await _initialize(mcp_client, observation_mode="semantic+pixels")
        result = await mcp_client.call_tool(
            "execute_action",
            {
                "session_id": init["session_id"],
                "action_type": "add_text",
                "content": "Summer Sale",
            },
        )
        data = _tool_data(result)

        assert data["observation_mode"] == "semantic+pixels"
        assert data["observation"]["semantic"] is not None
        assert data["observation"]["pixels_shape"] == [96, 128, 3]

    async def test_execute_action_add_shape_updates_canvas(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "execute_action",
            {
                "session_id": init["session_id"],
                "action_type": "add_shape",
                "content": "Shop Now",
                "color": "#FFD700",
            },
        )
        data = _tool_data(result)

        assert data["action_result"]["success"] is True
        assert data["canvas_state"]["element_count"] == 1
        assert data["canvas_state"]["elements"][0]["type"] == "SHAPE"

    async def test_execute_action_move_existing_element(self, mcp_client):
        init = await _initialize(mcp_client)
        add = _tool_data(
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "add_text",
                    "content": "Summer Sale",
                },
            )
        )
        element_id = add["action_result"]["element_id"]

        moved = _tool_data(
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "move",
                    "element_id": element_id,
                    "x": 333,
                    "y": 222,
                },
            )
        )

        assert moved["action_result"]["success"] is True
        assert moved["canvas_state"]["elements"][0]["x"] == 333
        assert moved["canvas_state"]["elements"][0]["y"] == 222

    async def test_execute_action_remove_existing_element(self, mcp_client):
        init = await _initialize(mcp_client)
        add = _tool_data(
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "add_text",
                    "content": "Summer Sale",
                },
            )
        )
        element_id = add["action_result"]["element_id"]

        removed = _tool_data(
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "remove",
                    "element_id": element_id,
                },
            )
        )

        assert removed["action_result"]["success"] is True
        assert removed["canvas_state"]["element_count"] == 0

    async def test_execute_action_done_terminates_episode(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "execute_action",
            {"session_id": init["session_id"], "action_type": "done"},
        )
        data = _tool_data(result)

        assert data["terminated"] is True
        assert data["reward_breakdown"]

    async def test_execute_action_missing_required_fields_errors(self, mcp_client):
        init = await _initialize(mcp_client)
        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "execute_action",
                {"session_id": init["session_id"], "action_type": "move", "x": 10, "y": 20},
            )

    async def test_execute_action_unknown_element_returns_structured_failure(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "execute_action",
            {
                "session_id": init["session_id"],
                "action_type": "move",
                "element_id": "missing",
                "x": 10,
                "y": 20,
            },
        )
        data = _tool_data(result)

        assert data["action_result"]["success"] is False
        assert data["observation_mode"] == "semantic"
        assert data["observation"]["mode"] == "semantic"
        assert data["canvas_state"]["element_count"] == 0

    async def test_low_level_pixels_observation_changes_after_mouse_move(self, mcp_client):
        init = await _initialize(
            mcp_client,
            observation_mode="pixels",
            action_interface="low_level",
        )
        before = _tool_data(
            await mcp_client.call_tool("get_observation", {"session_id": init["session_id"]})
        )
        moved = _tool_data(
            await mcp_client.call_tool(
                "mouse_move",
                {"session_id": init["session_id"], "x": 250, "y": 120},
            )
        )

        assert before["mode"] == "pixels"
        assert moved["observation"]["mode"] == "pixels"
        assert before["pixels_shape"] == [96, 128, 3]
        assert moved["observation"]["pixels_shape"] == [96, 128, 3]
        assert before["pixels"] != moved["observation"]["pixels"]


class TestBridgeBehavior:
    async def test_execute_action_accepts_custom_content_not_in_templates(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "execute_action",
            {
                "session_id": init["session_id"],
                "action_type": "add_text",
                "content": "Black Friday Blowout",
            },
        )
        data = _tool_data(result)

        assert data["canvas_state"]["elements"][0]["content"] == "Black Friday Blowout"

    async def test_execute_action_accepts_custom_hex_color_not_in_palette(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "execute_action",
            {
                "session_id": init["session_id"],
                "action_type": "add_shape",
                "content": "Shop Now",
                "color": "#123456",
            },
        )
        data = _tool_data(result)
        element = data["canvas_state"]["elements"][0]

        assert element["color"] == "#123456"

    async def test_recolor_text_element_updates_text_color(self, mcp_client):
        init = await _initialize(mcp_client)
        added = _tool_data(
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "add_text",
                    "content": "Summer Sale",
                },
            )
        )
        element_id = added["action_result"]["element_id"]
        recolored = _tool_data(
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "recolor",
                    "element_id": element_id,
                    "color": "#123456",
                },
            )
        )

        assert recolored["canvas_state"]["elements"][0]["text_color"] == "#123456"

    async def test_recolor_shape_keeps_contrasting_text_color(self, mcp_client):
        init = await _initialize(mcp_client)
        added = _tool_data(
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "add_shape",
                    "content": "Shop Now",
                },
            )
        )
        element_id = added["action_result"]["element_id"]
        recolored = _tool_data(
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "recolor",
                    "element_id": element_id,
                    "color": "#0000FF",
                },
            )
        )
        element = recolored["canvas_state"]["elements"][0]

        assert element["color"] == "#0000FF"
        assert element["text_color"] == "#FFFFFF"

    async def test_add_action_clamps_dimensions_to_minimum_size(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "execute_action",
            {
                "session_id": init["session_id"],
                "action_type": "add_text",
                "width": 1,
                "height": 5,
                "content": "Summer Sale",
            },
        )
        data = _tool_data(result)
        element = data["canvas_state"]["elements"][0]

        assert element["width"] >= 20
        assert element["height"] >= 20


class TestRewardAndSave:
    async def test_get_current_reward_returns_scalar_and_breakdown(self, mcp_client):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "get_current_reward",
            {"session_id": init["session_id"]},
        )
        data = _tool_data(result)

        assert isinstance(data["reward"], float)
        assert set(data["breakdown"]) == {
            "constraint",
            "aesthetics",
            "accessibility",
            "coverage",
            "efficiency",
        }

    async def test_save_canvas_writes_png(self, mcp_client, tmp_path: Path):
        init = await _initialize(mcp_client)
        await mcp_client.call_tool(
            "execute_action",
            {
                "session_id": init["session_id"],
                "action_type": "add_text",
                "content": "Summer Sale",
            },
        )
        output_path = tmp_path / "canvas.png"
        result = await mcp_client.call_tool(
            "save_canvas",
            {"session_id": init["session_id"], "filepath": str(output_path)},
        )
        data = _tool_data(result)

        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert data["status"] == "saved"

    async def test_save_canvas_returns_absolute_path(self, mcp_client, tmp_path: Path):
        init = await _initialize(mcp_client)
        result = await mcp_client.call_tool(
            "save_canvas",
            {
                "session_id": init["session_id"],
                "filepath": str(tmp_path / "nested" / "canvas.png"),
            },
        )
        data = _tool_data(result)

        assert Path(data["path"]).is_absolute()


class TestLowLevelTools:
    async def test_initialize_env_low_level_reports_interface_and_interaction(self, mcp_client):
        data = await _initialize(mcp_client, action_interface="low_level")

        assert data["action_interface"] == "low_level"
        assert data["canvas_state"]["interaction"]["active_tool"] == "select"
        assert data["observation"]["semantic"]["cursor"] == [0.0, 0.0]

    async def test_low_level_mouse_move_updates_cursor(self, mcp_client):
        init = await _initialize(mcp_client, action_interface="low_level")
        result = await mcp_client.call_tool(
            "mouse_move",
            {"session_id": init["session_id"], "x": 250, "y": 120},
        )
        data = _tool_data(result)

        assert data["action_result"]["success"] is True
        assert data["canvas_state"]["interaction"]["cursor"] == {"x": 250, "y": 120}

    async def test_low_level_click_and_type_create_and_edit_text(self, mcp_client):
        init = await _initialize(mcp_client, action_interface="low_level")
        await mcp_client.call_tool(
            "set_active_tool",
            {"session_id": init["session_id"], "tool": "text"},
        )
        created = _tool_data(
            await mcp_client.call_tool("mouse_click", {"session_id": init["session_id"]})
        )
        element_id = created["action_result"]["element_id"]

        typed = _tool_data(
            await mcp_client.call_tool(
                "keyboard_type",
                {"session_id": init["session_id"], "text": "Black Friday Blowout"},
            )
        )
        element = typed["canvas_state"]["elements"][0]

        assert created["action_result"]["success"] is True
        assert element["id"] == element_id
        assert element["content"] == "Black Friday Blowout"

    async def test_low_level_drag_in_shape_mode_creates_shape(self, mcp_client):
        init = await _initialize(mcp_client, action_interface="low_level")
        await mcp_client.call_tool(
            "set_active_tool",
            {"session_id": init["session_id"], "tool": "shape"},
        )
        dragged = _tool_data(
            await mcp_client.call_tool(
                "mouse_drag",
                {"session_id": init["session_id"], "x1": 100, "y1": 150, "x2": 280, "y2": 230},
            )
        )
        element = dragged["canvas_state"]["elements"][0]

        assert dragged["action_result"]["success"] is True
        assert element["type"] == "SHAPE"
        assert element["width"] == 180
        assert element["height"] == 80

    async def test_low_level_drag_in_select_mode_moves_element(self, mcp_client):
        init = await _initialize(mcp_client, action_interface="low_level")
        await mcp_client.call_tool(
            "set_active_tool",
            {"session_id": init["session_id"], "tool": "text"},
        )
        created = _tool_data(
            await mcp_client.call_tool("mouse_click", {"session_id": init["session_id"]})
        )
        element = created["canvas_state"]["elements"][0]
        start_x = element["x"] + 10
        start_y = element["y"] + 10
        await mcp_client.call_tool(
            "set_active_tool",
            {"session_id": init["session_id"], "tool": "select"},
        )
        moved = _tool_data(
            await mcp_client.call_tool(
                "mouse_drag",
                {
                    "session_id": init["session_id"],
                    "x1": start_x,
                    "y1": start_y,
                    "x2": start_x + 30,
                    "y2": start_y + 20,
                },
            )
        )
        moved_element = moved["canvas_state"]["elements"][0]

        assert moved["action_result"]["success"] is True
        assert moved_element["x"] == element["x"] + 30
        assert moved_element["y"] == element["y"] + 20

    async def test_low_level_tools_error_in_semantic_mode(self, mcp_client):
        init = await _initialize(mcp_client, action_interface="semantic")

        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "mouse_click",
                {"session_id": init["session_id"]},
            )

    async def test_semantic_mutation_actions_are_rejected_in_low_level_mode(self, mcp_client):
        init = await _initialize(mcp_client, action_interface="low_level")

        with pytest.raises(Exception):
            await mcp_client.call_tool(
                "execute_action",
                {
                    "session_id": init["session_id"],
                    "action_type": "add_text",
                    "content": "Summer Sale",
                },
            )

    async def test_execute_action_done_still_finishes_low_level_episode(self, mcp_client):
        init = await _initialize(mcp_client, action_interface="low_level")
        result = await mcp_client.call_tool(
            "execute_action",
            {"session_id": init["session_id"], "action_type": "done"},
        )
        data = _tool_data(result)

        assert data["terminated"] is True
        assert data["action_result"]["action"] == "done"
