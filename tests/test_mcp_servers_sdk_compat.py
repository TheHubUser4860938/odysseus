"""SDK-compatibility coverage for the built-in stdio MCP servers.

The MCP Python SDK 2.0.0 removed `@server.list_tools()` / `@server.call_tool()`;
every built-in server under mcp_servers/ imported through those decorators and
therefore raised AttributeError during import on a 2.x install, silently
deleting memory/RAG/email/image tools from chat. CI pins mcp<2, so these tests
run green on the 1.x line as well (they assert the *behavior*, not the API).

Run under both supported SDKs to prove the compat shim:

    python -m pytest tests/test_mcp_servers_sdk_compat.py   # installed SDK
"""

import asyncio
import importlib

import pytest

SERVER_MODULES = [
    "mcp_servers.memory_server",
    "mcp_servers.rag_server",
    "mcp_servers.image_gen_server",
    "mcp_servers.email_server",
]


def _registration_mode():
    from src.mcp_server_compat import registration_mode

    return registration_mode()


@pytest.mark.parametrize("module_name", SERVER_MODULES)
def test_builtin_server_imports_and_registers_on_installed_sdk(module_name):
    """Import must succeed and register both tool handlers on THIS SDK.

    On 1.x this covers the decorators path; on 2.x it covers the constructor-
    handler path. Before the compat shim the second case died at import with
    AttributeError: 'Server' object has no attribute 'list_tools'.
    """
    module = importlib.import_module(module_name)
    assert hasattr(module, "server"), module_name
    assert callable(module.list_tools), module_name
    assert callable(module.call_tool), module_name
    # Registration bookkeeping differs per SDK: 1.x keeps a public
    # `request_handlers` dict keyed by Request CLASS (types.ListToolsRequest);
    # 2.x renamed it to `_request_handlers` keyed by method STRING
    # ("tools/list") and added get_request_handler(). Either way, a missing
    # registration = the silent-0-tools regression this file guards.
    if _registration_mode() == "sdk1-decorators":
        from mcp import types as mcp_types

        handlers = module.server.request_handlers
        assert mcp_types.ListToolsRequest in handlers, (
            f"{module_name}: tools/list not registered"
        )
        assert mcp_types.CallToolRequest in handlers, (
            f"{module_name}: tools/call not registered"
        )
    else:
        handlers = module.server._request_handlers
        assert "tools/list" in handlers, f"{module_name}: tools/list not registered"
        assert "tools/call" in handlers, f"{module_name}: tools/call not registered"


def test_memory_server_tool_round_trip_via_registered_server():
    """A tools/list + tools/call exchange through whichever registration is live."""
    module = importlib.import_module("mcp_servers.memory_server")

    handlers = getattr(module.server, "request_handlers", None)
    if handlers is None:
        handlers = module.server._request_handlers

    mode = _registration_mode()
    if mode == "sdk1-decorators":
        from mcp import types as mcp_types

        list_handler = handlers[mcp_types.ListToolsRequest]
        call_handler = handlers[mcp_types.CallToolRequest]
    else:
        list_entry = module.server._request_handlers["tools/list"]
        call_entry = module.server._request_handlers["tools/call"]
        list_handler = list_entry.handler if hasattr(list_entry, "handler") else list_entry
        call_handler = call_entry.handler if hasattr(call_entry, "handler") else call_entry

    async def exchange():
        if mode == "sdk1-decorators":
            # 1.x wrapped handlers take a Request object and return a
            # ServerResult envelope.
            from mcp import types as mcp_types

            tools = await list_handler(
                mcp_types.ListToolsRequest(method="tools/list")
            )
            result = await call_handler(
                mcp_types.CallToolRequest(
                    method="tools/call",
                    params=mcp_types.CallToolRequestParams(
                        name="manage_memory", arguments={"action": "list"}
                    ),
                )
            )
        else:
            # 2.x entries are adapters taking (ctx, validated params).
            from mcp import types

            tools = await list_handler(ctx=None, params=types.PaginatedRequestParams())
            result = await call_handler(
                ctx=None,
                params=types.CallToolRequestParams(name="manage_memory", arguments={"action": "list"}),
            )
        return tools, result

    tools_result, call_result = asyncio.run(exchange())
    # 1.x wraps payloads in a ServerResult RootModel (.root); 2.x adapters
    # return the ListToolsResult / CallToolResult directly.
    tools_result = getattr(tools_result, "root", None) or getattr(tools_result, "result", tools_result)
    call_result = getattr(call_result, "root", None) or getattr(call_result, "result", call_result)
    tool_names = [t.name for t in tools_result.tools]
    assert tool_names == ["manage_memory"]
    # 1.x decorators return the handler's list[TextContent]; 2.x adapters return
    # a CallToolResult envelope. Read the text the same way on either.
    content = call_result.content if hasattr(call_result, "content") else call_result
    text = content[0].text
    assert text  # list succeeds or returns a scoped-owner error; never empty
    if not text.startswith("Error:"):
        assert not (hasattr(call_result, "is_error") and call_result.is_error)


def test_client_schema_reader_survives_alias_rename():
    """`tool.inputSchema if hasattr(tool, "inputSchema") else {}` silently yields
    {} on mcp 2.x (attribute renamed to input_schema; the wire alias raises).
    tool_input_schema() must return the real schema on either SDK."""
    from mcp import types
    from src.mcp_server_compat import result_is_error, tool_input_schema

    tool = types.Tool(
        name="manage_memory",
        description="Manage the user's memory system.",
        inputSchema={
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
        },
    )
    schema = tool_input_schema(tool)
    assert schema.get("type") == "object"
    assert "properties" in schema, (
        "schema stripped: the compat reader must see through the 2.x attribute rename"
    )

    ok = types.CallToolResult(content=[types.TextContent(type="text", text="fine")])
    failed = types.CallToolResult(
        content=[types.TextContent(type="text", text="boom")], isError=True
    )
    assert result_is_error(ok) is False
    assert result_is_error(failed) is True
