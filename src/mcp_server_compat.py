"""Version-agnostic server registration for the built-in stdio MCP servers.

The MCP Python SDK removed the deprecated ``@server.list_tools()`` /
``@server.call_tool()`` decorators in 2.0.0. On a 2.x install every built-in
server under ``mcp_servers/`` therefore fails at import with
``AttributeError: 'Server' object has no attribute 'list_tools'`` and chat
loses all tools from the memory / RAG / email / image servers. The project's
requirements still pin ``mcp<2`` — CI and Docker keep the 1.x line — but 2.x
is released and lands in unpinned installs regardless, so the servers have to
register on whichever SDK is importable.

Each server keeps exactly one pair of business functions, in the pre-2.0
shape that the existing unit tests already import directly:

    async def list_tools() -> list[types.Tool]
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]

``make_server()`` registers those on whichever SDK is present. The 2.x path
adapts (constructor handlers arrive as ``handler(ctx, params)`` and must
return result envelopes); the 1.x path delegates to the decorators, which
already accept exactly these signatures.

Schema construction needs no adapter: ``Tool(inputSchema=...)`` and
``TextContent(type="text", ...)`` are native on 1.x and accepted as wire
aliases on 2.x (populate-by-name is enabled), and both serialize to the same
wire JSON. ``Tool(input_schema=...)``, by contrast, is a hard
``ValidationError`` on 1.x — the wire-style names are the compatible choice.

Detection inspects ``Server.__init__``'s signature rather than version
metadata: 1.x exposes no ``on_*`` handler kwargs, 2.x does, which holds
across both lines' point releases and needs no ``importlib.metadata``.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from mcp import types
from mcp.server import Server

ListToolsFn = Callable[[], Awaitable[list[types.Tool]]]
CallToolFn = Callable[[str, dict[str, Any]], Awaitable[list[types.TextContent]]]


def _supports_ctor_handlers() -> bool:
    """True on mcp>=2.0 (``Server`` takes ``on_list_tools``/``on_call_tool``)."""
    try:
        params = inspect.signature(Server.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic interpreter
        return False
    return "on_list_tools" in params or "on_call_tool" in params


def registration_mode() -> str:
    """Which registration path this SDK takes; asserted by the compat tests."""
    return "sdk2-ctor" if _supports_ctor_handlers() else "sdk1-decorators"


def tool_input_schema(tool: Any) -> dict[str, Any]:
    """Read a listed tool's input schema on either SDK.

    1.x exposes the field as the ``inputSchema`` attribute; 2.x renames the
    Python attribute to ``input_schema`` and keeps ``inputSchema`` as a
    wire-only alias that *raises* on attribute access (pydantic ``__getattr__``).
    Field-name readers like ``tool.inputSchema if hasattr(tool, "inputSchema")
    else {}`` therefore silently yield ``{}`` on 2.x — every tool's parameters
    vanish from the agent prompt even from healthy servers (the NPX/browser
    servers included, since this is the client side).
    """
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        schema = getattr(tool, "input_schema", None)
    return dict(schema) if schema else {}


def result_is_error(result: Any) -> bool:
    """Read ``isError`` off a CallToolResult on either SDK (2.x renamed it to
    ``is_error``; ``getattr(result, "isError", False)`` silently means
    "success" on every failed 2.x call)."""
    value = getattr(result, "isError", None)
    if value is None:
        value = getattr(result, "is_error", None)
    return bool(value)


def make_server(
    name: str,
    list_tools: ListToolsFn,
    call_tool: CallToolFn,
) -> Server:
    """Build ``Server(name)`` wired to the server's business functions.

    ``list_tools``/``call_tool`` keep the pre-2.0 signatures described in the
    module docstring; unit tests importing the server modules call them
    directly and must keep working unmodified.
    """
    if not _supports_ctor_handlers():
        server = Server(name)
        # 1.x decorators already accept `() -> list[Tool]` and
        # `(name, arguments) -> list[TextContent]`, coercing the list into a
        # CallToolResult themselves; pass the business functions straight in.
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
        return server

    async def _on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=await list_tools())

    async def _on_call_tool(ctx: Any, params: Any) -> types.CallToolResult:
        try:
            content = await call_tool(params.name, dict(params.arguments or {}))
        except Exception as exc:  # mirror the 1.x decorator: tool failures
            # travel as isError results, not JSON-RPC faults.
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: {exc}")],
                isError=True,
            )
        return types.CallToolResult(content=content)

    return Server(name, on_list_tools=_on_list_tools, on_call_tool=_on_call_tool)
