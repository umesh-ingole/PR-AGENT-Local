"""Static Analysis stage. Calls linters through MCP or directly."""
import asyncio
import os
import sys

import tools

TOOL_FOR_LANG = {
    "c": ("cppcheck", {"cpp": False}),
    "cpp": ("cppcheck", {"cpp": True}),
    "python": ("pylint", {}),
    "java": ("checkstyle", {}),
    "ruby": ("rubocop", {}),
}


def static_analysis_mode() -> str:
    configured = os.getenv("USE_MCP")
    if configured is None:
        enabled = sys.platform != "win32"
        source = "Windows default" if sys.platform == "win32" else "platform default"
    else:
        enabled = configured == "1"
        source = "explicit setting"
    mode = "MCP" if enabled else "direct calls"
    value = "1" if enabled else "0"
    return f"{mode} (USE_MCP={value}; {source})"


async def _call_mcp(tool: str, args: dict) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    here = os.path.dirname(os.path.abspath(__file__))
    params = StdioServerParameters(
        command=sys.executable, args=[os.path.join(here, "mcp_server.py")]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return "\n".join(c.text for c in result.content if hasattr(c, "text"))


def run_static_analysis(language: str, code: str) -> tuple[str, str]:
    """Return (tool_name, report). Falls back to direct calls if MCP fails."""
    if language not in TOOL_FOR_LANG:
        return tools.analyze(language, code)
    tool, extra = TOOL_FOR_LANG[language]
    default_use_mcp = "0" if sys.platform == "win32" else "1"
    if os.getenv("USE_MCP", default_use_mcp) == "1":
        try:
            return tool, asyncio.run(_call_mcp(tool, {"code": code, **extra}))
        except Exception as exc:  # MCP unavailable -> deterministic direct call
            name, out = tools.analyze(language, code)
            return name, f"[MCP unavailable ({type(exc).__name__}); ran directly]\n{out}"
    return tools.analyze(language, code)
