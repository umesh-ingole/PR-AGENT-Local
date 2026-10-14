"""Static Analysis stage. Calls linters through MCP (default) or directly."""
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
    if os.getenv("USE_MCP", "1") == "1":
        try:
            return tool, asyncio.run(_call_mcp(tool, {"code": code, **extra}))
        except Exception as exc:  # MCP unavailable -> deterministic direct call
            name, out = tools.analyze(language, code)
            return name, f"[MCP unavailable ({type(exc).__name__}); ran directly]\n{out}"
    return tools.analyze(language, code)
