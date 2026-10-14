"""MCP server that exposes the linters as tools.

Run standalone to test:   python mcp_server.py
The app launches it automatically over stdio when USE_MCP=1.
Adding a language later = add one @mcp.tool() here; agent prompts stay unchanged.
"""
from mcp.server.fastmcp import FastMCP

import tools

mcp = FastMCP("static-analysis")


@mcp.tool()
def cppcheck(code: str, cpp: bool = False) -> str:
    """Run cppcheck on C or C++ source code."""
    return tools.run_cppcheck(code, cpp)


@mcp.tool()
def pylint(code: str) -> str:
    """Run pylint on Python source code."""
    return tools.run_pylint(code)


@mcp.tool()
def checkstyle(code: str) -> str:
    """Run checkstyle (Google style) on Java source code."""
    return tools.run_checkstyle(code)


@mcp.tool()
def rubocop(code: str) -> str:
    """Run rubocop on Ruby source code."""
    return tools.run_rubocop(code)


if __name__ == "__main__":
    mcp.run()
