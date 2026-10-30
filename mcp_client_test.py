"""
Steps 3.3 & 3.5 - Test the MCP server through the real protocol:
discover tools, call one, and read a resource.
"""
import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_PATH = Path(__file__).parent / "helpdesk_server.py"


async def main():
    client = MultiServerMCPClient(
        {
            "helpdesk": {
                "command": sys.executable,        # the venv's Python, not "python"
                "args": [str(SERVER_PATH)],
                "transport": "stdio",
            }
        }
    )

    # 1. Tools (model-controlled): discover them
    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tool(s): {[t.name for t in tools]}")

    # 2. Call one tool through MCP
    status_tool = next(t for t in tools if t.name == "check_service_status")
    result = await status_tool.ainvoke({"service": "shared drives"})
    print(f"\ncheck_service_status('shared drives') -> {result}")

    # 3. Resources (application-controlled): read the catalogue
    blobs = await client.get_resources("helpdesk", uris=["helpdesk://catalogue"])
    print("\nResource helpdesk://catalogue ->")
    print(blobs[0].as_string())


if __name__ == "__main__":
    asyncio.run(main())