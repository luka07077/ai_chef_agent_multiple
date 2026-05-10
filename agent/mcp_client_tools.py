import os
from typing import List, Tuple
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from conf import get_project_root
from utils.logger_handler import get_logger

"""
MCP client and tool loader.
Connects to two MCP servers at once:
1. [chef_core_service]: our custom Python business server (fridge, orders, nutrition, weather)
2. [local_filesystem]:  the official Node.js filesystem server (reads files from local_privacy/)

Note: files in local_privacy/ are NOT indexed into the public RAG vector store.
The agent reads them on demand via MCP's read_file / list_directory tools.
"""

logger = get_logger("ai_chef.mcp_client")


async def load_mcp_tools() -> Tuple[List[BaseTool], MultiServerMCPClient]:
    """
    Async function that connects to all MCP servers and returns a combined tool list + client handle.
    """
    project_root = get_project_root()
    custom_server_path = os.path.join(project_root, "agent", "mcp_server.py")
    privacy_dir = os.path.join(project_root, "local_privacy")

    os.makedirs(privacy_dir, exist_ok=True)

    servers_config = {
        # Server A: our custom business logic (Python)
        "chef_core_service": {
            "command": "python",
            "args": [custom_server_path],
            "transport": "stdio",
        },
        # Server B: official filesystem server (Node.js)
        # Only allows the agent to safely read files inside local_privacy/
        "local_filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", privacy_dir],
            "transport": "stdio",
        },
    }

    logger.info("Connecting to MCP servers (Python business service + filesystem service)...")

    client = MultiServerMCPClient(servers_config)
    tools = await client.get_tools()

    logger.info(f"Loaded {len(tools)} tools from MCP:")
    for tool in tools:
        logger.info(f"  - {tool.name}")

    return tools, client
