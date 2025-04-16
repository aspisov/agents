import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mem0 import Memory
from utils import get_mem0_client

load_dotenv()

DEFAULT_USER_ID = "user"


@dataclass
class Mem0Context:
    """Context for the mem0 MCP server."""

    mem0_client: Memory


@asynccontextmanager
async def mem0_lifespan(server: FastMCP) -> AsyncIterator[Mem0Context]:
    mem0_client = get_mem0_client()

    try:
        yield Mem0Context(mem0_client=mem0_client)
    finally:
        pass


mcp = FastMCP(
    "mcp-mem0",
    description="MCP server for long term memory storage and retrieval with Mem0",
    lifespan=mem0_lifespan,
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=os.getenv("MCP_PORT", 8050),
)


@mcp.tool()
async def save_memory(ctx: Context, text: str):
    """
    Save information to the long term memory.

    Args:
        ctx: The context of the MCP server.
        text: The information to save.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        messages = [{"role": "user", "content": text}]
        mem0_client.add(messages, user_id=DEFAULT_USER_ID)
        return (
            f"Successfully saved memory: {text[:100]}..."
            if len(text) > 100
            else f"Successfully saved memory: {text}"
        )
    except Exception as e:
        return f"Failed to save memory: {e}"


@mcp.tool()
async def get_all_memories(ctx: Context) -> str:
    """Get all stored memories for the user.

    Args:
        ctx: The context of the MCP server.

    Returns:
        A string containing all the stored memories.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.get_all(user_id=DEFAULT_USER_ID)
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories
        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Failed to get all memories: {e}"


@mcp.tool()
async def search_memories(ctx: Context, query: str, limit: int = 3) -> str:
    """ "Search memories with semantic search.

    Args:
        ctx: The context of the MCP server.
        query: The query to search for.
        limit: The maximum number of memories to return.

    Returns:
        A string containing the search results.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.search(query, user_id=DEFAULT_USER_ID, limit=limit)
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories
        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Failed to search memories: {e}"


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == "sse":
        await mcp.run_sse_async()
    else:
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
