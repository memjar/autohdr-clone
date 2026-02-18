"""
Agent Loop — Multi-turn tool-calling for Klaus via Ollama.
Qwen3 32B natively supports tool calling through Ollama's /api/chat endpoint.
"""

import json
import re
import time
import logging
import httpx

from config import OLLAMA_BASE, IMI_MODEL, OLLAMA_CHAT_TIMEOUT, OLLAMA_CONNECT_TIMEOUT
from tool_registry import get_tools, execute_tool

logger = logging.getLogger("agent_loop")

MAX_ITERATIONS = 5
TOOL_TIMEOUT = 30  # seconds per tool execution


async def agent_loop(
    messages: list[dict],
    system_prompt: str,
    model: str = None,
    max_iterations: int = MAX_ITERATIONS,
    tools_enabled: bool = True,
    on_tool_use: callable = None,  # callback: (tool_name, args) -> None, for streaming status to frontend
) -> dict:
    """
    Run the agent loop.

    Args:
        messages: conversation history [{"role": "user"|"assistant"|"tool", "content": "..."}]
        system_prompt: system prompt for the agent
        model: Ollama model to use (defaults to IMI_MODEL from config)
        max_iterations: max tool-calling rounds
        tools_enabled: whether to include tool definitions
        on_tool_use: optional callback when a tool is invoked (for streaming UI updates)

    Returns:
        {
            "content": "final text response",
            "tool_calls_made": [{"tool": "name", "args": {...}, "result": {...}, "duration_ms": int}],
            "iterations": int,
            "total_duration_ms": int
        }
    """
    model = model or IMI_MODEL
    tools = get_tools() if tools_enabled else None
    tool_calls_made = []
    t_start = time.time()

    # Build full message list with system prompt
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(OLLAMA_CHAT_TIMEOUT, connect=OLLAMA_CONNECT_TIMEOUT)
    ) as client:
        for iteration in range(max_iterations):
            # Call Ollama with tools
            payload = {
                "model": model,
                "messages": full_messages,
                "stream": False,
                "keep_alive": "24h",
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 32768,
                    "num_predict": 2048,
                },
            }
            if tools:
                payload["tools"] = tools

            logger.info(f"Agent loop iteration {iteration + 1}/{max_iterations}")

            try:
                resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Ollama call failed: {e}")
                return {
                    "content": f"I encountered an error communicating with the AI model: {e}",
                    "tool_calls_made": tool_calls_made,
                    "iterations": iteration + 1,
                    "total_duration_ms": int((time.time() - t_start) * 1000),
                }

            msg = data.get("message", {})
            tool_calls = msg.get("tool_calls", [])

            # If no tool calls, we have our final answer
            if not tool_calls:
                content = msg.get("content", "")
                # Strip thinking tags if present (Qwen3 thinking mode)
                if "<think>" in content:
                    content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    ).strip()

                return {
                    "content": content,
                    "tool_calls_made": tool_calls_made,
                    "iterations": iteration + 1,
                    "total_duration_ms": int((time.time() - t_start) * 1000),
                }

            # Execute each tool call
            # Append the assistant message (with tool_calls) to conversation
            full_messages.append(msg)

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})

                logger.info(
                    f"Executing tool: {tool_name} with args: {json.dumps(tool_args)[:200]}"
                )

                # Notify frontend if callback provided
                if on_tool_use:
                    try:
                        on_tool_use(tool_name, tool_args)
                    except Exception:
                        pass

                t_tool = time.time()
                result = execute_tool(tool_name, tool_args)
                duration_ms = int((time.time() - t_tool) * 1000)

                tool_calls_made.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                        "duration_ms": duration_ms,
                    }
                )

                logger.info(f"Tool {tool_name} completed in {duration_ms}ms")

                # Add tool result to conversation
                # Truncate large results to avoid blowing context
                result_str = json.dumps(result)
                if len(result_str) > 8000:
                    result_str = result_str[:8000] + "... [truncated]"

                full_messages.append({"role": "tool", "content": result_str})

        # Max iterations reached — return whatever we have
        # Make one final call without tools to get a summary
        payload_final = {
            "model": model,
            "messages": full_messages
            + [
                {
                    "role": "user",
                    "content": "Please provide your final answer based on all the tool results above.",
                }
            ],
            "stream": False,
            "keep_alive": "24h",
            "options": {"temperature": 0.3, "num_ctx": 32768, "num_predict": 2048},
        }

        try:
            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload_final)
            resp.raise_for_status()
            content = (
                resp.json()
                .get("message", {})
                .get("content", "Max tool iterations reached.")
            )
        except Exception:
            content = "I used several tools but ran out of processing steps. Here's what I found so far."

        return {
            "content": content,
            "tool_calls_made": tool_calls_made,
            "iterations": max_iterations,
            "total_duration_ms": int((time.time() - t_start) * 1000),
        }


async def agent_loop_streaming(
    messages: list[dict],
    system_prompt: str,
    model: str = None,
    max_iterations: int = MAX_ITERATIONS,
    tools_enabled: bool = True,
):
    """
    Generator version of agent_loop that yields NDJSON events for streaming to frontend.

    Yields events:
    - {"type": "tool_start", "tool": "name", "args": {...}}
    - {"type": "tool_result", "tool": "name", "duration_ms": int, "preview": "..."}
    - {"type": "thinking", "content": "..."}  (from Qwen3 <think> blocks)
    - {"type": "token", "content": "text chunk"}  (final streamed response)
    - {"type": "done", "tool_calls_made": [...], "iterations": int, "total_duration_ms": int}
    """
    model = model or IMI_MODEL
    tools = get_tools() if tools_enabled else None
    tool_calls_made = []
    t_start = time.time()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(OLLAMA_CHAT_TIMEOUT, connect=OLLAMA_CONNECT_TIMEOUT)
    ) as client:
        for iteration in range(max_iterations):
            # Non-streaming call for tool-calling iterations
            payload = {
                "model": model,
                "messages": full_messages,
                "stream": False,
                "keep_alive": "24h",
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 32768,
                    "num_predict": 2048,
                },
            }
            if tools:
                payload["tools"] = tools

            try:
                resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                yield {"type": "token", "content": f"Error: {e}"}
                yield {
                    "type": "done",
                    "tool_calls_made": tool_calls_made,
                    "iterations": iteration + 1,
                    "total_duration_ms": int((time.time() - t_start) * 1000),
                }
                return

            msg = data.get("message", {})
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                # Final answer — stream it token by token
                content = msg.get("content", "")
                # Handle thinking tags
                thinking_match = re.search(
                    r"<think>(.*?)</think>", content, re.DOTALL
                )
                if thinking_match:
                    yield {
                        "type": "thinking",
                        "content": thinking_match.group(1).strip(),
                    }
                    content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    ).strip()

                # Yield the full content (already complete since we used stream=False for tool calls)
                yield {"type": "token", "content": content}
                yield {
                    "type": "done",
                    "tool_calls_made": tool_calls_made,
                    "iterations": iteration + 1,
                    "total_duration_ms": int((time.time() - t_start) * 1000),
                }
                return

            # Execute tools
            full_messages.append(msg)

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})

                yield {"type": "tool_start", "tool": tool_name, "args": tool_args}

                t_tool = time.time()
                result = execute_tool(tool_name, tool_args)
                duration_ms = int((time.time() - t_tool) * 1000)

                tool_calls_made.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                        "duration_ms": duration_ms,
                    }
                )

                # Preview of result for frontend
                preview = json.dumps(
                    result.get("result", result.get("error", ""))
                )[:200]
                yield {
                    "type": "tool_result",
                    "tool": tool_name,
                    "duration_ms": duration_ms,
                    "preview": preview,
                }

                result_str = json.dumps(result)
                if len(result_str) > 8000:
                    result_str = result_str[:8000] + "... [truncated]"

                full_messages.append({"role": "tool", "content": result_str})

        # Max iterations — final summary
        yield {
            "type": "token",
            "content": "I've completed my analysis using multiple tools. Here's a summary of what I found.",
        }
        yield {
            "type": "done",
            "tool_calls_made": tool_calls_made,
            "iterations": max_iterations,
            "total_duration_ms": int((time.time() - t_start) * 1000),
        }
