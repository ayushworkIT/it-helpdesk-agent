"""
Chainlit web UI for the RakFort IT Help Desk agent (Phase 3: RAG + MCP tools).
- Friendly error handling (Phase 1)
- Clickable, path-safe source documents (Phase 2)
- Visible tool steps, secret redaction, human approval for tickets (Phase 3)
"""
import asyncio
import logging
import re

import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from agent import build_graph, redact
from ingest import KB_DIR

logger = logging.getLogger("helpdesk")
KB_ROOT = KB_DIR.resolve()
APPROVAL_TIMEOUT_SECONDS = 300

# ---------- Build the agent graph once and share it across all chats ----------
_graph = None
_graph_lock = asyncio.Lock()


async def get_graph():
    """Lazily build the graph on first use (starts MCP discovery), then reuse it."""
    global _graph
    async with _graph_lock:
        if _graph is None:
            _graph = await build_graph()
    return _graph


# ---------- Clickable sources (Phase 2) ----------
def source_elements(reply: str) -> list[cl.Text]:
    """Turn the 'Source: a.md, b.md' line into side-panel elements - SAFELY."""
    match = re.search(r"^Source:\s*(.+)$", reply, re.MULTILINE)
    if not match:
        return []

    elements = []
    for name in (n.strip() for n in match.group(1).split(",")):
        path = (KB_ROOT / name).resolve()
        is_safe = path.parent == KB_ROOT and path.suffix == ".md" and path.is_file()
        if not is_safe:
            logger.warning("Blocked unsafe source reference from LLM: %r", name)
            continue
        elements.append(
            cl.Text(name=name, content=path.read_text(encoding="utf-8"), display="side")
        )
    return elements


# ---------- Visible tool steps (Phase 3) ----------
async def show_tool_steps(messages) -> None:
    """Show each tool call from this turn as an expandable step: name, inputs, result."""
    last_user = max(i for i, m in enumerate(messages) if m.type == "human")
    turn = messages[last_user + 1:]
    results = {m.tool_call_id: m.text for m in turn if m.type == "tool"}

    for m in turn:
        if m.type == "ai":
            for call in m.tool_calls:
                async with cl.Step(name=call["name"], type="tool") as step:
                    step.input = call["args"]
                    step.output = results.get(call["id"], "(no result)")


# ---------- Human approval (Phase 3) ----------
def format_ticket_draft(request: dict) -> str:
    """Show the ticket the assistant wants to create, field by field."""
    lines = ["**Approval needed - the assistant wants to raise this ticket:**", ""]
    for args in request["calls"]:
        lines += [
            f"- **Summary:** {args.get('summary', '')}",
            f"- **Category:** {args.get('category', '')}",
            f"- **Priority:** {args.get('priority', 'P3')}",
            f"- **Description:** {args.get('description', '')}",
        ]
    return "\n".join(lines)


async def ask_for_approval(request: dict) -> str:
    """Show Approve / Reject buttons. Anything other than Approve counts as reject."""
    response = await cl.AskActionMessage(
        content=format_ticket_draft(request),
        actions=[
            cl.Action(name="approve", payload={"decision": "approve"}, label="Approve"),
            cl.Action(name="reject", payload={"decision": "reject"}, label="Reject"),
        ],
        timeout=APPROVAL_TIMEOUT_SECONDS,
    ).send()

    if response and response.get("name") == "approve":
        return "approve"
    return "reject"   # fail closed: rejected, timed out or closed


# ---------- Chat handlers ----------
@cl.on_chat_start
async def start():
    """Runs once when a user opens a new chat."""
    try:
        await get_graph()
    except Exception:
        logger.exception("Failed to start the help desk agent (MCP server or setup)")
        await cl.Message(
            content="The help desk assistant could not start. Please raise a ticket with the IT team."
        ).send()
        return

    await cl.Message(
        content=(
            "Hi! I'm the **RakFort IT Help Desk** assistant.\n\n"
            "I can answer IT questions from RakFort's knowledge base, **check service "
            "status and connectivity**, and **raise or look up support tickets** "
            "(you approve every ticket before it is created). "
            "Click a **Source** file name to read the original document."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Runs every time the user sends a message."""
    config = {"configurable": {"thread_id": cl.context.session.id}}
    elements = []

    try:
        graph = await get_graph()
        safe_text = redact(message.content)
        result = await graph.ainvoke({"messages": [HumanMessage(content=safe_text)]}, config)

        # Paused for approval? Ask the user, then resume with their decision.
        while result.get("__interrupt__"):
            decision = await ask_for_approval(result["__interrupt__"][0].value)
            result = await graph.ainvoke(Command(resume=decision), config)

        await show_tool_steps(result["messages"])
        reply = result["messages"][-1].text
        elements = source_elements(reply)

    except RateLimitError:
        logger.exception("OpenAI rate limit or quota exceeded")
        reply = "The help desk assistant is busy right now. Please wait a minute and try again."

    except AuthenticationError:
        logger.exception("OpenAI rejected the API key - check OPENAI_API_KEY in .env")
        reply = "The assistant is temporarily unavailable. Please raise a ticket with the IT team."

    except APIConnectionError:
        logger.exception("Could not reach the OpenAI API")
        reply = "I can't reach the AI service at the moment. Please try again shortly, or raise a ticket with the IT team."

    except APIError:
        logger.exception("OpenAI API returned an error")
        reply = "Something went wrong on my side. Please try again, or raise a ticket with the IT team."

    except Exception:
        logger.exception("Unexpected error in the help desk agent")
        reply = "Something unexpected went wrong. Please try again, or raise a ticket with the IT team."

    await cl.Message(content=reply, elements=elements).send()