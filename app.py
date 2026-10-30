"""
Chainlit web UI for the RakFort IT Help Desk agent (Phase 2: RAG).
- Friendly error handling (Phase 1)
- Clickable source documents in a side panel (Phase 2)
"""
import logging
import re

import chainlit as cl
from langchain_core.messages import HumanMessage
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from agent import graph       # the LangGraph RAG agent
from ingest import KB_DIR     # the knowledge_base folder

logger = logging.getLogger("helpdesk")
KB_ROOT = KB_DIR.resolve()    # full, absolute path of knowledge_base


def source_elements(reply: str) -> list[cl.Text]:
    """Turn the 'Source: a.md, b.md' line into side-panel elements - SAFELY.

    The file names come from the LLM, so they are untrusted input.
    Only real .md files directly inside knowledge_base are allowed (blocks ../.env etc.).
    """
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


@cl.on_chat_start
async def start():
    """Runs once when a user opens a new chat."""
    await cl.Message(
        content=(
            "Hi! I'm the **RakFort IT Help Desk** assistant.\n\n"
            "Ask me about Wi-Fi, passwords, email, printers, VPN or software. "
            "I answer from RakFort's IT knowledge base - click a **Source** file name "
            "to read the original document."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Runs every time the user sends a message."""
    config = {"configurable": {"thread_id": cl.context.session.id}}
    elements = []

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=message.content)]},
            config,
        )
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