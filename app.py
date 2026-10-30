"""
Step 1.14 - Chainlit web UI for the RakFort IT Help Desk agent, with error handling.
Users see friendly messages; technical details go to the terminal log.
"""
import logging

import chainlit as cl
from langchain_core.messages import HumanMessage
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from agent import graph   # the LangGraph agent from agent.py

logger = logging.getLogger("helpdesk")


@cl.on_chat_start
async def start():
    """Runs once when a user opens a new chat."""
    await cl.Message(
        content=(
            "Hi! I'm the **RakFort IT Help Desk** assistant.\n\n"
            "Describe your IT problem (Wi-Fi, passwords, email, printers, VPN, software) "
            "and I'll walk you through fixing it."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Runs every time the user sends a message."""
    config = {"configurable": {"thread_id": cl.context.session.id}}

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=message.content)]},
            config,
        )
        reply = result["messages"][-1].text

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

    await cl.Message(content=reply).send()