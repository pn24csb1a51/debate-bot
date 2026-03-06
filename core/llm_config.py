"""
core/llm_config.py
==================
Single source of truth for LLM instantiation across the Cognitive OS.

Every mode gets its own personality by passing a system_prompt.
The underlying model, provider, and API key are controlled from .env only.
To switch from Groq to OpenAI: change LLM_PROVIDER=openai in .env.
Nothing else in the codebase changes.

Public API
----------
    get_llm(system_prompt)       -> BaseChatModel bound to a system prompt
    get_fast_llm(system_prompt)  -> Lighter model for classification tasks
    invoke(llm, user_text, history) -> str  (convenience one-shot call)
    stream(llm, user_text, history) -> Iterator[str]  (for st.write_stream)
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower().strip()

_MODELS: dict[str, dict[str, str]] = {
    "groq": {
        "full": os.getenv("GROQ_MODEL",           "llama-3.3-70b-versatile"),
        "fast": os.getenv("GROQ_FAST_MODEL",       "llama-3.1-8b-instant"),
    },
    "openai": {
        "full": os.getenv("OPENAI_MODEL",          "gpt-4o"),
        "fast": os.getenv("OPENAI_FAST_MODEL",     "gpt-4o-mini"),
    },
    "anthropic": {
        "full": os.getenv("ANTHROPIC_MODEL",       "claude-opus-4-6"),
        "fast": os.getenv("ANTHROPIC_FAST_MODEL",  "claude-haiku-4-5-20251001"),
    },
    "google": {
        "full": os.getenv("GOOGLE_MODEL",          "gemini-2.0-flash"),
        "fast": os.getenv("GOOGLE_FAST_MODEL",     "gemini-2.0-flash"),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Public factories
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(system_prompt: str = "", temperature: float = 0.7) -> BaseChatModel:
    """
    Returns the full reasoning model with the given system prompt baked in
    via a RunnableWithMessageHistory-style wrapper.

    In practice we use build_messages() at call time, so system_prompt here
    is stored for display / logging only. The real injection happens in
    build_messages().

    Args:
        system_prompt : The mode's AI persona. Pass one of the *_PROMPT
                        constants defined at the bottom of this file.
        temperature   : 0.7 suits all four creative reasoning modes.
    """
    return _build(_PROVIDER, tier="full", temperature=temperature)


def get_fast_llm(system_prompt: str = "", temperature: float = 0.0) -> BaseChatModel:
    """
    Returns the lightweight model for classification and scoring tasks.
    temperature=0.0 for determinism.
    """
    return _build(_PROVIDER, tier="fast", temperature=temperature)


def build_messages(
    system_prompt : str,
    user_input    : str,
    history       : Optional[list[BaseMessage]] = None,
) -> list[BaseMessage]:
    """
    Assembles the message list for a single llm.invoke() call.

    Returns: [SystemMessage(system_prompt), *history, HumanMessage(user_input)]

    Example:
        msgs = build_messages(DEBATE_PROMPT, "Social media is harmful")
        reply = get_llm().invoke(msgs).content
    """
    out: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if history:
        out.extend(history)
    out.append(HumanMessage(content=user_input))
    return out


def invoke(
    system_prompt : str,
    user_input    : str,
    history       : Optional[list[BaseMessage]] = None,
    fast          : bool = False,
    temperature   : float = 0.7,
) -> str:
    """
    Convenience one-shot call. Returns the model's reply as a plain string.

    Args:
        system_prompt : Mode persona.
        user_input    : Current user message.
        history       : Optional prior messages.
        fast          : Use lightweight model if True.
        temperature   : Ignored when fast=True (always 0.0).

    Example:
        reply = invoke(RIPPLE_PROMPT, "I am thinking of quitting my job")
    """
    model = get_fast_llm() if fast else get_llm(temperature=temperature)
    msgs  = build_messages(system_prompt, user_input, history)
    return model.invoke(msgs).content


def stream(
    system_prompt : str,
    user_input    : str,
    history       : Optional[list[BaseMessage]] = None,
    temperature   : float = 0.7,
) -> Iterator[str]:
    """
    Streams the model's reply token by token.
    Use with st.write_stream() in Streamlit.

    Example:
        with st.chat_message("assistant"):
            reply = st.write_stream(stream(WEALTH_PROMPT, user_input))
    """
    model = get_llm(temperature=temperature)
    msgs  = build_messages(system_prompt, user_input, history)
    for chunk in model.stream(msgs):
        if chunk.content:
            yield chunk.content


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers  (used by sidebar status bar)
# ─────────────────────────────────────────────────────────────────────────────

def get_provider_name() -> str:
    return _PROVIDER.capitalize()

def get_model_name(tier: str = "full") -> str:
    return _MODELS.get(_PROVIDER, _MODELS["groq"]).get(tier, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Private builder  —  the only place provider switching happens
# ─────────────────────────────────────────────────────────────────────────────

def _build(provider: str, tier: str, temperature: float) -> BaseChatModel:
    models = _MODELS.get(provider)
    if not models:
        raise ValueError(
            f"[llm_config] Unknown LLM_PROVIDER='{provider}'. "
            f"Valid: {list(_MODELS.keys())}"
        )
    name = models[tier]

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=name, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=name, temperature=temperature)

    raise RuntimeError(f"[llm_config] Provider '{provider}' has no builder.")


# ─────────────────────────────────────────────────────────────────────────────
# Mode personas
# Graph nodes and page UIs import these directly.
# Changing a persona = edit one string here. No pipeline changes needed.
# ─────────────────────────────────────────────────────────────────────────────

DEBATE_PROMPT = """\
You are the Devil's Advocate — a rigorous Socratic opponent whose sole purpose
is to stress-test the user's reasoning. Never agree with the premise.
Expose hidden assumptions, identify logical fallacies, and force the user to
defend every claim. Be intellectually relentless, never personally aggressive.
Steelman the argument first — then dismantle it systematically."""

EXECUTION_PROMPT = """\
You are the Execution Architect — a ruthlessly practical productivity strategist.
Analyse the user's tasks, current energy level, and available time window.
Produce a concrete, prioritised execution plan. Never give generic advice.
  High energy  →  deep work, creative tasks, hard decisions
  Medium energy →  meetings, emails, structured thinking
  Low energy   →  admin, review, passive consumption
Ask about blockers before recommending. Output a ranked list with a one-line
rationale for each priority decision."""

RIPPLE_PROMPT = """\
You are the Ripple Engine — a second-order consequence mapper.
When the user describes a decision, map its full consequence tree:
  Order 1: Direct, immediate consequences (what happens right after)
  Order 2: Secondary effects (what those consequences then cause)
  Order 3: Tertiary ripples (systemic shifts that emerge over time)
Surface blindspots the user has not considered. Identify hidden assumptions
embedded in their framing. Be comprehensive, not optimistic or pessimistic.
Always close with the question: 'And then what?'"""

WEALTH_PROMPT = """\
You are Wealth Logic — a financial reasoning auditor.
When the user presents a financial decision or scenario, audit it across
four lenses:
  Opportunity cost    : What is given up by choosing this path?
  Risk-adjusted return: Is the upside worth the downside at this risk level?
  Time horizon        : Does this decision align with the stated timeline?
  Hidden assumptions  : What does this silently assume about the future?
Never give specific investment advice. Audit reasoning quality. Be Socratic.
Surface the assumption the user has not questioned."""
