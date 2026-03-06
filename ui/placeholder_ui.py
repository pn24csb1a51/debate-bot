"""
pages/placeholder_ui.py
========================
Self-contained template page used by all four modes.

HOW TO USE AS A TEMPLATE
--------------------------
Each mode file (debate_ui.py, execution_ui.py, etc.) calls render() from
here and passes its own MODE_CONFIG dict. No copy-pasting needed — the
template is reused, not duplicated.

The only thing a mode file needs to do:
    1.  Define its MODE_CONFIG dict.
    2.  Call render(MODE_CONFIG) in its own render() function.
    3.  When ready to wire a real graph, replace _placeholder_reply()
        with a graph invocation in that file.

All shared state (memory_manager, llm) lives in st.session_state.
Page files never instantiate those objects themselves.
"""
from __future__ import annotations

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Mode config schema  (document the expected keys)
# ─────────────────────────────────────────────────────────────────────────────
#
#  mode_key     str  — matches MemoryManager collection key
#  icon         str  — single emoji
#  title        str  — display name
#  tagline      str  — one-line descriptor shown under the title
#  description  str  — 2-3 sentence explanation shown in the header banner
#  accent       str  — CSS hex colour  ("#e63946" etc.)
#  input_hint   str  — chat input placeholder text
#  memory_label str  — noun used in the memory count ("debate exchanges" etc.)
#  system_prompt str — imported from core.llm_config, injected into LLM calls


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render(cfg: dict) -> None:
    """
    Renders the full page for any mode by consuming its config dict.

    Called by each mode's own render() wrapper:
        from ui.placeholder_ui import render as _render_template
        def render():
            _render_template(MODE_CONFIG)
    """
    mm = st.session_state.get("memory_manager")

    _render_header(cfg)

    if mm:
        _render_memory_bar(cfg, mm)

    st.divider()
    _render_chat(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

def _render_header(cfg: dict) -> None:
    """
    Full-width mode banner: large title, tagline, description.
    Accent colour is injected from cfg so each mode looks distinct.
    """
    accent = cfg["accent"]
    st.markdown(
        f"""
        <div style="
            border-left   : 3px solid {accent};
            padding       : 0.9rem 1.4rem;
            margin-bottom : 1.4rem;
            background    : rgba(255,255,255,0.015);
            border-radius : 0 6px 6px 0;
        ">
            <div style="
                font-family   : 'IBM Plex Mono', monospace;
                font-size     : 1.5rem;
                font-weight   : 700;
                color         : {accent};
                letter-spacing: 0.02em;
                margin-bottom : 0.2rem;
            ">{cfg["icon"]}  {cfg["title"].upper()}</div>

            <div style="
                font-family   : 'IBM Plex Mono', monospace;
                font-size     : 0.65rem;
                color         : #555;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                margin-bottom : 0.75rem;
            ">{cfg["tagline"]}</div>

            <div style="
                font-size   : 0.88rem;
                color       : #999;
                line-height : 1.72;
                max-width   : 760px;
            ">{cfg["description"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memory bar
# ─────────────────────────────────────────────────────────────────────────────

def _render_memory_bar(cfg: dict, mm) -> None:
    """
    Shows the live memory count for this mode and a scoped Reset button.

    ISOLATION GUARANTEE:
    The Reset button calls mm.clear_collection(cfg["mode_key"]).
    That wipes ONLY this mode's ChromaDB collection.
    The other three collections are completely untouched.

    A two-step confirmation prevents accidental wipes.
    """
    mode_key = cfg["mode_key"]
    accent   = cfg["accent"]
    count    = mm.get_stats().get(mode_key, 0)

    col_stat, col_btn = st.columns([4, 1])

    with col_stat:
        st.markdown(
            f"<div style='"
            f"font-family: IBM Plex Mono, monospace;"
            f"font-size: .62rem;"
            f"color: #444;"
            f"letter-spacing: .08em;"
            f"text-transform: uppercase;"
            f"padding-top: .4rem;'>"
            f"🗄️ Memory &nbsp;·&nbsp; "
            f"<span style='color:{accent}'>{count}</span> "
            f"{cfg['memory_label']} stored"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_btn:
        if st.button(
            "🗑️ Reset",
            key  = f"reset_btn_{mode_key}",
            help = f"Wipes ONLY the {cfg['title']} collection. Other modes unaffected.",
            use_container_width=True,
        ):
            st.session_state[f"_confirm_{mode_key}"] = True

    # Two-step confirmation
    if st.session_state.get(f"_confirm_{mode_key}", False):
        st.warning(
            f"⚠️  Delete all **{count}** {cfg['memory_label']}? "
            "Other modes are **not** affected."
        )
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("✅ Confirm", key=f"_confirm_do_{mode_key}", type="primary"):
                mm.clear_collection(mode_key)
                # Also clear the in-session chat history for this mode
                st.session_state[f"_chat_{mode_key}"]    = []
                st.session_state[f"_confirm_{mode_key}"] = False
                st.success(f"✅ {cfg['title']} memory cleared.")
                st.rerun()
        with c2:
            if st.button("❌ Cancel", key=f"_confirm_no_{mode_key}"):
                st.session_state[f"_confirm_{mode_key}"] = False
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Chat area
# ─────────────────────────────────────────────────────────────────────────────

def _render_chat(cfg: dict) -> None:
    """
    Namespaced conversation area.

    Each mode stores its messages in st.session_state["_chat_{mode_key}"].
    Switching modes in the sidebar never clears another mode's history —
    they are completely independent lists.
    """
    mode_key  = cfg["mode_key"]
    chat_key  = f"_chat_{mode_key}"

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    history: list[dict] = st.session_state[chat_key]

    # Render prior messages
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input(cfg["input_hint"])
    if not (user_input and user_input.strip()):
        return

    user_input = user_input.strip()
    history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Reply ──────────────────────────────────────────────────────────────
    # This calls _placeholder_reply() by default.
    # When you wire a real graph for a mode, override this in that mode's
    # own file by replacing the call with your graph invocation, e.g.:
    #
    #   result = st.session_state.graph.invoke({"messages": [...], ...})
    #   reply  = result["messages"][-1].content
    #
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = _placeholder_reply(cfg, user_input, history)
            st.markdown(reply)

    history.append({"role": "assistant", "content": reply})


def _placeholder_reply(cfg: dict, user_input: str, history: list) -> str:
    """
    Default reply used until a real LangGraph pipeline is connected.

    To connect the real LLM immediately (without a full graph),
    replace this function body with:

        from core.llm_config import invoke
        return invoke(cfg["system_prompt"], user_input)

    To connect a full LangGraph pipeline, override _render_chat() in
    the mode's own file and build the graph call there.
    """
    return (
        f"**[{cfg['icon']} {cfg['title']}]** — pipeline not yet connected.\n\n"
        f"Input received: *\"{user_input}\"*\n\n"
        f"To connect the LLM, replace `_placeholder_reply()` in "
        f"`pages/placeholder_ui.py` with a call to `core.llm_config.invoke()`, "
        f"or wire a full LangGraph pipeline in `pages/{cfg['mode_key']}_ui.py`."
    )
