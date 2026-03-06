"""
pages/ripple_ui.py
Ripple Engine — 3-tier consequence mapping.
render() is called directly by app.py router.
"""
import re
import streamlit as st
from core.llm_config import RIPPLE_PROMPT, stream as llm_stream


# Appended to RIPPLE_PROMPT to force structured ## section output
_FORMAT_INSTRUCTION = """

MANDATORY FORMAT — use these exact section headers:

## 1st Order — Immediate Consequences
What happens directly and immediately.

## 2nd Order — Systems & People Effects
What those immediate effects then cause in the people and systems around you.

## 3rd Order — Long-term Ripple
The slow, systemic shifts that emerge over months and years.

## Blindspots
Hidden assumptions the user is NOT questioning that they should be.

## And then what?
One sharp closing question that pushes thinking one level deeper.
"""


def render() -> None:
    mm = st.session_state.memory_manager

    # ── Title ─────────────────────────────────────────────────────────────────
    st.title("🔮 Ripple Engine")
    st.caption("Map 1st, 2nd, and 3rd order consequences of any major decision.")

    st.divider()

    # ── Decision input ────────────────────────────────────────────────────────
    decision = st.text_area(
        "🔮 Decision to Analyse",
        placeholder=(
            "Describe the decision you are weighing. Be specific.\n\n"
            "e.g. 'I am considering leaving my stable job to freelance full-time'\n"
            "e.g. 'I want to relocate from Mumbai to Berlin'\n"
            "e.g. 'I am turning down a promotion to stay in my current team'"
        ),
        height = 120,
    )

    # Optional domain context
    col_domain, col_horizon = st.columns(2)
    with col_domain:
        domain = st.selectbox(
            "Primary Domain",
            ["Other", "Career", "Financial", "Relationships", "Health", "Business"],
        )
    with col_horizon:
        horizon = st.selectbox(
            "Time Horizon",
            ["Short-term (weeks)", "Medium-term (months)", "Long-term (years)"],
            index=1,
        )

    # ── Map Ripples button ────────────────────────────────────────────────────
    if st.button("🌊 Map Ripples", type="primary"):

        if not decision.strip():
            st.warning("Enter a decision before mapping.")
            return

        # Check for similar past decisions in memory (blindspot detection)
        past = mm.query_past_decisions(decision.strip(), top_k=3)
        past_blindspots = [
            r for r in past
            if r["metadata"].get("blindspot") and r["relevance"] >= 0.35
        ]

        # Surface repeating pattern warning if found
        if past_blindspots:
            st.warning(
                f"⚠️ **Repeating Pattern Detected** — "
                f"{len(past_blindspots)} similar decision(s) in your history. "
                f"Blindspot from last time: "
                f"*\"{past_blindspots[0]['metadata'].get('blindspot','')[:150]}\"*"
            )

        # Build context for LLM
        context = (
            f"DECISION: {decision.strip()}\n"
            f"DOMAIN: {domain}\n"
            f"TIME HORIZON: {horizon}"
        )

        st.divider()
        st.subheader("🌊 Consequence Map")

        # Stream response with forced structure
        system = RIPPLE_PROMPT + _FORMAT_INSTRUCTION
        with st.spinner("Mapping consequences..."):
            full_text = st.write_stream(
                llm_stream(system_prompt=system, user_input=context)
            )

        # Parse the ## sections from the streamed text
        sections = _parse_sections(full_text)

        # Re-render as structured cards below the raw stream
        st.divider()
        _render_cards(sections)

        # Save each order to memory separately
        _save_to_memory(decision.strip(), domain, sections, mm)
        st.success("✅ Decision tree saved to ripple memory.")

    st.divider()

    # ── Memory count ──────────────────────────────────────────────────────────
    count = mm.get_stats().get("ripple", 0)
    st.caption(f"🗄️ {count} decision tree(s) in ripple memory.")

    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("🗑️ Clear Memory", help="Wipes ONLY ripple memory"):
            mm.clear_collection("ripple")
            st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_sections(text: str) -> dict[str, str]:
    """Splits LLM output on ## headers into a dict of section → content."""
    parts    = re.split(r"##\s*(.+?)(?=\n|$)", text, flags=re.MULTILINE)
    sections : dict[str, str] = {}
    it = iter(parts)
    next(it)   # skip preamble before first header
    try:
        while True:
            header  = next(it).strip()
            content = next(it).strip()
            sections[header] = content
    except StopIteration:
        pass
    return sections


def _render_cards(sections: dict[str, str]) -> None:
    """Renders each consequence order as a colour-coded expandable card."""
    card_config = [
        ("1st Order",  "🟢 1ST ORDER — Immediate Consequences",   "#2a9d8f"),
        ("2nd Order",  "🟡 2ND ORDER — Systems & People Effects",  "#f4a261"),
        ("3rd Order",  "🔴 3RD ORDER — Long-term Ripple",          "#e63946"),
    ]

    # Match parsed section keys flexibly (LLM may vary capitalisation slightly)
    def _find(key_fragment: str) -> str:
        for k, v in sections.items():
            if key_fragment.lower() in k.lower():
                return v
        return ""

    for (fragment, label, colour) in card_config:
        content = _find(fragment)
        if not content:
            continue
        with st.expander(label, expanded=True):
            st.markdown(
                f"<div style='border-left:3px solid {colour};"
                f"padding-left:.8rem;color:#aaa;line-height:1.8;font-size:.9rem;'>"
                f"{content.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )

    # Blindspots card
    blindspot = _find("Blindspot")
    if blindspot:
        with st.expander("⚠️ BLINDSPOTS & HIDDEN ASSUMPTIONS", expanded=True):
            st.warning(blindspot)

    # Closing question
    atw = _find("And then")
    if atw:
        st.info(f"**And then what?** {atw}")


def _save_to_memory(
    decision: str, domain: str, sections: dict, mm
) -> None:
    """Saves each consequence order as a separate memory entry."""
    def _find(key_fragment: str) -> str:
        for k, v in sections.items():
            if key_fragment.lower() in k.lower():
                return v
        return ""

    blindspot = _find("Blindspot")[:400]
    domain_clean = domain.lower() if domain != "Other" else "other"

    order_map = {
        "1st Order" : 1,
        "2nd Order" : 2,
        "3rd Order" : 3,
    }
    for fragment, order_num in order_map.items():
        content = _find(fragment)
        if content:
            mm.save_decision(
                decision    = decision,
                consequence = content[:400],
                order       = order_num,
                domain      = domain_clean,
                blindspot   = blindspot if order_num == 1 else "",
            )
