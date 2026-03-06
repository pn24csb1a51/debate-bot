"""
app.py  —  Cognitive OS  |  Entry Point & Router
=================================================
Run with:
    streamlit run app.py

Responsibilities of this file (and ONLY this file):
  1. Apply the global dark-theme CSS once.
  2. Initialise MemoryManager + LLMs once into st.session_state.
  3. Render the sidebar mode selector.
  4. Route to the correct page's render() based on the selected mode.

Adding a new mode in the future:
  1. Create pages/newmode_ui.py with a MODE_CONFIG dict and render().
  2. Add one import line here.
  3. Add one elif in route().
  That's it. Nothing else changes.
"""

import sys
import os

# Make sure 'core' and 'pages' are importable when running from any directory.
# This is the only sys.path manipulation needed — it runs once at startup.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ── Shared infrastructure ────────────────────────────────────────────────────
from core.memory_manager import MemoryManager
from core.llm_config import (
    get_llm,
    get_fast_llm,
    get_provider_name,
    get_model_name,
)

# ── Mode page renderers ───────────────────────────────────────────────────────
from ui.debate_ui    import render as render_debate
from ui.brain_map_ui import render as render_brain_map
from ui.execution_ui import render as render_execution
from ui.ripple_ui    import render as render_ripple
from ui.wealth_ui    import render as render_wealth


# ─────────────────────────────────────────────────────────────────────────────
# Page config  —  MUST be the first Streamlit call in the file
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title            = "Cognitive OS",
    page_icon             = "🧠",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Mode registry
# ─────────────────────────────────────────────────────────────────────────────

# Display label in the selectbox  →  (render_fn, memory_key, subtitle)
_MODES: dict[str, tuple] = {
    "🔴 Devil's Advocate"   : (render_debate,    "devils_advocate", "Logic & rhetoric stress-testing"),
    "📅 Execution Architect": (render_execution, "execution",       "Energy-aware task strategy"),
    "🔮 Ripple Engine"      : (render_ripple,    "ripple",          "2nd-order consequence mapping"),
    "📉 Wealth Logic"       : (render_wealth,    "wealth",          "Financial reasoning auditor"),
    "🧠 Brain Map"          : (render_brain_map, "brain_map",       "Semantic memory network"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Global styles  —  shared dark editorial design system
# ─────────────────────────────────────────────────────────────────────────────

def global_styles() -> None:
    """
    Injects the CSS design system once. Every mode renders inside this.
    Mode pages only need to declare their accent colour — base styles
    are inherited automatically.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* ── Design tokens ───────────────────────────────── */
    :root {
        --bg          : #080808;
        --panel       : #101010;
        --border      : #1a1a1a;
        --text        : #bebebe;
        --text-bright : #ededed;
        --muted       : #424242;
        --mono        : 'IBM Plex Mono', monospace;
        --sans        : 'Inter', sans-serif;
    }

    /* ── Base ────────────────────────────────────────── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color : var(--bg)   !important;
        color            : var(--text) !important;
        font-family      : var(--sans) !important;
    }

    .main .block-container {
        padding-top : 2rem   !important;
        max-width   : 860px  !important;
    }

    /* ── Sidebar ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background   : var(--panel)  !important;
        border-right : 1px solid var(--border) !important;
    }
    [data-testid="stSidebarContent"] {
        padding : 1.5rem 1.1rem !important;
    }

    /* ── Selectbox ───────────────────────────────────── */
    [data-testid="stSelectbox"] > div > div {
        background    : var(--bg)          !important;
        border        : 1px solid #262626  !important;
        border-radius : 3px                !important;
        color         : var(--text-bright) !important;
        font-family   : var(--mono)        !important;
        font-size     : 0.84rem            !important;
    }
    [data-testid="stSelectbox"] svg {
        fill : var(--muted) !important;
    }

    /* ── Chat messages ───────────────────────────────── */
    [data-testid="stChatMessage"] {
        background    : transparent !important;
        border-bottom : 1px solid var(--border) !important;
        padding       : 1rem 0 !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-family : var(--sans)         !important;
        font-size   : 0.92rem             !important;
        line-height : 1.78                !important;
        color       : var(--text-bright)  !important;
    }

    /* ── Chat input ──────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        background    : var(--panel)       !important;
        border        : 1px solid #1e1e1e  !important;
        color         : var(--text-bright) !important;
        font-family   : var(--sans)        !important;
        font-size     : 0.9rem             !important;
        border-radius : 4px                !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color : #2e2e2e  !important;
        box-shadow   : none     !important;
        outline      : none     !important;
    }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        background    : transparent      !important;
        border        : 1px solid #222   !important;
        color         : var(--muted)     !important;
        font-family   : var(--mono)      !important;
        font-size     : 0.72rem          !important;
        border-radius : 2px              !important;
        transition    : all 0.15s        !important;
        padding       : 0.3rem 0.75rem   !important;
    }
    .stButton > button:hover {
        border-color : #3a3a3a            !important;
        color        : var(--text-bright) !important;
    }
    .stButton > button[kind="primary"] {
        border-color : #484848            !important;
        color        : var(--text-bright) !important;
    }

    /* ── Alert boxes ─────────────────────────────────── */
    [data-testid="stAlert"] {
        background  : #0f0f0f        !important;
        border      : 1px solid #222 !important;
        border-radius: 3px           !important;
    }

    /* ── Dividers ────────────────────────────────────── */
    hr {
        border-color : var(--border) !important;
        margin       : 0.9rem 0      !important;
    }

    /* ── Spinner ─────────────────────────────────────── */
    [data-testid="stSpinner"] p {
        color       : var(--muted)  !important;
        font-family : var(--mono)   !important;
        font-size   : 0.75rem       !important;
    }

    /* ── Hide Streamlit chrome ───────────────────────── */
    #MainMenu                          { visibility: hidden  !important; }
    footer                             { visibility: hidden  !important; }
    header                             { visibility: visible !important; }
    [data-testid="stHeader"]           { background: transparent !important;
                                         border-bottom: none !important; }
    .stDeployButton                    { display: none       !important; }
    [data-testid="stDecoration"]       { display: none       !important; }
    [data-testid="stMainBlockContainer"] > div:first-child { padding-top: 0 !important; }

    /* ── Scrollbar ───────────────────────────────────── */
    ::-webkit-scrollbar       { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: #202020; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #2e2e2e; }

    /* ── Sidebar always visible ──────────────────────────── */
    [data-testid="stSidebarCollapsedControl"] {
        display      : flex    !important;
        visibility   : visible !important;
    }
    section[data-testid="stSidebar"] {
        min-width    : 280px   !important;
        transform    : none    !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session initialisation  —  runs once per browser session
# ─────────────────────────────────────────────────────────────────────────────

def init_session() -> None:
    """
    Creates expensive shared objects once and stores them in session_state.

    MemoryManager opens ChromaDB connections — we never want to do this
    on every Streamlit rerun. Same for LLM clients.

    Every page module and graph node reads these from session_state:
        mm       = st.session_state.memory_manager
        llm      = st.session_state.llm
        fast_llm = st.session_state.fast_llm
    """
    if "memory_manager" not in st.session_state:
        with st.spinner("Initialising memory system..."):
            st.session_state.memory_manager = MemoryManager()

    if "llm" not in st.session_state:
        st.session_state.llm      = get_llm()
        st.session_state.fast_llm = get_fast_llm()

    if "active_mode" not in st.session_state:
        st.session_state.active_mode = list(_MODES.keys())[0]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """
    Renders the sidebar navigation and status information.
    Returns the currently selected mode label string.
    """
    with st.sidebar:

        # ── Identity ──────────────────────────────────────────────────────
        st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace;
                        margin-bottom:1.8rem;padding-bottom:.2rem;
                        border-bottom:1px solid #151515;">
                <div style="font-size:1.05rem;font-weight:700;
                            color:#e8e8e8;letter-spacing:.03em;">
                    🧠 COGNITIVE OS
                </div>
                <div style="font-size:.56rem;color:#2c2c2c;
                            letter-spacing:.22em;text-transform:uppercase;
                            margin-top:.3rem;">
                    Multi-Mode Reasoning Engine
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── Mode selector ─────────────────────────────────────────────────
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;"
            "font-size:.56rem;letter-spacing:.2em;text-transform:uppercase;"
            "color:#2e2e2e;margin-bottom:.4rem;'>Operating Mode</div>",
            unsafe_allow_html=True,
        )

        mode_labels = list(_MODES.keys())
        current_idx = mode_labels.index(st.session_state.active_mode)

        selected = st.selectbox(
            label            = "mode_select",
            options          = mode_labels,
            index            = current_idx,
            label_visibility = "collapsed",
            key              = "_sidebar_mode",
        )

        # Persist + rerun so the main area updates immediately on change
        if selected != st.session_state.active_mode:
            st.session_state.active_mode = selected
            st.rerun()

        # Mode subtitle
        subtitle = _MODES[selected][2]
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;"
            f"font-size:.6rem;color:#2e2e2e;margin-top:.3rem;"
            f"line-height:1.55;'>{subtitle}</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Memory counts ─────────────────────────────────────────────────
        mm    = st.session_state.memory_manager
        stats = mm.get_stats()

        labels_keys = [
            ("🔴 Devil's Advocate", "devils_advocate"),
            ("📅 Execution",        "execution"),
            ("🔮 Ripple",           "ripple"),
            ("📉 Wealth",           "wealth"),
        ]
        rows = "".join(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:.18rem 0;'>"
            f"<span style='color:#282828;'>{lbl}</span>"
            f"<span style='color:#303030;font-variant-numeric:tabular-nums;'>"
            f"{stats.get(key, 0)}"
            f"</span></div>"
            for lbl, key in labels_keys
        )
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;"
            f"font-size:.62rem;margin-bottom:.5rem;'>"
            f"<div style='font-size:.54rem;letter-spacing:.16em;"
            f"text-transform:uppercase;color:#222;margin-bottom:.55rem;'>"
            f"Memory Stores</div>"
            f"{rows}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Status bar ────────────────────────────────────────────────────
        provider = get_provider_name()
        model    = get_model_name("full")
        fast     = get_model_name("fast")
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;"
            f"font-size:.56rem;color:#252525;line-height:2.1;'>"
            f"<span style='color:#1e1e1e;'>Provider</span>&ensp;{provider}<br>"
            f"<span style='color:#1e1e1e;'>Model  </span>&ensp;{model}<br>"
            f"<span style='color:#1e1e1e;'>Fast   </span>&ensp;{fast}"
            f"</div>",
            unsafe_allow_html=True,
        )

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Router  —  delegates to the correct page module
# ─────────────────────────────────────────────────────────────────────────────

def route(mode_label: str) -> None:
    """
    Calls render() on the page module that owns the selected mode.

    The render function for each mode is stored in _MODES at index [0].
    This keeps the routing logic to a single dict lookup — no if/elif chain.
    """
    entry = _MODES.get(mode_label)
    if entry is None:
        st.error(
            f"Unknown mode: **'{mode_label}'**. "
            "Check the `_MODES` dict in `app.py`."
        )
        return
    render_fn = entry[0]
    render_fn()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global_styles()
    init_session()
    active_mode = render_sidebar()
    route(active_mode)


if __name__ == "__main__":
    main()
