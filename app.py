"""
app.py  ·  Scorecard Edition – Streamlit Web UI
================================================
Adds to Phase 5:
  • Live score metrics for both Human and Advocate in sidebar
  • Score delta flash (+/-) after each turn
  • "Generate Final Report" button with full post-match verdict
  • Reasoning tooltip showing why each side gained/lost points

Launch with:
    streamlit run app.py
"""

import html as _html
import re
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from brain import (
    DebateState,
    SelfPlayTurnResult,
    _extract_severity,
    build_debate_graph,
    build_llm,
    generate_final_report,
    generate_self_play_verdict,
    run_self_play_round,
)
from memory import DebateMemory

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Devil's Advocate",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --red        : #e63946;
    --red-dim    : #a52633;
    --blue       : #4a90d9;
    --blue-dim   : #2a5a8c;
    --black      : #0d0d0d;
    --charcoal   : #141414;
    --panel      : #1a1a1a;
    --border     : #2a2a2a;
    --muted      : #555;
    --text       : #d4d4d4;
    --text-bright: #f0f0f0;
    --mono       : 'IBM Plex Mono', monospace;
    --sans       : 'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family : var(--sans) !important;
    background  : var(--black) !important;
    color       : var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.main .block-container {
    padding    : 2rem 2.5rem 4rem !important;
    max-width  : 860px !important;
    margin     : 0 auto !important;
}

/* ── Masthead ── */
.masthead {
    display        : flex;
    align-items    : baseline;
    gap            : 1rem;
    border-bottom  : 1px solid var(--red);
    padding-bottom : 1rem;
    margin-bottom  : 2rem;
}
.masthead-title {
    font-family   : var(--mono);
    font-size     : 1.5rem;
    font-weight   : 600;
    color         : var(--red);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.masthead-sub {
    font-family   : var(--mono);
    font-size     : 0.7rem;
    color         : var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background : transparent !important;
    border     : none !important;
    padding    : 0 !important;
    margin     : 0 !important;
}

.msg-user {
    background    : var(--panel);
    border        : 1px solid var(--border);
    border-radius : 2px 12px 12px 12px;
    padding       : 1rem 1.25rem;
    margin        : 0.75rem 0;
}
.msg-user::before {
    content      : "YOU";
    font-family  : var(--mono);
    font-size    : 0.6rem;
    font-weight  : 600;
    color        : var(--muted);
    letter-spacing: 0.15em;
    display      : block;
    margin-bottom: 0.4rem;
}
.msg-bot {
    background    : #180a0a;
    border        : 1px solid var(--red-dim);
    border-left   : 3px solid var(--red);
    border-radius : 12px 2px 12px 12px;
    padding       : 1rem 1.25rem;
    margin        : 0.75rem 0;
}
.msg-bot-header {
    font-family   : var(--mono);
    font-size     : 0.6rem;
    font-weight   : 600;
    color         : var(--red);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom : 0.5rem;
    display       : flex;
    align-items   : center;
    gap           : 0.5rem;
    flex-wrap     : wrap;
}
.msg-bot-body, .msg-user-body {
    font-family : var(--sans);
    font-size   : 0.92rem;
    line-height : 1.7;
    white-space : pre-wrap;
}
.msg-bot-body  { color: var(--text-bright); }
.msg-user-body { color: var(--text); }

/* ── Score flash on message ── */
.score-flash {
    font-family  : var(--mono);
    font-size    : 0.65rem;
    border-radius: 3px;
    padding      : 2px 6px;
    margin-left  : 4px;
}
.score-plus { background: #1a3a1a; color: #4caf50; border: 1px solid #2a6a2a; }
.score-minus{ background: #3a1a1a; color: #e57373; border: 1px solid #6a2a2a; }
.score-zero { background: #2a2a2a; color: #888; border: 1px solid #444; }

/* ── Badges ── */
.badge {
    display       : inline-block;
    font-family   : var(--mono);
    font-size     : 0.55rem;
    font-weight   : 600;
    letter-spacing: 0.1em;
    padding       : 2px 6px;
    border-radius : 2px;
    text-transform: uppercase;
}
.badge-high   { background: #e63946; color: #fff; }
.badge-medium { background: #e6a817; color: #000; }
.badge-low    { background: #2a9d5c; color: #fff; }
.badge-memory { background: #1a3a5c; color: #7eb8f7; border: 1px solid #2a5a8c; }
.badge-tactic { background: #2a1a3a; color: #b87ef7; border: 1px solid #5a2a8c; }

/* ── Scorecard widget ── */
.scorecard {
    border        : 1px solid var(--border);
    border-radius : 4px;
    padding       : 0;
    overflow      : hidden;
    margin-bottom : 1rem;
    background    : var(--panel);
}
.scorecard-title {
    font-family   : var(--mono);
    font-size     : 0.6rem;
    font-weight   : 600;
    color         : var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding       : 0.5rem 0.75rem;
    border-bottom : 1px solid var(--border);
}
.scorecard-body {
    display : flex;
    gap     : 0;
}
.score-side {
    flex          : 1;
    padding       : 0.75rem;
    text-align    : center;
}
.score-side-human  { border-right: 1px solid var(--border); }
.score-side-label {
    font-family   : var(--mono);
    font-size     : 0.55rem;
    color         : var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom : 0.2rem;
}
.score-number {
    font-family : var(--mono);
    font-size   : 1.8rem;
    font-weight : 600;
    line-height : 1;
}
.score-number-human    { color: var(--blue); }
.score-number-advocate { color: var(--red); }
.score-delta {
    font-family : var(--mono);
    font-size   : 0.65rem;
    margin-top  : 0.2rem;
    font-weight : 600;
}
.delta-neg { color: #e57373; }
.delta-pos { color: #4caf50; }
.delta-zer { color: #888; }
.score-reason {
    font-family : var(--sans);
    font-size   : 0.62rem;
    color       : var(--muted);
    padding     : 0.4rem 0.75rem;
    border-top  : 1px solid var(--border);
    line-height : 1.4;
    font-style  : italic;
}
.score-vs {
    font-family   : var(--mono);
    font-size     : 0.65rem;
    color         : var(--muted);
    display       : flex;
    align-items   : center;
    justify-content: center;
    padding       : 0 0.5rem;
    border-right  : 1px solid var(--border);
}
.leader-badge {
    font-family   : var(--mono);
    font-size     : 0.6rem;
    padding       : 2px 6px;
    border-radius : 2px;
    font-weight   : 600;
    text-align    : center;
    margin        : 0 0.75rem 0.5rem;
}
.leader-human    { background: #1a2a3a; color: var(--blue); border: 1px solid var(--blue-dim); }
.leader-bot      { background: #2a1010; color: var(--red);  border: 1px solid var(--red-dim);  }
.leader-tied     { background: #252525; color: #888;        border: 1px solid #444; }

/* ── Final report ── */
.final-report {
    background    : #0e0e0e;
    border        : 1px solid var(--border);
    border-radius : 4px;
    padding       : 1.25rem;
    font-family   : var(--sans);
    font-size     : 0.85rem;
    line-height   : 1.8;
    color         : var(--text-bright);
    white-space   : pre-wrap;
    margin-top    : 0.5rem;
    max-height    : 500px;
    overflow-y    : auto;
}

/* ── Chat input ── */
[data-testid="stChatInput"] > div {
    background    : var(--panel) !important;
    border        : 1px solid var(--border) !important;
    border-radius : 4px !important;
}
[data-testid="stChatInput"] textarea {
    font-family : var(--sans) !important;
    color       : var(--text-bright) !important;
    background  : transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background   : var(--charcoal) !important;
    border-right : 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }

.sidebar-section {
    border        : 1px solid var(--border);
    border-radius : 4px;
    padding       : 1rem;
    margin-bottom : 1rem;
    background    : var(--panel);
}
.sidebar-label {
    font-size     : 0.6rem;
    font-weight   : 600;
    color         : var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom : 0.5rem;
}
.sidebar-report {
    font-size   : 0.72rem;
    color       : var(--text);
    line-height : 1.6;
    white-space : pre-wrap;
    max-height  : 320px;
    overflow-y  : auto;
    padding-right: 4px;
}
.stat-grid {
    display              : grid;
    grid-template-columns: 1fr 1fr;
    gap                  : 0.5rem;
    margin-top           : 0.5rem;
}
.stat-cell {
    background    : var(--charcoal);
    border        : 1px solid var(--border);
    border-radius : 3px;
    padding       : 0.5rem 0.6rem;
    text-align    : center;
}
.stat-num {
    font-size   : 1.2rem;
    font-weight : 600;
    color       : var(--red);
    display     : block;
}
.stat-lbl {
    font-size     : 0.55rem;
    color         : var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Buttons ── */
.stButton > button {
    background    : transparent !important;
    border        : 1px solid var(--border) !important;
    color         : var(--text) !important;
    font-family   : var(--mono) !important;
    font-size     : 0.7rem !important;
    letter-spacing: 0.1em !important;
    border-radius : 3px !important;
    width         : 100% !important;
    transition    : all 0.15s ease !important;
}
.stButton > button:hover {
    border-color : var(--red) !important;
    color        : var(--red) !important;
}

/* ── Judge panel ── */
.judge-upheld {
    border        : 1px solid #2a4a2a;
    border-left   : 3px solid #2a9d5c;
    background    : #0a180a;
    border-radius : 3px;
    padding       : 0.6rem 0.8rem;
    margin-bottom : 0.5rem;
}
.judge-overruled {
    border        : 1px solid #6a2020;
    border-left   : 3px solid #e63946;
    background    : #1a0808;
    border-radius : 3px;
    padding       : 0.6rem 0.8rem;
    margin-bottom : 0.5rem;
    animation     : pulse-red 1.5s ease-in-out 2;
}
@keyframes pulse-red {
    0%, 100% { border-color: #6a2020; border-left-color: #e63946; }
    50%       { border-color: #a03030; border-left-color: #ff6b6b; }
}
.judge-label {
    font-family   : var(--mono);
    font-size     : 0.6rem;
    font-weight   : 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom : 0.3rem;
}
.judge-label-upheld   { color: #2a9d5c; }
.judge-label-overruled{ color: var(--red); }
.judge-text {
    font-family : var(--sans);
    font-size   : 0.72rem;
    line-height : 1.5;
    color       : var(--text);
}
.overrule-banner {
    background    : #2a0808;
    border        : 1px solid #e63946;
    border-radius : 3px;
    padding       : 0.75rem 1rem;
    margin-bottom : 0.75rem;
    font-family   : var(--mono);
    font-size     : 0.7rem;
    color         : #ff8a8a;
    text-align    : center;
    font-weight   : 600;
    letter-spacing: 0.05em;
}

/* ── Verdict button ── */
div[data-testid="stButton"].verdict-btn > button {
    border-color : #4a90d9 !important;
    color        : #4a90d9 !important;
    font-weight  : 600 !important;
}
div[data-testid="stButton"].verdict-btn > button:hover {
    background   : #0a1a2a !important;
}

.stSpinner > div { border-top-color: var(--red) !important; }
hr { border-color: var(--border) !important; }
[data-testid="stAlert"] {
    border-radius : 3px !important;
    font-family   : var(--mono) !important;
    font-size     : 0.75rem !important;
}
.tactic-item {
    border-left   : 2px solid #5a2a8c;
    padding       : 0.4rem 0.6rem;
    margin-bottom : 0.4rem;
    font-size     : 0.72rem;
    color         : var(--text);
    background    : #1a0e2a;
}
.tactic-name { color: #b87ef7; font-weight: 600; display: block; margin-bottom: 2px; }
.tactic-meta { color: var(--muted); font-size: 0.62rem; }
.welcome-card {
    border     : 1px solid var(--border);
    border-left: 3px solid var(--red);
    padding    : 1.5rem;
    background : #120808;
    margin     : 2rem 0;
}
.welcome-title { font-family: var(--mono); font-size: 1rem; color: var(--red); font-weight: 600; margin-bottom: 0.5rem; }
.welcome-text  { font-size: 0.85rem; color: var(--text); line-height: 1.7; }
.welcome-cmd   { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); border: 1px solid var(--border); padding: 0.2rem 0.5rem; border-radius: 2px; display: inline-block; margin-top: 0.3rem; }

/* ── Echo Chamber ── */
.echo-header { font-family: var(--mono); font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.4rem; }
.echo-proponent { background: #090e1a; border: 1px solid #1e3a6e; border-left: 3px solid #4a90d9; border-radius: 12px 2px 12px 12px; padding: 1rem 1.25rem; margin: 0.6rem 0; }
.echo-advocate  { background: #180a0a; border: 1px solid var(--red-dim); border-left: 3px solid var(--red); border-radius: 2px 12px 12px 12px; padding: 1rem 1.25rem; margin: 0.6rem 0; }
.echo-header-pro { color: #4a90d9; }
.echo-header-adv { color: var(--red); }
.echo-body { font-family: var(--sans); font-size: 0.9rem; line-height: 1.7; color: var(--text-bright); white-space: pre-wrap; }
.echo-meta { font-family: var(--mono); font-size: 0.6rem; color: var(--muted); margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px solid var(--border); display: flex; gap: 0.75rem; flex-wrap: wrap; }
.round-divider { font-family: var(--mono); font-size: 0.65rem; color: var(--muted); letter-spacing: 0.2em; text-transform: uppercase; text-align: center; padding: 0.5rem 0; border-top: 1px solid var(--border); margin: 0.75rem 0 0.25rem; }
.echo-scoreboard { display: grid; grid-template-columns: 1fr auto 1fr; gap: 0.5rem; align-items: center; background: var(--panel); border: 1px solid var(--border); border-radius: 4px; padding: 0.75rem; margin: 0.75rem 0; font-family: var(--mono); }
.echo-score-num-pro { font-size: 1.4rem; font-weight: 700; color: #4a90d9; display: block; }
.echo-score-num-adv { font-size: 1.4rem; font-weight: 700; color: var(--red); display: block; }
.echo-score-lbl { font-size: 0.55rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; display: block; }
.echo-vs { font-size: 0.8rem; color: var(--muted); font-weight: 700; }
.echo-verdict { background: var(--panel); border: 1px solid var(--border); border-radius: 4px; padding: 1.25rem; font-family: var(--sans); font-size: 0.85rem; line-height: 1.8; color: var(--text-bright); white-space: pre-wrap; margin-top: 1rem; }
.echo-topic-banner { font-family: var(--mono); font-size: 0.75rem; color: #4a90d9; border: 1px solid #1e3a6e; background: #060d1a; border-radius: 3px; padding: 0.5rem 0.75rem; margin-bottom: 1rem; }
.echo-running-badge { display: inline-block; font-family: var(--mono); font-size: 0.65rem; color: #4a90d9; border: 1px solid #1e3a6e; background: #060d1a; border-radius: 3px; padding: 0.3rem 0.6rem; margin-bottom: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
def init_session():
    if "debate_memory" not in st.session_state:
        st.session_state.debate_memory        = DebateMemory()
    if "llm" not in st.session_state:
        st.session_state.llm                  = build_llm(fast=False)
        st.session_state.fast_llm             = build_llm(fast=True)
    if "graph" not in st.session_state:
        st.session_state.graph = build_debate_graph(
            st.session_state.llm,
            st.session_state.fast_llm,
            st.session_state.debate_memory,
        )
    defaults = {
        "conversation_history"  : [],
        "display_messages"      : [],
        "last_bot_response"     : "",
        "last_analysis_report"  : "",
        "last_severity"         : "",
        "last_memories_used"    : 0,
        "last_strategies_used"  : 0,
        "last_concession"       : False,
        "turn_count"            : 0,
        # ── Scorecard ──
        "user_score"            : 0,
        "bot_score"             : 0,
        "last_user_delta"       : 0,
        "last_bot_delta"        : 0,
        "last_score_reason_user": "",
        "last_score_reason_bot" : "",
        "final_report"          : "",
        "show_final_report"     : False,
        # ── Judge ──
        "last_judge_verdict"    : "",
        "last_judge_reasoning"  : "",
        "last_is_overruled"     : False,
        # ── Echo Chamber ──
        "echo_topic"            : "",
        "echo_messages"         : [],    # list of SelfPlayTurnResult
        "echo_history"          : [],    # list of (role, content) for brain
        "echo_running"          : False,
        "echo_current_round"    : 0,
        "echo_max_rounds"       : 5,
        "echo_pro_score"        : 0,
        "echo_adv_score"        : 0,
        "echo_verdict"          : "",
        "echo_active_tab"       : "debate",  # "debate" | "echo"
        # ── Weakness Radar (cumulative across all turns) ──
        "weakness_profile"      : {
            "Fallacies"       : 0,
            "Contradictions"  : 0,
            "Factual Errors"  : 0,
            "Severity"        : 0,
            "Bullying"        : 0,
        },
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_debate():
    keys = [
        "conversation_history", "display_messages", "last_bot_response",
        "last_analysis_report", "last_severity", "last_memories_used",
        "last_strategies_used", "last_concession", "turn_count",
        "user_score", "bot_score", "last_user_delta", "last_bot_delta",
        "last_score_reason_user", "last_score_reason_bot",
        "final_report", "show_final_report",
        "last_judge_verdict", "last_judge_reasoning", "last_is_overruled",
    ]
    for k in keys:
        st.session_state[k] = [] if "history" in k or "messages" in k else (
            0 if "score" in k or "delta" in k or "count" in k or "used" in k
            else False if k in ("last_concession", "show_final_report")
            else ""
        )
    # Reset weakness radar separately (it's a dict, not a simple scalar)
    st.session_state.weakness_profile = {
        "Fallacies"     : 0,
        "Contradictions": 0,
        "Factual Errors": 0,
        "Severity"      : 0,
        "Bullying"      : 0,
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _delta_html(delta: int) -> str:
    if delta > 0:
        return f'<span class="score-delta delta-pos">+{delta} pts this turn</span>'
    elif delta < 0:
        return f'<span class="score-delta delta-neg">{delta} pts this turn</span>'
    return '<span class="score-delta delta-zer">±0 this turn</span>'


def _leader_html(user: int, bot: int) -> str:
    if user > bot:
        return '<div class="leader-badge leader-human">🧑 HUMAN LEADS</div>'
    elif bot > user:
        return '<div class="leader-badge leader-bot">🔴 ADVOCATE LEADS</div>'
    return '<div class="leader-badge leader-tied">🤝 TIED</div>'


def build_badges(severity: str, memories: int, strategies: int, concession: bool) -> str:
    parts = []
    if concession:
        parts.append('<span class="badge badge-tactic">✅ concession</span>')
    if strategies > 0:
        parts.append(f'<span class="badge badge-tactic">🎯 {strategies} tactic{"s" if strategies > 1 else ""}</span>')
    if memories > 0:
        parts.append(f'<span class="badge badge-memory">🧠 {memories}</span>')
    if severity:
        cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(severity, "")
        parts.append(f'<span class="badge {cls}">{severity}</span>')
    return " ".join(parts)


# ─────────────────────────────────────────────
# Weakness Profile Helpers
# ─────────────────────────────────────────────

def _count_in_report(report: str, marker: str) -> int:
    """Count occurrences of a marker string in the analysis report."""
    return report.upper().count(marker.upper())


def _update_weakness_profile(
    analysis_report  : str,
    fact_verdict     : str,
    is_overruled     : bool,
    severity         : str,
    profile          : dict,
) -> dict:
    """
    Parse one turn's outputs and increment the cumulative weakness profile.
    Returns the updated profile dict.
    """
    p = dict(profile)  # shallow copy so we don't mutate in place

    # ── Fallacies — count "FALLACY:" lines in the analysis report ─────────────
    p["Fallacies"] += _count_in_report(analysis_report, "\nFALLACY:")

    # ── Contradictions — count "PAST:" lines (each contradiction has one) ──────
    p["Contradictions"] += _count_in_report(analysis_report, "\nPAST:")

    # ── Factual Errors — researcher found the claim INACCURATE ────────────────
    if fact_verdict == "INACCURATE":
        p["Factual Errors"] += 1
    elif fact_verdict == "UNVERIFIABLE":
        p["Factual Errors"] += 0  # suspicious but not confirmed — don't count

    # ── Severity — HIGH = 2 pts, MEDIUM = 1 pt ───────────────────────────────
    p["Severity"] += {"HIGH": 2, "MEDIUM": 1}.get(severity.upper(), 0)

    # ── Bullying — judge overruled for aggression/dishonesty ─────────────────
    if is_overruled:
        p["Bullying"] += 1

    return p


def _build_echo_weakness_profile(echo_messages: list) -> dict:
    """
    Builds the Proponent's weakness profile from completed Echo Chamber rounds.
    Mirrors _update_weakness_profile but operates on SelfPlayTurnResult objects.
    """
    profile = {"Fallacies": 0, "Contradictions": 0, "Factual Errors": 0,
               "Severity": 0, "Bullying": 0}
    for r in echo_messages:
        # Fact errors from researcher verdict on proponent's claim
        if r.fact_verdict == "INACCURATE":
            profile["Factual Errors"] += 1
        # Judge overruled the ADVOCATE (not proponent), but we track it as
        # "the proponent survived bullying" — flip: overruled = Advocate weakness,
        # so for Proponent radar we instead count their own argument quality.
        # Use score_reason_pro to heuristically detect poor reasoning:
        if r.proponent_delta < 0:
            profile["Fallacies"] += 1          # negative score = weak argument
        if r.is_overruled:
            # Advocate was overruled — Proponent didn't cause this, skip for Proponent
            pass
        # Severity proxy: large negative delta = high severity
        if r.proponent_delta <= -5:
            profile["Severity"] += 2
        elif r.proponent_delta < 0:
            profile["Severity"] += 1
    return profile


def render_weakness_radar(profile: dict, title: str = "Logical Weakness Radar"):
    """
    Renders a Plotly radar / spider chart for the given weakness profile dict.
    Uses Plotly (already a Streamlit dependency — no extra install needed).
    Styled to match the dark editorial aesthetic.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("Install plotly to see the Weakness Radar.")
        return

    categories = list(profile.keys())
    values     = list(profile.values())

    # Scale values to 0-10 for readable radar shape
    max_val    = max(max(values), 1)
    scaled     = [min(10, round((v / max_val) * 10)) for v in values]

    # Close the polygon
    cats_closed   = categories + [categories[0]]
    vals_closed   = scaled     + [scaled[0]]
    raw_closed    = values     + [values[0]]

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatterpolar(
        r           = vals_closed,
        theta       = cats_closed,
        fill        = "toself",
        fillcolor   = "rgba(230, 57, 70, 0.15)",
        line        = dict(color="#e63946", width=2),
        name        = title,
        hovertemplate = (
            "<b>%{theta}</b><br>"
            "Score: %{r}/10<br>"
            "<extra></extra>"
        ),
        customdata  = raw_closed,
    ))

    # Dot markers
    fig.add_trace(go.Scatterpolar(
        r         = vals_closed,
        theta     = cats_closed,
        mode      = "markers",
        marker    = dict(color="#e63946", size=6),
        hoverinfo = "skip",
        showlegend= False,
    ))

    fig.update_layout(
        polar=dict(
            bgcolor        = "rgba(0,0,0,0)",
            angularaxis    = dict(
                tickfont   = dict(size=9, color="#888", family="IBM Plex Mono"),
                linecolor  = "#2a2a2a",
                gridcolor  = "#2a2a2a",
            ),
            radialaxis     = dict(
                visible    = True,
                range      = [0, 10],
                tickfont   = dict(size=7, color="#444"),
                gridcolor  = "#222",
                linecolor  = "#2a2a2a",
                tickvals   = [2, 4, 6, 8, 10],
                ticktext   = ["", "", "", "", ""],
            ),
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(l=20, r=20, t=28, b=10),
        height        = 230,
        showlegend    = False,
        font          = dict(family="IBM Plex Mono", color="#888"),
    )

    # Header
    all_zero = all(v == 0 for v in values)
    st.markdown(
        f'<div class="sidebar-label">🕸️ {title.upper()}</div>',
        unsafe_allow_html=True,
    )
    if all_zero:
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.62rem;'
            'color:#555;padding:0.5rem 0;text-align:center">'
            'No weaknesses detected yet.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(
            fig,
            use_container_width = True,
            config              = {"displayModeBar": False, "staticPlot": False},
        )
        # Raw counts legend
        legend_parts = [
            f'<span style="color:#888;font-size:0.58rem;font-family:\'IBM Plex Mono\',monospace">'
            f'{cat}: <span style="color:#e63946">{val}</span></span>'
            for cat, val in zip(categories, values)
        ]
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:0.4rem 0.75rem;padding:0 0.1rem 0.5rem">'
            + "  ".join(legend_parts)
            + "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-label">⚔️ DEVIL\'S ADVOCATE</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#e63946;font-size:0.7rem;margin-bottom:1.2rem">Scorecard Edition · v5.1</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("↺ RESET", key="reset_btn"):
                reset_debate()
                st.rerun()
        with col2:
            if st.button("🗑️ WIPE MEM", key="forget_btn"):
                st.session_state.debate_memory.clear_all()
                reset_debate()
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── LIVE SCORECARD ────────────────────────────────
        us   = st.session_state.user_score
        bs   = st.session_state.bot_score
        udel = st.session_state.last_user_delta
        bdel = st.session_state.last_bot_delta
        ur   = st.session_state.last_score_reason_user
        br   = st.session_state.last_score_reason_bot

        st.markdown(
            f"""
            <div class="scorecard">
                <div class="scorecard-title">📊 Live Debate Scorecard</div>
                <div class="scorecard-body">
                    <div class="score-side score-side-human">
                        <div class="score-side-label">🧑 Human</div>
                        <div class="score-number score-number-human">{us}</div>
                        {_delta_html(udel)}
                    </div>
                    <div class="score-side">
                        <div class="score-side-label">🔴 Advocate</div>
                        <div class="score-number score-number-advocate">{bs}</div>
                        {_delta_html(bdel)}
                    </div>
                </div>
                {f'<div class="score-reason">You: {ur}</div>' if ur else ''}
                {f'<div class="score-reason">Bot: {br}</div>' if br else ''}
            </div>
            {_leader_html(us, bs)}
            """,
            unsafe_allow_html=True,
        )

        # ── Severity alerts ───────────────────────────────
        severity = st.session_state.last_severity
        if severity == "HIGH":
            st.error("🔴 HIGH SEVERITY — Direct contradiction or multiple fallacies.")
        elif severity == "MEDIUM":
            st.warning("🟡 MEDIUM SEVERITY — Logical weakness detected.")

        # ── Weakness Radar ────────────────────────────────────
        render_weakness_radar(
            st.session_state.weakness_profile,
            title="Human Weakness Radar",
        )

        # ── Memory Stats ──────────────────────────────────
        stats = st.session_state.debate_memory.get_stats()
        st.markdown(
            f"""
            <div class="sidebar-section">
                <div class="sidebar-label">🗄️ Memory Stats</div>
                <div class="stat-grid">
                    <div class="stat-cell">
                        <span class="stat-num">{stats['total_debate_memories']}</span>
                        <span class="stat-lbl">Arguments</span>
                    </div>
                    <div class="stat-cell">
                        <span class="stat-num">{stats['total_winning_strategies']}</span>
                        <span class="stat-lbl">Tactics</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Logic Audit ───────────────────────────────────
        report = st.session_state.last_analysis_report
        if report:
            st.markdown(
                f"""
                <div class="sidebar-section">
                    <div class="sidebar-label">🔬 Logic Audit — Last Turn</div>
                    <div class="sidebar-report">{report}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Judicial Oversight ────────────────────────────
        is_overruled    = st.session_state.last_is_overruled
        judge_reasoning = st.session_state.last_judge_reasoning
        judge_verdict   = st.session_state.last_judge_verdict

        if judge_verdict:
            if is_overruled:
                st.markdown(
                    f"""
                    <div class="judge-overruled">
                        <div class="judge-label judge-label-overruled">
                            ⚖️ JUDICIAL OVERSIGHT — 🔨 OVERRULED
                        </div>
                        <div class="overrule-banner">
                            The Judge has overruled the Advocate's last point<br>
                            for being logically unsound.
                        </div>
                        <div class="judge-text">{judge_reasoning}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="judge-upheld">
                        <div class="judge-label judge-label-upheld">
                            ⚖️ JUDICIAL OVERSIGHT — ✅ UPHELD
                        </div>
                        <div class="judge-text">{judge_reasoning or "Advocate's response met intellectual honesty standards."}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Winning Tactics ───────────────────────────────
        tactics = st.session_state.debate_memory.get_all_strategies()
        if tactics:
            tactics_html = '<div class="sidebar-section"><div class="sidebar-label">🏆 Tactics Playbook</div>'
            for t in tactics[-4:]:
                tactics_html += (
                    f'<div class="tactic-item">'
                    f'<span class="tactic-name">{t["tactic_name"]}</span>'
                    f'<span class="tactic-meta">{t["topic"] or "General"} · {t["timestamp"][:10]}</span>'
                    f'</div>'
                )
            tactics_html += "</div>"
            st.markdown(tactics_html, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── FINAL REPORT BUTTON ───────────────────────────
        st.markdown('<div class="sidebar-label">🏁 Post-Match</div>', unsafe_allow_html=True)
        if st.button("📋 GENERATE FINAL REPORT", key="verdict_btn"):
            if not st.session_state.conversation_history:
                st.warning("No debate to analyse yet!")
            else:
                with st.spinner("Generating verdict..."):
                    st.session_state.final_report = generate_final_report(
                        llm                  = st.session_state.llm,
                        conversation_history = st.session_state.conversation_history,
                        user_score           = st.session_state.user_score,
                        bot_score            = st.session_state.bot_score,
                    )
                    st.session_state.show_final_report = True
                st.rerun()


# ─────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────
def render_chat():
    st.markdown(
        """
        <div class="masthead">
            <span class="masthead-title">🔴 Devil's Advocate</span>
            <span class="masthead-sub">Scorecard Edition · Socratic Debate Training</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Final Report panel (shown above chat when triggered) ─────────────────
    if st.session_state.get("show_final_report") and st.session_state.final_report:
        with st.expander("📋 FINAL DEBATE REPORT", expanded=True):
            st.markdown(
                f'<div class="final-report">{st.session_state.final_report}</div>',
                unsafe_allow_html=True,
            )
            if st.button("✕ Close Report", key="close_report"):
                st.session_state.show_final_report = False
                st.rerun()

    # ── Welcome card ──────────────────────────────────────────────────────────
    if not st.session_state.display_messages:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-title">Make a claim. Earn your score.</div>
                <div class="welcome-text">
                    Every argument is audited for logical fallacies and contradictions.
                    Points are awarded and deducted in real time. At the end, click
                    "Generate Final Report" for a full post-match verdict.
                    <br><br>
                    Sound logic gains standing. Fallacies cost points.
                </div>
                <div class="welcome-cmd">💡 Try: "Social media is destroying society"</div><br>
                <div class="welcome-cmd">💡 Try: "AI will take all our jobs"</div><br>
                <div class="welcome-cmd">💡 Try: "Democracy is the best form of government"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Render conversation ───────────────────────────────────────────────────
    for msg in st.session_state.display_messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user"><div class="msg-user-body">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            badges     = msg.get("badges", "")
            score_html = msg.get("score_html", "")
            content    = _html.escape(re.sub(r"<[^>]*>", "", msg["content"])).replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="msg-bot">
                    <div class="msg-bot-header">DEVIL'S ADVOCATE {badges} {score_html}</div>
                    <div class="msg-bot-body">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# Process user input
# ─────────────────────────────────────────────
def process_input(user_input: str):
    st.session_state.display_messages.append({
        "role"   : "user",
        "content": user_input,
    })
    st.session_state.conversation_history.append(HumanMessage(content=user_input))

    initial_state: DebateState = {
        "messages"               : st.session_state.conversation_history,
        "retrieved_memories"     : [],
        "memory_context"         : "",
        "analysis_report"        : "",
        "last_bot_response"      : st.session_state.last_bot_response,
        "winning_strategies"     : [],
        "strategy_context"       : "",
        "concession_detected"    : False,
        "user_score"             : st.session_state.user_score,
        "bot_score"              : st.session_state.bot_score,
        "last_user_delta"        : 0,
        "last_bot_delta"         : 0,
        "last_score_reason_user" : "",
        "last_score_reason_bot"  : "",
        "fact_check_result"      : "",
        "fact_check_verdict"     : "SKIPPED",
        "fact_check_performed"   : False,
        "fact_check_context"     : "",
        "judge_verdict"          : "",
        "judge_reasoning"        : "",
        "is_overruled"           : False,
    }

    result = st.session_state.graph.invoke(initial_state)

    ai_message            = result["messages"][-1]
    memories_used         = len(result.get("retrieved_memories",    []))
    strategies_used       = len(result.get("winning_strategies",    []))
    analysis_report       = result.get("analysis_report",           "")
    concession_detected   = result.get("concession_detected",       False)
    severity              = _extract_severity(analysis_report)
    user_score            = result.get("user_score",                st.session_state.user_score)
    bot_score             = result.get("bot_score",                 st.session_state.bot_score)
    user_delta            = result.get("last_user_delta",           0)
    bot_delta             = result.get("last_bot_delta",            0)
    score_reason_user     = result.get("last_score_reason_user",    "")
    score_reason_bot      = result.get("last_score_reason_bot",     "")
    is_overruled          = result.get("is_overruled",              False)
    judge_reasoning       = result.get("judge_reasoning",           "")
    judge_verdict         = result.get("judge_verdict",             "")

    # Persist to session state
    st.session_state.last_bot_response      = ai_message.content
    st.session_state.last_analysis_report   = analysis_report
    st.session_state.last_severity          = severity
    st.session_state.last_memories_used     = memories_used
    st.session_state.last_strategies_used   = strategies_used
    st.session_state.last_concession        = concession_detected
    st.session_state.user_score             = user_score
    st.session_state.bot_score              = bot_score
    st.session_state.last_user_delta        = user_delta
    st.session_state.last_bot_delta         = bot_delta
    st.session_state.last_score_reason_user = score_reason_user
    st.session_state.last_score_reason_bot  = score_reason_bot
    st.session_state.last_is_overruled      = is_overruled
    st.session_state.last_judge_reasoning   = judge_reasoning
    st.session_state.last_judge_verdict     = judge_verdict
    st.session_state.turn_count            += 1

    # ── Update cumulative weakness profile ────────────────────────────────────
    st.session_state.weakness_profile = _update_weakness_profile(
        analysis_report = analysis_report,
        fact_verdict    = result.get("fact_check_verdict", "SKIPPED"),
        is_overruled    = is_overruled,
        severity        = severity,
        profile         = st.session_state.weakness_profile,
    )

    st.session_state.conversation_history.append(ai_message)
    st.session_state.debate_memory.store_argument(user_input, ai_message.content)

    # Build per-message score html
    def _score_badge(delta: int, label: str) -> str:
        if delta > 0:
            cls = "score-plus"
        elif delta < 0:
            cls = "score-minus"
        else:
            cls = "score-zero"
        sign = "+" if delta > 0 else ""
        return f'<span class="score-flash {cls}">{label} {sign}{delta}</span>'

    score_html = (
        _score_badge(user_delta, "You") +
        _score_badge(bot_delta,  "Bot")
    )
    overrule_badge = '<span class="badge" style="background:#3a0808;color:#ff8a8a;border:1px solid #6a2020">🔨 OVERRULED</span> ' if is_overruled else ""
    badges = overrule_badge + build_badges(severity, memories_used, strategies_used, concession_detected)

    st.session_state.display_messages.append({
        "role"      : "assistant",
        "content"   : ai_message.content,
        "badges"    : badges,
        "score_html": score_html,
    })


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def render_echo_chamber():
    """
    Renders the Echo Chamber (Self-Play) tab.
    Uses rerun-per-round pattern: each Streamlit rerun processes one round,
    appends it to echo_messages, and reruns again until max_rounds reached.
    This gives the 'live transcript' feel without threading.
    """
    st.markdown(
        """
        <div class="masthead">
            <span class="masthead-title">🤖 Echo Chamber</span>
            <span class="masthead-sub">Autonomous AI Self-Play · Proponent vs Advocate</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Setup controls (only shown when not running) ──────────────────────────
    if not st.session_state.echo_running and not st.session_state.echo_messages:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-title">Watch two AIs debate autonomously.</div>
                <div class="welcome-text">
                    The <span style="color:#4a90d9">Proponent</span> defends your chosen topic.
                    The <span style="color:#e63946">Devil's Advocate</span> attacks it.<br><br>
                    Every claim is fact-checked. Every response is reviewed by the Judge.
                    After all rounds, a verdict is declared.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        topic = st.text_input(
            "Debate Topic",
            placeholder='e.g. "Artificial intelligence will create more jobs than it destroys"',
            key="echo_topic_input",
        )
        rounds = st.slider("Number of Rounds", min_value=2, max_value=8, value=5, key="echo_rounds_slider")

        if st.button("▶  START SIMULATION", key="echo_start_btn"):
            if not topic.strip():
                st.warning("Please enter a debate topic first.")
            else:
                st.session_state.echo_topic        = topic.strip()
                st.session_state.echo_max_rounds   = rounds
                st.session_state.echo_messages     = []
                st.session_state.echo_history      = []
                st.session_state.echo_running      = True
                st.session_state.echo_current_round = 1
                st.session_state.echo_pro_score    = 0
                st.session_state.echo_adv_score    = 0
                st.session_state.echo_verdict      = ""
                st.rerun()
        return

    # ── Show topic banner ──────────────────────────────────────────────────────
    st.markdown(
        f'<div class="echo-topic-banner">📌 TOPIC: {st.session_state.echo_topic}</div>',
        unsafe_allow_html=True,
    )

    # ── Render completed rounds ───────────────────────────────────────────────
    for result in st.session_state.echo_messages:
        _render_echo_round(result)

    # ── Run next round if simulation is active ────────────────────────────────
    if st.session_state.echo_running:
        current = st.session_state.echo_current_round
        max_r   = st.session_state.echo_max_rounds

        if current <= max_r:
            # Show live status
            st.markdown(
                f'<div class="echo-running-badge">⚡ ROUND {current}/{max_r} PROCESSING...</div>',
                unsafe_allow_html=True,
            )

            with st.spinner(f"Round {current} of {max_r} — agents debating..."):
                result = run_self_play_round(
                    llm        = st.session_state.llm,
                    fast_llm   = st.session_state.fast_llm,
                    topic      = st.session_state.echo_topic,
                    round_num  = current,
                    history    = st.session_state.echo_history,   # mutated in-place
                    pro_score  = st.session_state.echo_pro_score,
                    adv_score  = st.session_state.echo_adv_score,
                )

            # Save result
            st.session_state.echo_messages.append(result)
            st.session_state.echo_pro_score    = result.proponent_score
            st.session_state.echo_adv_score    = result.advocate_score
            st.session_state.echo_current_round += 1

            # Trigger next round
            st.rerun()

        else:
            # All rounds done
            st.session_state.echo_running = False

            # Generate verdict
            with st.spinner("Generating final verdict..."):
                st.session_state.echo_verdict = generate_self_play_verdict(
                    llm       = st.session_state.llm,
                    topic     = st.session_state.echo_topic,
                    history   = st.session_state.echo_history,
                    pro_score = st.session_state.echo_pro_score,
                    adv_score = st.session_state.echo_adv_score,
                )
            st.rerun()

    # ── Final scoreboard and verdict ──────────────────────────────────────────
    if not st.session_state.echo_running and st.session_state.echo_messages:
        pro = st.session_state.echo_pro_score
        adv = st.session_state.echo_adv_score

        if pro > adv:
            winner_line = "🟦 PROPONENT WINS"
            winner_color = "#4a90d9"
        elif adv > pro:
            winner_line = "🔴 ADVOCATE WINS"
            winner_color = "#e63946"
        else:
            winner_line = "🤝 DRAW"
            winner_color = "#888"

        st.markdown(
            f"""
            <div class="echo-scoreboard">
                <div class="echo-score-pro">
                    <span class="echo-score-num-pro">{pro}</span>
                    <span class="echo-score-lbl">🟦 Proponent</span>
                </div>
                <div style="text-align:center">
                    <div class="echo-vs">VS</div>
                    <div style="font-family:var(--mono);font-size:0.6rem;color:{winner_color};margin-top:0.3rem;font-weight:700">{winner_line}</div>
                </div>
                <div class="echo-score-adv">
                    <span class="echo-score-num-adv">{adv}</span>
                    <span class="echo-score-lbl">🔴 Advocate</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.echo_verdict:
            with st.expander("📋 SIMULATION VERDICT", expanded=True):
                st.markdown(
                    f'<div class="echo-verdict">{st.session_state.echo_verdict}</div>',
                    unsafe_allow_html=True,
                )

        # ── Proponent Weakness Radar ──────────────────────────────────────────
        if st.session_state.echo_messages:
            pro_profile = _build_echo_weakness_profile(st.session_state.echo_messages)
            if any(v > 0 for v in pro_profile.values()):
                st.markdown("<hr style='border-color:#2a2a2a;margin:1rem 0'>", unsafe_allow_html=True)
                render_weakness_radar(pro_profile, title="Proponent Weakness Radar")

        if st.button("↺  RUN NEW SIMULATION", key="echo_reset_btn"):
            for key in ["echo_topic", "echo_messages", "echo_history", "echo_running",
                        "echo_current_round", "echo_pro_score", "echo_adv_score", "echo_verdict"]:
                st.session_state[key] = [] if "messages" in key or "history" in key else (
                    False if key == "echo_running" else
                    0 if "score" in key or "round" in key else ""
                )
            st.rerun()


def _render_echo_round(result: SelfPlayTurnResult):
    """Renders a single completed round in the Echo Chamber transcript."""
    import html as _html
    st.markdown(
        f'<div class="round-divider">── Round {result.round_num} ──</div>',
        unsafe_allow_html=True,
    )

    # Strip any HTML tags the LLM may have output, then escape remaining chars
    def _clean(text: str) -> str:
        """Remove HTML tags entirely, then escape special chars. Belt + suspenders."""
        tag_stripped = re.sub(r"<[^>]*>", "", text)
        return _html.escape(tag_stripped)

    pro_body   = _clean(result.proponent_msg).replace("\n", "<br>")
    adv_body   = _clean(result.advocate_msg).replace("\n", "<br>")
    pro_reason = _clean(result.score_reason_pro)
    adv_reason = _clean(result.score_reason_adv)
    j_reason   = _clean(result.judge_reasoning[:120]) if result.judge_reasoning else ""

    # Proponent message
    fact_meta = ""
    if result.fact_performed:
        icon = {"ACCURATE": "✅", "INACCURATE": "❌", "UNVERIFIABLE": "⚠️"}.get(result.fact_verdict, "🔍")
        finding_safe = _html.escape(result.fact_finding[:120]) if result.fact_finding else ""
        fact_meta = f'<span>{icon} Fact: {result.fact_verdict}</span>'
        if finding_safe:
            fact_meta += f' · <span style="font-style:italic">{finding_safe}...</span>'

    pro_delta_html = (
        f'<span style="color:#4caf50">+{result.proponent_delta}</span>'
        if result.proponent_delta >= 0
        else f'<span style="color:#e57373">{result.proponent_delta}</span>'
    )

    st.markdown(
        f"""
        <div class="echo-proponent">
            <div class="echo-header echo-header-pro">🟦 PROPONENT</div>
            <div class="echo-body">{pro_body}</div>
            <div class="echo-meta">
                <span>Score: {result.proponent_score} ({pro_delta_html})</span>
                <span>{pro_reason}</span>
                {fact_meta}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Advocate message
    judge_meta = (
        '<span style="color:#e63946;font-weight:600">🔨 OVERRULED</span>'
        if result.is_overruled
        else '<span style="color:#2a9d5c">⚖️ UPHELD</span>'
    )

    adv_delta_html = (
        f'<span style="color:#4caf50">+{result.advocate_delta}</span>'
        if result.advocate_delta >= 0
        else f'<span style="color:#e57373">{result.advocate_delta}</span>'
    )

    overrule_note = (
        f'<span style="font-style:italic">{j_reason}...</span>'
        if result.is_overruled and j_reason else ""
    )

    st.markdown(
        f"""
        <div class="echo-advocate">
            <div class="echo-header echo-header-adv">🔴 DEVIL'S ADVOCATE</div>
            <div class="echo-body">{adv_body}</div>
            <div class="echo-meta">
                <span>Score: {result.advocate_score} ({adv_delta_html})</span>
                <span>{adv_reason}</span>
                {judge_meta}
                {overrule_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    init_session()

    # ── Tab layout — Debate mode | Echo Chamber ───────────────────────────────
    tab_debate, tab_echo = st.tabs(["💬 Live Debate", "🤖 Echo Chamber"])

    with tab_debate:
        render_sidebar()
        render_chat()
        user_input = st.chat_input("Make a claim and defend it...", key="chat_input")
        if user_input and user_input.strip():
            with st.spinner("🔬 Auditing logic and calculating scores..."):
                process_input(user_input.strip())
            st.rerun()

    with tab_echo:
        render_echo_chamber()

if __name__ == '__main__':
    main()
