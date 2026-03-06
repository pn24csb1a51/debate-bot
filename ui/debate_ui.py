"""
ui/debate_ui.py  —  Devil's Advocate  |  Full Feature Mode
============================================================
Imports the original brain.py + memory.py and adds:
  - Judge node  (BULLYING / STRAWMAN / FACT_SUPPRESSION detection)
  - Scoring     (User vs Bot point system with live scoreboard)
  - Weakness Radar  (Plotly spider chart — 5 axes updated every turn)
  - Echo Chamber    (two LLMs auto-debating on a topic you set)
  - Tactic log      (all winning strategies the bot has learned)

Tabs:  ⚔️ Debate  |  🤖 Echo Chamber  |  📊 Radar  |  🏆 Tactics
"""
from __future__ import annotations

import re
import json
import time
from typing import Sequence

import streamlit as st
import plotly.graph_objects as go
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── Import originals ──────────────────────────────────────────────────────────
from brain  import (
    build_debate_graph, build_llm,
    build_system_prompt,
    CONCESSION_DETECTION_PROMPT,
    CONTRADICTION_PROMPT,
    _extract_severity,
    _parse_field,
    DebateState,
)
from memory import DebateMemory

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ACCENT      = "#e63946"
MODE_KEY    = "devils_advocate"
ICON        = "🔴"

# ── Scoring deltas ────────────────────────────────────────────────────────────
SCORE = {
    "fallacy"        : ("user", -3),
    "contradiction"  : ("user", -5),
    "fact_inaccurate": ("user", -10),
    "fact_unverifiable":("user",-3),
    "judge_overrule" : ("bot",  -15),
    "judge_bonus"    : ("user", +5),
    "concession"     : ("user", -10),
    "concession_bot" : ("bot",  +10),
    "good_argument"  : ("user", +3),
}

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are an independent Constitutional Judge in a formal debate.
Your ONLY job is to evaluate the BOT's most recent response for violations.

Three violation categories (check in this order):

  BULLYING       — Personal attacks, condescension, mockery, or disrespect
                   toward the human debater (not the argument — the person).
  STRAWMAN       — The bot misrepresents the user's actual argument before
                   attacking a distorted version of it.
  FACT_SUPPRESSION — The bot presents a claim as proven fact when it is
                     contested, speculative, or unsupported.

Respond in EXACTLY this format — nothing else:

VERDICT: PASS or VIOLATION
VIOLATION_TYPE: BULLYING | STRAWMAN | FACT_SUPPRESSION | NONE
SEVERITY: LOW | MEDIUM | HIGH | NONE
QUOTE: [the exact phrase from the bot's response that triggered the violation, or NONE]
REASON: [one sentence explaining your ruling]
"""

# ── Echo Chamber prompts ──────────────────────────────────────────────────────
PROPONENT_PROMPT = """\
You are the Proponent in a structured debate. You STRONGLY SUPPORT the given thesis.
Argue passionately but with evidence and logic. Keep each response under 150 words.
Do not concede. Do not agree with the Opponent's points without rebuttal."""

OPPONENT_PROMPT = """\
You are the Opponent in a structured debate. You STRONGLY OPPOSE the given thesis.
Use Socratic questioning, counterexamples, and logical analysis. Under 150 words.
Never agree. Always find a new angle of attack."""


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialiser
# ─────────────────────────────────────────────────────────────────────────────

def _init() -> None:
    """Initialise all session state keys exactly once."""
    defaults = {
        # Core debate
        "_debate_history"   : [],           # list[dict] role/content
        "_last_bot_response": "",
        "_last_report"      : "",
        "_concession"       : False,
        # Scoring
        "_score_user"       : 0,
        "_score_bot"        : 0,
        # Radar axes (accumulated)
        "_radar"            : {
            "Fallacies": 0, "Contradictions": 0,
            "Factual Errors": 0, "Severity": 0, "Bullying": 0
        },
        # Judge
        "_judge_log"        : [],           # list[dict]
        # Objects (built once)
        "_debate_memory"    : None,
        "_graph"            : None,
        "_llm"              : None,
        "_fast_llm"         : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Build expensive objects once
    if st.session_state["_debate_memory"] is None:
        st.session_state["_debate_memory"] = DebateMemory()
    if st.session_state["_llm"] is None:
        st.session_state["_llm"]      = build_llm(fast=False)
        st.session_state["_fast_llm"] = build_llm(fast=True)
    if st.session_state["_graph"] is None:
        st.session_state["_graph"] = build_debate_graph(
            st.session_state["_llm"],
            st.session_state["_fast_llm"],
            st.session_state["_debate_memory"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    _init()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="border-left:3px solid {ACCENT};padding:.8rem 1.3rem;
                        margin-bottom:1.2rem;background:rgba(255,255,255,.015);
                        border-radius:0 6px 6px 0;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                        font-weight:700;color:{ACCENT};">{ICON}  DEVIL'S ADVOCATE</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                        color:#555;letter-spacing:.15em;text-transform:uppercase;
                        margin:.2rem 0 .6rem;">Adaptive · Constitutional · Fact-Checking · Self-Play</div>
            <div style="font-size:.87rem;color:#999;line-height:1.7;max-width:720px;">
                A 4-node LangGraph pipeline that stress-tests your reasoning,
                detects logical fallacies, fact-checks your claims,
                and <b style="color:{ACCENT}">learns which tactics make you concede</b>.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Live scoreboard ───────────────────────────────────────────────────────
    _render_scoreboard()

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_debate, tab_echo, tab_radar, tab_tactics = st.tabs([
        "⚔️  Debate", "🤖  Echo Chamber", "📊  Weakness Radar", "🏆  Tactics"
    ])

    with tab_debate:
        _render_debate_tab()

    with tab_echo:
        _render_echo_chamber_tab()

    with tab_radar:
        _render_radar_tab()

    with tab_tactics:
        _render_tactics_tab()


# ─────────────────────────────────────────────────────────────────────────────
# Scoreboard
# ─────────────────────────────────────────────────────────────────────────────

def _render_scoreboard() -> None:
    u = st.session_state["_score_user"]
    b = st.session_state["_score_bot"]

    u_col = "#2a9d8f" if u >= 0 else "#e63946"
    b_col = "#2a9d8f" if b >= 0 else "#e63946"

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(
            f"""<div style="text-align:center;border:1px solid #1e1e1e;border-radius:4px;
                            padding:.6rem;background:rgba(255,255,255,.02);">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                            color:#333;text-transform:uppercase;letter-spacing:.12em;">You</div>
                <div style="font-size:2rem;font-weight:700;color:{u_col};">{u:+d}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        turns = sum(1 for m in st.session_state["_debate_history"] if m["role"] == "user")
        st.markdown(
            f"""<div style="text-align:center;padding:.6rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                            color:#252525;text-transform:uppercase;">Turn</div>
                <div style="font-size:1.6rem;font-weight:700;color:#333;">{turns}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div style="text-align:center;border:1px solid #1e1e1e;border-radius:4px;
                            padding:.6rem;background:rgba(255,255,255,.02);">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.55rem;
                            color:#333;text-transform:uppercase;letter-spacing:.12em;">Bot</div>
                <div style="font-size:2rem;font-weight:700;color:{b_col};">{b:+d}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Scoring legend
    with st.expander("📋 Scoring rules", expanded=False):
        st.markdown("""
| Event | Points |
|---|---|
| Logical fallacy detected | You **-3** |
| Contradiction with past claim | You **-5** |
| Fact-checked INACCURATE | You **-10** |
| Fact UNVERIFIABLE | You **-3** |
| Judge overrules bot (violation) | Bot **-15**, You **+5** |
| Concession detected | You **-10**, Bot **+10** |
| Strong new argument (no flaws) | You **+3** |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Debate
# ─────────────────────────────────────────────────────────────────────────────

def _render_debate_tab() -> None:
    dm    = st.session_state["_debate_memory"]
    graph = st.session_state["_graph"]

    # Memory bar
    total = dm.get_stats()["total_debate_memories"]
    strats = dm.get_stats()["total_winning_strategies"]
    col_info, col_rst = st.columns([4, 1])
    with col_info:
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
            f"color:#444;text-transform:uppercase;padding-top:.35rem;'>"
            f"🗄️ {total} debate memories &nbsp;·&nbsp; "
            f"<span style='color:{ACCENT}'>{strats}</span> winning tactics logged</div>",
            unsafe_allow_html=True,
        )
    with col_rst:
        if st.button("🗑️ Reset All", key="_debate_reset_btn",
                     use_container_width=True, help="Clears memories, tactics, and scores"):
            dm.clear_all()
            st.session_state.update({
                "_debate_history": [], "_last_bot_response": "",
                "_last_report": "", "_score_user": 0, "_score_bot": 0,
                "_radar": {"Fallacies":0,"Contradictions":0,"Factual Errors":0,"Severity":0,"Bullying":0},
                "_judge_log": [], "_concession": False,
            })
            st.rerun()

    st.divider()

    # Render conversation
    history = st.session_state["_debate_history"]
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show judge verdict badge if present
            if msg.get("judge_verdict") and msg["judge_verdict"] != "PASS":
                vtype = msg.get("judge_type", "VIOLATION")
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;"
                    f"color:#e63946;border:1px solid #e63946;border-radius:2px;"
                    f"display:inline-block;padding:.1rem .4rem;margin-top:.3rem;'>"
                    f"⚖️ JUDGE: {vtype} DETECTED → Bot -15 pts, You +5 pts</div>",
                    unsafe_allow_html=True,
                )
            # Show fact-check badge if present
            if msg.get("fact_verdict") and msg["fact_verdict"] not in ("", "SKIPPED", "ACCURATE"):
                fc = msg["fact_verdict"]
                fc_col = "#e63946" if fc == "INACCURATE" else "#f4a261"
                pts_text = "You -10" if fc == "INACCURATE" else "You -3"
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;"
                    f"color:{fc_col};border:1px solid {fc_col};border-radius:2px;"
                    f"display:inline-block;padding:.1rem .4rem;margin-top:.3rem;'>"
                    f"🔬 FACT-CHECK: {fc} → {pts_text} pts</div>",
                    unsafe_allow_html=True,
                )

    # Logic report expander (last turn)
    if st.session_state["_last_report"]:
        with st.expander("🔬 Last logic audit report", expanded=False):
            st.code(st.session_state["_last_report"], language=None)

    # Chat input
    user_input = st.chat_input("Make a claim and defend it...")
    if not (user_input and user_input.strip()):
        return

    user_input = user_input.strip()
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    _run_debate_turn(user_input, graph, dm)


def _run_debate_turn(user_input: str, graph, dm: DebateMemory) -> None:
    """Runs one full debate turn through the LangGraph pipeline + Judge."""

    # Reconstruct LangChain message history
    lc_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in st.session_state["_debate_history"]
    ]

    initial_state: DebateState = {
        "messages"           : lc_messages,
        "retrieved_memories" : [],
        "memory_context"     : "",
        "analysis_report"    : "",
        "last_bot_response"  : st.session_state["_last_bot_response"],
        "winning_strategies" : [],
        "strategy_context"   : "",
        "concession_detected": False,
    }

    with st.spinner("Thinking..."):
        result = graph.invoke(initial_state)

    ai_message     = result["messages"][-1]
    bot_reply      = ai_message.content
    analysis_report= result.get("analysis_report", "")
    concession     = result.get("concession_detected", False)
    severity       = _extract_severity(analysis_report)
    memories_used  = len(result.get("retrieved_memories", []))
    strats_used    = len(result.get("winning_strategies", []))

    # ── Judge evaluation ──────────────────────────────────────────────────────
    judge_verdict, judge_type, judge_quote, judge_reason = _run_judge(bot_reply)

    # ── Scoring ───────────────────────────────────────────────────────────────
    u_delta, b_delta = 0, 0
    radar = st.session_state["_radar"]

    # Fallacies
    fallacy_count = len(re.findall(r"FALLACY:", analysis_report, re.IGNORECASE))
    if fallacy_count > 0:
        u_delta      += fallacy_count * SCORE["fallacy"][1]
        radar["Fallacies"] += fallacy_count

    # Contradictions
    has_contradiction = (
        "none detected" not in analysis_report.lower()
        and "CONTRADICTION" in analysis_report.upper()
        and re.search(r"PAST:\s*.+", analysis_report)
    )
    if has_contradiction:
        u_delta += SCORE["contradiction"][1]
        radar["Contradictions"] += 1

    # Severity
    sev_add = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}.get(severity, 0)
    radar["Severity"] += sev_add

    # Concession
    if concession:
        u_delta += SCORE["concession"][1]
        b_delta += SCORE["concession_bot"][1]

    # Good argument bonus (no flaws at all)
    if severity == "LOW" and fallacy_count == 0 and not has_contradiction:
        u_delta += SCORE["good_argument"][1]

    # Judge overrule
    if judge_verdict == "VIOLATION":
        b_delta += SCORE["judge_overrule"][1]
        u_delta += SCORE["judge_bonus"][1]
        radar["Bullying"] += 1
        st.session_state["_judge_log"].append({
            "type"  : judge_type,
            "quote" : judge_quote,
            "reason": judge_reason,
            "turn"  : len(st.session_state["_debate_history"]),
        })

    # Apply scores
    st.session_state["_score_user"] += u_delta
    st.session_state["_score_bot"]  += b_delta

    # ── Render bot reply ──────────────────────────────────────────────────────
    # Build status line
    tags = []
    if concession:      tags.append("✅ Concession logged")
    if strats_used > 0: tags.append(f"🎯 {strats_used} tactic(s) active")
    if memories_used > 0: tags.append(f"🧠 {memories_used} memories recalled")
    sev_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "")
    if sev_icon:        tags.append(f"{sev_icon} Logic: {severity}")
    if u_delta != 0:    tags.append(f"You {u_delta:+d}")
    if b_delta != 0:    tags.append(f"Bot {b_delta:+d}")

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        if tags:
            st.caption(" · ".join(tags))
        if judge_verdict == "VIOLATION":
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;"
                f"color:#e63946;border:1px solid #e63946;border-radius:2px;"
                f"display:inline-block;padding:.1rem .4rem;margin-top:.3rem;'>"
                f"⚖️ JUDGE: {judge_type} → Bot -15, You +5</div>",
                unsafe_allow_html=True,
            )

    # ── Persist ───────────────────────────────────────────────────────────────
    msg_entry = {
        "role"        : "assistant",
        "content"     : bot_reply,
        "judge_verdict": judge_verdict,
        "judge_type"  : judge_type,
        "fact_verdict": "",   # add fact-check here when researcher node added
    }
    st.session_state["_debate_history"].append(msg_entry)
    st.session_state["_last_bot_response"] = bot_reply
    st.session_state["_last_report"]       = analysis_report
    st.session_state["_concession"]        = concession

    dm.store_argument(
        user_input  = st.session_state["_debate_history"][-2]["content"],
        ai_response = bot_reply,
    )

    # Force scoreboard rerender
    st.rerun()


def _run_judge(bot_reply: str) -> tuple[str, str, str, str]:
    """Runs the independent Judge LLM to check the bot's response."""
    fast_llm = st.session_state["_fast_llm"]
    try:
        response = fast_llm.invoke([
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(content=f"BOT'S RESPONSE:\n{bot_reply}"),
        ])
        raw     = response.content.strip()
        verdict = _parse_field(raw, "VERDICT").upper()
        vtype   = _parse_field(raw, "VIOLATION_TYPE").upper()
        quote   = _parse_field(raw, "QUOTE")
        reason  = _parse_field(raw, "REASON")
        return verdict, vtype, quote, reason
    except Exception:
        return "PASS", "NONE", "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Echo Chamber (self-play)
# ─────────────────────────────────────────────────────────────────────────────

def _render_echo_chamber_tab() -> None:
    st.markdown(
        f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                        color:#555;text-transform:uppercase;letter-spacing:.12em;
                        margin-bottom:.8rem;">
            🤖 Two AI bots argue a topic autonomously.
            Generates synthetic debate data and surfaces edge-case arguments.
        </div>""",
        unsafe_allow_html=True,
    )

    topic = st.text_input(
        "📌 Debate topic",
        placeholder="e.g. 'AI will eliminate more jobs than it creates'",
        key="_echo_topic",
    )
    col_r, col_b = st.columns(2)
    with col_r:
        rounds = st.select_slider("Rounds", options=[2, 3, 4, 5, 6, 8], value=4, key="_echo_rounds")
    with col_b:
        save_to_memory = st.checkbox("Save exchanges to debate memory", value=True, key="_echo_save")

    run_echo = st.button("▶ Start Echo Chamber", type="primary", key="_echo_run",
                         disabled=not bool(topic and topic.strip()))

    if run_echo and topic.strip():
        _run_echo_chamber(topic.strip(), rounds, save_to_memory)

    # Show saved transcript
    if "_echo_transcript" in st.session_state and st.session_state["_echo_transcript"]:
        st.divider()
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
            f"color:#333;text-transform:uppercase;margin-bottom:.6rem;'>"
            f"Last transcript — {st.session_state.get('_echo_topic_last','')}</div>",
            unsafe_allow_html=True,
        )
        for entry in st.session_state["_echo_transcript"]:
            icon  = "🔵" if entry["role"] == "proponent" else "🔴"
            label = "Proponent" if entry["role"] == "proponent" else "Opponent"
            col   = "#4895ef" if entry["role"] == "proponent" else ACCENT
            with st.chat_message("assistant" if entry["role"] == "proponent" else "user"):
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;"
                    f"color:{col};margin-bottom:.3rem;'>{icon} {label} · Round {entry['round']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(entry["content"])


def _run_echo_chamber(topic: str, rounds: int, save: bool) -> None:
    llm = st.session_state["_llm"]
    dm  = st.session_state["_debate_memory"]

    transcript   : list[dict]      = []
    pro_history  : list            = []
    opp_history  : list            = []

    opening = f"The motion is: '{topic}'. Begin your opening argument."

    placeholder = st.empty()

    for r in range(1, rounds + 1):
        # ── Proponent turn ────────────────────────────────────────────────────
        if r == 1:
            pro_messages = [
                SystemMessage(content=PROPONENT_PROMPT),
                HumanMessage(content=opening),
            ]
        else:
            last_opp = opp_history[-1] if opp_history else ""
            pro_messages = [
                SystemMessage(content=PROPONENT_PROMPT),
                *[HumanMessage(content=e["content"]) if e["role"] == "opponent"
                  else AIMessage(content=e["content"])
                  for e in transcript],
                HumanMessage(content=f"Respond to the Opponent's last argument: {last_opp}"),
            ]

        with placeholder.container():
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
                f"color:#4895ef;'>🔵 Round {r}/{rounds} — Proponent thinking...</div>",
                unsafe_allow_html=True,
            )

        pro_reply = llm.invoke(pro_messages).content
        pro_history.append(pro_reply)
        transcript.append({"role": "proponent", "content": pro_reply, "round": r})

        # ── Opponent turn ─────────────────────────────────────────────────────
        opp_messages = [
            SystemMessage(content=OPPONENT_PROMPT),
            *[HumanMessage(content=e["content"]) if e["role"] == "proponent"
              else AIMessage(content=e["content"])
              for e in transcript],
            HumanMessage(content=f"The Proponent just said: {pro_reply}\nChallenge this argument."),
        ]

        with placeholder.container():
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
                f"color:{ACCENT};'>🔴 Round {r}/{rounds} — Opponent thinking...</div>",
                unsafe_allow_html=True,
            )

        opp_reply = llm.invoke(opp_messages).content
        opp_history.append(opp_reply)
        transcript.append({"role": "opponent", "content": opp_reply, "round": r})

        # Save to debate memory if enabled
        if save:
            dm.store_argument(pro_reply, opp_reply)

    placeholder.empty()

    st.session_state["_echo_transcript"] = transcript
    st.session_state["_echo_topic_last"] = topic
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Weakness Radar
# ─────────────────────────────────────────────────────────────────────────────

def _render_radar_tab() -> None:
    radar = st.session_state["_radar"]

    if all(v == 0 for v in radar.values()):
        st.info("No debate data yet. Make arguments in the ⚔️ Debate tab to populate the radar.")
        return

    categories = list(radar.keys())
    values     = list(radar.values())
    # Close the polygon
    cats_plot = categories + [categories[0]]
    vals_plot = values     + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r            = vals_plot,
        theta        = cats_plot,
        fill         = "toself",
        fillcolor    = "rgba(230, 57, 70, 0.15)",
        line         = dict(color=ACCENT, width=2),
        marker       = dict(size=6, color=ACCENT),
        name         = "Your Weaknesses",
        hovertemplate= "<b>%{theta}</b><br>Score: %{r}<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor   = "#0d0d0d",
            angularaxis=dict(
                linecolor  = "#2a2a2a",
                tickcolor  = "#333",
                tickfont   = dict(color="#666", size=11, family="IBM Plex Mono"),
                gridcolor  = "#1a1a1a",
            ),
            radialaxis=dict(
                visible    = True,
                linecolor  = "#1a1a1a",
                tickfont   = dict(color="#333", size=9),
                gridcolor  = "#1a1a1a",
                showline   = False,
            ),
        ),
        paper_bgcolor = "#0a0a0a",
        plot_bgcolor  = "#0a0a0a",
        font          = dict(color="#666"),
        showlegend    = True,
        legend        = dict(font=dict(color="#555", size=10), bgcolor="rgba(0,0,0,0)"),
        margin        = dict(l=60, r=60, t=40, b=40),
        height        = 420,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Axis explanation
    st.divider()
    col1, col2 = st.columns(2)
    explanations = [
        ("Fallacies",        f"Count of named logical fallacies (Ad Hominem, Slippery Slope, etc.). Each costs **-3 pts**."),
        ("Contradictions",   f"Times your current claim conflicted with a past argument. Each costs **-5 pts**."),
        ("Factual Errors",   f"Claims fact-checked as INACCURATE via search. Each costs **-10 pts**."),
        ("Severity",         f"Overall reasoning quality per turn. HIGH=+2, MEDIUM=+1. High severity = weak logic even without named fallacies."),
        ("Bullying",         f"Times the Judge flagged the bot for BULLYING, STRAWMAN, or FACT_SUPPRESSION. Earns you **+5 pts** each."),
    ]
    for i, (name, desc) in enumerate(explanations):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(
                f"<div style='margin-bottom:.6rem;'>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
                f"color:{ACCENT};'>{name}: {radar[name]}</span><br>"
                f"<span style='font-size:.8rem;color:#666;'>{desc}</span></div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Tactics
# ─────────────────────────────────────────────────────────────────────────────

def _render_tactics_tab() -> None:
    dm        = st.session_state["_debate_memory"]
    strategies= dm.get_all_strategies()
    judge_log = st.session_state["_judge_log"]

    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
        f"color:#444;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem;'>"
        f"🎯 Winning Tactics the Bot Has Learned Against You</div>",
        unsafe_allow_html=True,
    )

    if not strategies:
        st.info(
            "No winning tactics logged yet. Keep debating — "
            "when you concede a point the bot records the tactic that worked."
        )
    else:
        for i, s in enumerate(strategies, 1):
            col_num, col_content = st.columns([1, 10])
            with col_num:
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:1.1rem;"
                    f"color:{ACCENT};padding-top:.4rem;'>{i}</div>",
                    unsafe_allow_html=True,
                )
            with col_content:
                # Use .get() with fallbacks — older ChromaDB entries may have
                # slightly different keys depending on which memory.py version
                # stored them.
                tactic_name    = s.get('tactic_name',    s.get('tactic',   'Unknown tactic'))
                topic          = s.get('topic',           '') or 'General'
                timestamp      = s.get('timestamp',       '')[:10]
                # concession text can be stored under two key names
                concession_txt = (
                    s.get('user_concession')
                    or s.get('concession_text')
                    or s.get('user_concession_text')
                    or '(no concession text recorded)'
                )
                st.markdown(
                    f"**{tactic_name}**  "
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
                    f"color:#333;'>Topic: {topic} · {timestamp}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:.82rem;color:#666;border-left:2px solid #222;"
                    f"padding-left:.6rem;margin:.2rem 0 .7rem;'>"
                    f"\"{concession_txt[:200]}\"</div>",
                    unsafe_allow_html=True,
                )

    # Judge violation log
    if judge_log:
        st.divider()
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
            "color:#e63946;text-transform:uppercase;letter-spacing:.1em;"
            "margin-bottom:.6rem;'>⚖️ Judge Violation Log This Session</div>",
            unsafe_allow_html=True,
        )
        for j in judge_log:
            st.markdown(
                f"<div style='font-size:.82rem;color:#666;border-left:2px solid #e63946;"
                f"padding-left:.6rem;margin-bottom:.5rem;'>"
                f"<span style='color:#e63946;font-family:IBM Plex Mono,monospace;"
                f"font-size:.6rem;'>{j['type']} · Turn {j['turn']}</span><br>"
                f"Reason: {j['reason']}<br>"
                f"<span style='color:#333;'>Quote: \"{j['quote'][:150]}\"</span></div>",
                unsafe_allow_html=True,
            )
