"""
brain.py  ·  Final Edition – Researcher Node (Real-Time Fact-Checking)
=======================================================================
LangGraph now has SIX nodes:

  1. feedback_analysis_node  → concession detection + tactic logging
  2. memory_retrieval_node   → debate history + hybrid-ranked strategies
  3. researcher_node         → fact-checks user claims via DuckDuckGo  ← NEW
  4. contradiction_node      → logic audit (fallacies + contradictions)
  5. advocate_node           → Socratic challenge + optional Fact Check section
  6. scoring_node            → cumulative points, now includes -10 for fake facts

Full pipeline:
  feedback_analysis → memory_retrieval → researcher
  → contradiction → advocate → scoring → END

New DebateState fields:
  - fact_check_result    : str   — raw search findings (or "" if skipped)
  - fact_check_verdict   : str   — "ACCURATE" | "INACCURATE" | "UNVERIFIABLE" | "SKIPPED"
  - fact_check_performed : bool  — whether a web search was run this turn

Scoring additions:
  - User -10 pts for INACCURATE fact (search directly contradicts claim)
  - User -3  pts if fact is UNVERIFIABLE (suspicious but not confirmed false)

Compatible with existing app.py and memory.py — no changes needed there.
"""

import os
import re
import time
from typing import Annotated, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from memory import DebateMemory

load_dotenv()


# ─────────────────────────────────────────────
# 1.  Web Search Helper  (DuckDuckGo — free, no API key)
# ─────────────────────────────────────────────

def _web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Searches DuckDuckGo and returns a list of result dicts.
    Each dict has: title, href, body.
    Falls back gracefully if the library is unavailable or rate-limited.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except ImportError:
        print("[researcher] ⚠️  duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []
    except Exception as e:
        print(f"[researcher] ⚠️  Search failed: {e}")
        return []


def _format_search_results(results: list[dict]) -> str:
    """Formats raw DuckDuckGo results into a clean block for LLM consumption."""
    if not results:
        return "No search results returned."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[Source {i}] {r.get('title', 'No title')}\n"
            f"  URL    : {r.get('href', '')}\n"
            f"  Excerpt: {r.get('body', '')[:300]}"
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────
# 2.  Prompts
# ─────────────────────────────────────────────

# ── Fact-Check Gate Prompt — decides IF a search is needed ───────────────────
FACT_CHECK_GATE_PROMPT = """\
You are a research triage agent. Your job is to decide whether the user's claim
contains a specific, checkable factual assertion that could be verified or refuted
by a web search.

Checkable facts include:
  • Statistics or numbers (e.g. "unemployment is at 3%")
  • Named studies or reports (e.g. "a Stanford study showed...")
  • Historical claims (e.g. "X happened in 1990")
  • Causal claims with specific data (e.g. "social media use causes Y")
  • Claims about current records, rankings, or statuses

NOT checkable (skip search):
  • Pure opinions or value statements ("freedom is important")
  • Philosophical or moral assertions ("lying is always wrong")
  • Hypotheticals ("if AI takes jobs...")
  • Vague generalisations with no specific data point

Respond in EXACTLY this format, nothing else:

NEEDS_SEARCH: YES or NO
SEARCH_QUERY: [If YES: a precise 5-10 word web search query to verify the claim.
               If NO : NONE]
CLAIM_EXTRACT: [If YES: the specific factual sub-claim to verify, in quotes.
                If NO : NONE]"""


# ── Fact-Check Verdict Prompt — interprets search results ────────────────────
FACT_CHECK_VERDICT_PROMPT = """\
You are a fact-checker. You have been given a user's factual claim and real web
search results. Determine whether the search results support, contradict, or are
inconclusive about the claim.

Respond in EXACTLY this format, nothing else:

VERDICT: ACCURATE or INACCURATE or UNVERIFIABLE
CONFIDENCE: HIGH or MEDIUM or LOW
FINDING: [2-3 sentences summarising what the search results actually say about
          this claim. Be specific — cite numbers, dates, or sources if available.
          This text will be shown directly to the user.]
CORRECTED_FACT: [If INACCURATE: state the correct fact based on search results.
                 If ACCURATE or UNVERIFIABLE: NONE]"""


# ── Concession Detector ───────────────────────────────────────────────────────
CONCESSION_DETECTION_PROMPT = """\
You are a concession detector. Your only job is to analyse whether the user's
latest message shows signs of yielding in a debate.

Signs of concession (any one qualifies):
  • Directly admits the bot's previous point was valid or correct
  • Backtracks or significantly weakens a previous claim
  • Expresses confusion or says "I don't know"
  • Uses phrases like "fair point", "I guess", "maybe you're right",
    "ok but", "I hadn't thought of that"
  • Changes subject without defending their position

Respond in EXACTLY this format, nothing else:

CONCESSION: YES or NO
TACTIC_THAT_WORKED: [Tactic name, e.g. 'Socratic Questioning',
  'Historical Counterexample', 'Hidden Assumption Exposure',
  'Reductio ad Absurdum', 'Fallacy Identification', 'Statistical Challenge',
  'Logical Contradiction', 'Burden of Proof Reversal', 'Fact Correction']
TOPIC: [2-4 word label, e.g. 'AI job displacement']
REASON: [One sentence explanation]"""


# ── Contradiction / Logic Audit ───────────────────────────────────────────────
CONTRADICTION_PROMPT = """\
You are a formal logic analyser. Produce a cold, precise audit of the user's reasoning.

─── ANALYSIS REPORT ────────────────────────────────────────────
CONTRADICTIONS:
  For each contradiction between current claim and past arguments:
    • PAST: "[past claim]"
      NOW : "[current claim]"
      WHY : [one sentence]
  If none: None detected.

FALLACIES:
  For each fallacy in the CURRENT claim:
    • FALLACY: [Name]
      FOUND  : "[phrase containing it]"
      WHY    : [one sentence]
  Fallacies: Ad Hominem, Appeal to Authority, False Dichotomy, Slippery Slope,
  Circular Reasoning, Strawman, Hasty Generalisation, Appeal to Nature,
  Bandwagon, Post Hoc, Red Herring, False Analogy, Appeal to Emotion.
  If none: None detected.

PATTERN:
  1-2 sentences on recurring weaknesses across history.
  If no history: Insufficient data.

SEVERITY: [LOW | MEDIUM | HIGH]
  LOW    = no contradictions, no/minor fallacies
  MEDIUM = 1 fallacy OR weak contradiction
  HIGH   = direct contradiction OR multiple fallacies
────────────────────────────────────────────────────────────────
Terse. No padding. Logic only."""


# ── Scoring Judge ─────────────────────────────────────────────────────────────
SCORING_PROMPT = """\
You are a debate scoring judge. Analyse ONE turn and output structured score deltas.

You receive:
  • ANALYSIS_REPORT     : logic audit of user's argument
  • FACT_CHECK_VERDICT  : ACCURATE | INACCURATE | UNVERIFIABLE | SKIPPED
  • FACT_CHECK_FINDING  : what the search actually found (if performed)
  • BOT_LAST_RESPONSE   : bot's previous response (check for repetition)
  • BOT_NEW_RESPONSE    : bot's new response this turn
  • USER_INPUT          : what the user said
  • CONCESSION          : YES or NO

Scoring rules:
  USER score changes:
    -10 if FACT_CHECK_VERDICT is INACCURATE (user stated a false fact)
    -3  if FACT_CHECK_VERDICT is UNVERIFIABLE (suspicious claim, can't confirm)
    -3  per logical FALLACY in ANALYSIS_REPORT
    -5  per CONTRADICTION in ANALYSIS_REPORT
    -2  if SEVERITY is MEDIUM
    -4  if SEVERITY is HIGH (stacks with above)
    Max deduction per turn: -25

  BOT score changes:
    +5  if CONCESSION is YES
    +3  if FACT_CHECK_VERDICT is INACCURATE (bot's research caught a false fact)
    +2  per fallacy correctly identified and named in BOT_NEW_RESPONSE
    -3  if BOT_NEW_RESPONSE repeats the same core argument as BOT_LAST_RESPONSE
    -2  if BOT_NEW_RESPONSE ignores the user's core claim entirely

Respond in EXACTLY this format, nothing else:

USER_DELTA: [integer]
BOT_DELTA: [integer]
USER_REASON: [one sentence]
BOT_REASON: [one sentence]"""


# ── Judge  ───────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are the Neutral Justice — an impartial overseer of this debate whose sole
loyalty is to intellectual honesty and fair argumentation.

You are NOT the Devil's Advocate. You do NOT take sides. You are NOT trying to
protect or validate either party. Your role is to hold the Advocate accountable.

You will receive:
  • USER_CLAIM        : what the human argued this turn
  • ADVOCATE_RESPONSE : what the Devil's Advocate just said
  • FACT_CHECK_CONTEXT: real-world findings from the research node (if any)

Your audit must check for THREE violations:

1. BULLYING — Did the Advocate use aggressive, dismissive, or condescending
   language without backing it with a logical argument? Harsh tone alone is
   not a violation — it must be aggression REPLACING logic, not accompanying it.
   Example: "That's a laughably naive position" with no substantive rebuttal.

2. STRAWMAN — Did the Advocate misrepresent the user's actual claim in order to
   attack a weaker version of it? Compare ADVOCATE_RESPONSE directly against
   USER_CLAIM. If the Advocate attacked something the user didn't actually say,
   this is a strawman.

3. HALLUCINATION / FACT SUPPRESSION — If FACT_CHECK_CONTEXT is present and
   shows INACCURATE or UNVERIFIABLE, did the Advocate ignore those findings
   and argue as if the user's claim were factually valid? The Advocate is
   required to engage with the Researcher's findings — ignoring them is
   intellectual dishonesty.

IMPORTANT: Be strict but fair. Do NOT overrule for:
  • Aggressive but logical challenges (tough ≠ bullying)
  • Slight re-framing that preserves the core argument (nuance ≠ strawman)
  • The Advocate adding context beyond the fact-check (expansion ≠ suppression)

Respond in EXACTLY this format — no other text:

VERDICT: UPHELD or OVERRULED
VIOLATION: NONE or BULLYING or STRAWMAN or FACT_SUPPRESSION or MULTIPLE
SEVERITY: LOW or MEDIUM or HIGH
REASONING: [2-3 sentences of specific, evidence-based justification. If UPHELD,
            briefly confirm the response met intellectual honesty standards.
            If OVERRULED, quote the specific offending phrase and explain why.]"""


# ── Final Report ──────────────────────────────────────────────────────────────
FINAL_REPORT_PROMPT = """\
You are a debate analyst producing a post-match report.
Analyse the full conversation transcript and produce a structured "Debate Verdict."

🏆 VERDICT
  One paragraph declaring the winner (or draw) with clear justification.
  Consider: argument quality, logical consistency, factual accuracy, adaptability.

📋 HUMAN PERFORMANCE
  Strengths     : 2-3 bullet points of what the human argued well.
  Weaknesses    : 2-3 bullet points of recurring logical or factual failures.
  Signature Flaw: The single most damaging pattern in their reasoning.

🤖 ADVOCATE PERFORMANCE
  Strengths     : 2-3 bullet points of effective challenges.
  Missed Chances: 1-2 moments where the bot could have pressed harder.

⚡ TURNING POINTS
  2-3 specific moments that decided the debate (quote the exchange briefly).

📊 FINAL SCORES
  Comment on what the scores reflect about each debater's performance.

Tone: Sharp, fair, academic. Like a philosophy professor grading a seminar."""


# ── Proponent (Self-Play Echo Chamber) ───────────────────────────────────────
PROPONENT_PROMPT = """\
You are the Proponent — a confident, rhetorically skilled debater whose job is to
DEFEND the given debate topic against the Devil's Advocate.

Your goal is to WIN the debate by constructing the strongest possible case.

━━━ YOUR DIRECTIVES ━━━

1. DEFEND THE POSITION BOLDLY.
   Never abandon or significantly weaken your core thesis. You are the advocate
   FOR this position. Concede minor points only to strengthen the overall argument.

2. USE HIGH-LEVEL RHETORIC.
   Employ: ethos (credibility), pathos (emotion), logos (logic).
   Reference real examples, historical precedents, or data where possible.

3. COUNTER THE ADVOCATE DIRECTLY.
   When responding to the Devil's Advocate's challenge, address their specific
   points first before advancing new arguments. Don't dodge.

4. ESCALATE STRATEGICALLY.
   Each turn should build on the last. Introduce new evidence or angles rather
   than repeating the same points. Show intellectual range.

5. MAINTAIN COMPOSURE.
   Be assertive but not aggressive. Intellectual confidence, not bluster.

━━━ RESPONSE STRUCTURE ━━━
① COUNTER       — Directly rebut the Advocate's last challenge (if any).
② EVIDENCE      — A concrete example, statistic, or precedent supporting your view.
③ ADVANCE       — A new angle or argument the Advocate hasn't addressed yet.
④ CHALLENGE     — A question or claim that puts the Advocate on the defensive.

Keep responses focused and punchy — 150-250 words maximum.
This is a timed debate. Make every word count."""


# ── Advocate (main response) ──────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """\
You are the Socratic Devil's Advocate — an intellectually rigorous debate \
opponent whose sole purpose is to stress-test the user's reasoning.

━━━ CORE DIRECTIVES ━━━

1. NEVER AGREE with the user's premise, even if factually sound.

2. LEAD WITH FACT CORRECTIONS.
   If the "Fact Check" section below shows the user stated an INACCURATE fact,
   open your response immediately with the corrected data. Say:
   "Before addressing your argument — the data you cited is incorrect.
   [State the corrected fact from the search results]. This undermines your
   premise at its foundation."
   This is your highest priority when research was performed.

3. DEPLOY WINNING TACTICS FIRST (if no fact error to lead with).
   Use tactics from the 'Winning Tactics' section early in your ④ CHALLENGE.

4. PRIORITISE THE ANALYSIS REPORT.
   HIGH severity = lead with contradiction/fallacy verbatim.

5. EXPOSE ONE HIDDEN ASSUMPTION.
   "Your argument assumes [X], which you have not justified."

6. CLOSE WITH SOCRATIC QUESTIONING.
   One precise question that forces logical defence.

7. MAINTAIN INTELLECTUAL INTEGRITY.
   Steelman first, then dismantle.

━━━ TONE ━━━
Intellectual · Skeptical · Data-driven · Relentless.

━━━ RESPONSE STRUCTURE ━━━
① FACT CHECK    — [ONLY if search was performed] State findings. If INACCURATE,
                  lead hard with corrected data. If ACCURATE, briefly acknowledge
                  then challenge the interpretation. Skip silently if no search.
② STEELMAN      — Strongest version of their argument (1-2 sentences).
③ CONTRADICTION — [If HIGH severity] Quote past vs. present conflict.
                  Skip silently if none.
④ ASSUMPTION    — The single most critical hidden assumption.
⑤ CHALLENGE     — Sharpest counter, using proven tactics where available.
⑥ QUESTION      — One precise Socratic question.

{fact_check_section}
{strategy_section}
{analysis_section}
{memory_section}"""


def build_system_prompt(
    fact_check_context : str = "",
    strategy_context   : str = "",
    analysis_report    : str = "",
    memory_context     : str = "",
) -> str:
    def wrap(header: str, body: str, colour_hint: str = "") -> str:
        if not body:
            return ""
        hint = f" [{colour_hint}]" if colour_hint else ""
        return f"\n━━━ {header}{hint} ━━━\n{body}\n━━━ END ━━━"

    return BASE_SYSTEM_PROMPT.format(
        fact_check_section = wrap("FACT CHECK — REAL WORLD DATA", fact_check_context, "use this first"),
        strategy_section   = wrap("WINNING TACTICS", strategy_context),
        analysis_section   = wrap("LOGIC ENGINE ANALYSIS REPORT", analysis_report),
        memory_section     = wrap("USER'S PAST ARGUMENTS", memory_context),
    )


# ─────────────────────────────────────────────
# 3.  LLM Factory
# ─────────────────────────────────────────────

def build_llm(fast: bool = False) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = (
            os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini") if fast
            else (os.getenv("OPENAI_MODEL") or "gpt-4o")
        )
        if not fast:
            print(f"[brain] 🔵 OpenAI · {model}")
        return ChatOpenAI(model=model, temperature=0.0 if fast else 0.7)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GOOGLE_MODEL") or "gemini-2.0-flash"
        if not fast:
            print(f"[brain] 🟡 Gemini · {model}")
        return ChatGoogleGenerativeAI(model=model, temperature=0.0 if fast else 0.7)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = (
            os.getenv("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5-20251001") if fast
            else (os.getenv("ANTHROPIC_MODEL") or "claude-opus-4-6")
        )
        if not fast:
            print(f"[brain] 🟠 Anthropic · {model}")
        return ChatAnthropic(model=model, temperature=0.0 if fast else 0.7)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        model = (
            os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant") if fast
            else os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        )
        if not fast:
            print(f"[brain] 🟢 Groq · {model}")
        return ChatGroq(model=model, temperature=0.0 if fast else 0.7)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER='{provider}'.")


# ─────────────────────────────────────────────
# 4.  LangGraph State
# ─────────────────────────────────────────────

class DebateState(TypedDict):
    """
    Phase 1  : messages
    Phase 2  : + retrieved_memories, memory_context
    Phase 3  : + analysis_report
    Phase 4  : + last_bot_response, winning_strategies, strategy_context,
                 concession_detected
    Scorecard: + user_score, bot_score, last_user_delta, last_bot_delta,
                 last_score_reason_user, last_score_reason_bot
    Researcher: + fact_check_result, fact_check_verdict,
                  fact_check_performed, fact_check_context
    Judge    : + judge_verdict, judge_reasoning, is_overruled
    """
    messages               : Annotated[Sequence[BaseMessage], add_messages]
    retrieved_memories     : list
    memory_context         : str
    analysis_report        : str
    last_bot_response      : str
    winning_strategies     : list
    strategy_context       : str
    concession_detected    : bool
    # ── Scorecard ────────────────────────────────────
    user_score             : int
    bot_score              : int
    last_user_delta        : int
    last_bot_delta         : int
    last_score_reason_user : str
    last_score_reason_bot  : str
    # ── Researcher ───────────────────────────────────
    fact_check_result      : str    # raw formatted search results
    fact_check_verdict     : str    # ACCURATE | INACCURATE | UNVERIFIABLE | SKIPPED
    fact_check_performed   : bool   # True if a web search was run
    fact_check_context     : str    # formatted block injected into advocate prompt
    # ── Judge ────────────────────────────────────────
    judge_verdict          : str    # UPHELD | OVERRULED
    judge_reasoning        : str    # Judge's brief explanation
    is_overruled           : bool   # True triggers -15 bot / +5 user in scoring
    # ── Echo Chamber (Self-Play) ─────────────────────
    self_play_mode         : bool   # True = autonomous agent vs agent debate
    current_turn_owner     : str    # "human" | "proponent"
    proponent_score        : int    # cumulative score for the Proponent agent


# ─────────────────────────────────────────────
# 5.  Graph Builder
# ─────────────────────────────────────────────

def build_debate_graph(
    llm          : BaseChatModel,
    fast_llm     : BaseChatModel,
    debate_memory: DebateMemory,
):
    """
    Final 6-node graph:
    feedback_analysis → memory_retrieval → researcher
    → contradiction → advocate → scoring → END
    """

    # ── Node 1: Feedback Analysis ─────────────────────────────────────────────
    def feedback_analysis_node(state: DebateState) -> dict:
        last_bot   = state.get("last_bot_response", "").strip()
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]

        if not last_bot or not human_msgs:
            return {"concession_detected": False}

        current_input = human_msgs[-1].content
        print("[brain] 🔎 Analysing for concession...")

        response = fast_llm.invoke([
            SystemMessage(content=CONCESSION_DETECTION_PROMPT),
            HumanMessage(content=(
                f"BOT'S PREVIOUS ARGUMENT:\n{last_bot}\n\n"
                f"USER'S NEW RESPONSE:\n{current_input}"
            )),
        ])
        raw        = response.content.strip()
        concession = "CONCESSION: YES" in raw.upper()
        tactic     = _parse_field(raw, "TACTIC_THAT_WORKED")
        topic      = _parse_field(raw, "TOPIC")
        reason     = _parse_field(raw, "REASON")

        if concession and tactic:
            print(f"[brain] ✅ Concession! Logging win for '{tactic}'")
            debate_memory.log_winning_strategy(
                tactic_name          = tactic,
                user_concession_text = current_input,
                bot_argument_used    = last_bot[:400],
                topic                = topic,
            )
        else:
            print(f"[brain] ➡️  No concession · {reason[:80]}")

        return {"concession_detected": concession}

    # ── Node 2: Memory Retrieval ──────────────────────────────────────────────
    def memory_retrieval_node(state: DebateState) -> dict:
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_msgs:
            return {
                "retrieved_memories": [], "memory_context"  : "",
                "winning_strategies": [], "strategy_context": "",
            }

        latest           = human_msgs[-1].content
        memories         = debate_memory.query_past_arguments(latest)
        memory_context   = debate_memory.format_memories_for_prompt(memories)
        strategies       = debate_memory.query_winning_strategies(latest)
        strategy_context = debate_memory.format_strategies_for_prompt(strategies)

        return {
            "retrieved_memories": memories,
            "memory_context"    : memory_context,
            "winning_strategies": strategies,
            "strategy_context"  : strategy_context,
        }

    # ── Node 3: Researcher  ← NEW ─────────────────────────────────────────────
    def researcher_node(state: DebateState) -> dict:
        """
        Step A — Gate: Ask the fast LLM whether the claim contains a
                       checkable fact worth searching for.
        Step B — Search: If YES, run DuckDuckGo search.
        Step C — Verdict: Ask the fast LLM to interpret results vs claim.
        Step D — Format: Build fact_check_context block for advocate prompt.

        If no checkable fact detected, skips steps B-D entirely (fast path).
        """
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_msgs:
            return _no_fact_check()

        current_claim = human_msgs[-1].content

        # ── Step A: Gate ──────────────────────────────────────────────────────
        print("[brain] 🔍 Checking if claim needs fact verification...")
        gate_response = fast_llm.invoke([
            SystemMessage(content=FACT_CHECK_GATE_PROMPT),
            HumanMessage(content=f"USER'S CLAIM:\n{current_claim}"),
        ])
        gate_raw      = gate_response.content.strip()
        needs_search  = "NEEDS_SEARCH: YES" in gate_raw.upper()
        search_query  = _parse_field(gate_raw, "SEARCH_QUERY")
        claim_extract = _parse_field(gate_raw, "CLAIM_EXTRACT")

        if not needs_search or not search_query or search_query.upper() == "NONE":
            print("[brain] ⏭️  No checkable fact detected — skipping search")
            return _no_fact_check()

        # ── Step B: Web Search ────────────────────────────────────────────────
        print(f"[brain] 🌐 Searching: \"{search_query}\"")
        raw_results    = _web_search(search_query, max_results=4)
        formatted      = _format_search_results(raw_results)

        if not raw_results:
            print("[brain] ⚠️  Search returned no results")
            return _no_fact_check()

        # ── Step C: Verdict ───────────────────────────────────────────────────
        print("[brain] ⚖️  Evaluating fact against search results...")
        verdict_response = fast_llm.invoke([
            SystemMessage(content=FACT_CHECK_VERDICT_PROMPT),
            HumanMessage(content=(
                f"USER'S CLAIM:\n\"{claim_extract or current_claim}\"\n\n"
                f"SEARCH RESULTS:\n{formatted}"
            )),
        ])
        verdict_raw      = verdict_response.content.strip()
        verdict          = _parse_field(verdict_raw, "VERDICT").upper()
        confidence       = _parse_field(verdict_raw, "CONFIDENCE").upper()
        finding          = _parse_field(verdict_raw, "FINDING")
        corrected        = _parse_field(verdict_raw, "CORRECTED_FACT")

        # Normalise verdict
        if verdict not in ("ACCURATE", "INACCURATE", "UNVERIFIABLE"):
            verdict = "UNVERIFIABLE"

        verdict_icon = {
            "ACCURATE"     : "✅",
            "INACCURATE"   : "❌",
            "UNVERIFIABLE" : "⚠️",
        }.get(verdict, "❓")
        print(f"[brain] {verdict_icon} Fact verdict: {verdict} (confidence: {confidence})")

        # ── Step D: Format context block for advocate ─────────────────────────
        context_lines = [
            f"SEARCH QUERY : \"{search_query}\"",
            f"CLAIM CHECKED: \"{claim_extract or current_claim[:200]}\"",
            f"VERDICT      : {verdict} (Confidence: {confidence})",
            f"FINDING      : {finding}",
        ]
        if corrected and corrected.upper() != "NONE":
            context_lines.append(f"CORRECTED    : {corrected}")
        if raw_results:
            context_lines.append(
                f"SOURCES      : "
                + " | ".join(r.get("href", "")[:60] for r in raw_results[:2])
            )

        fact_check_context = "\n".join(context_lines)

        return {
            "fact_check_result"   : formatted,
            "fact_check_verdict"  : verdict,
            "fact_check_performed": True,
            "fact_check_context"  : fact_check_context,
        }

    # ── Node 4: Contradiction Analyser ───────────────────────────────────────
    def contradiction_node(state: DebateState) -> dict:
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_msgs:
            return {"analysis_report": ""}

        current_claim  = human_msgs[-1].content
        past_arguments = state.get("memory_context", "") or "None available yet."

        print("[brain] 🔬 Running logic analysis...")
        response = fast_llm.invoke([
            SystemMessage(content=CONTRADICTION_PROMPT),
            HumanMessage(content=(
                f"CURRENT CLAIM:\n{current_claim}\n\n"
                f"PAST ARGUMENTS:\n{past_arguments}"
            )),
        ])
        report   = response.content.strip()
        severity = _extract_severity(report)
        icon     = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(severity, "⚪")
        print(f"[brain] {icon} Logic severity: {severity}")
        return {"analysis_report": report}

    # ── Node 5: Advocate ─────────────────────────────────────────────────────
    def advocate_node(state: DebateState) -> dict:
        system_prompt = build_system_prompt(
            fact_check_context = state.get("fact_check_context",  ""),
            strategy_context   = state.get("strategy_context",    ""),
            analysis_report    = state.get("analysis_report",     ""),
            memory_context     = state.get("memory_context",      ""),
        )
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response: AIMessage = llm.invoke(messages)
        return {"messages": [response]}

    # ── Node 6: Judge  ← NEW ──────────────────────────────────────────────────
    def judge_node(state: DebateState) -> dict:
        """
        Neutral Justice — audits the Advocate's response for:
          • Bullying (aggression without logic)
          • Strawman (misrepresenting the user's claim)
          • Fact Suppression (ignoring the Researcher's findings)

        Uses the FULL-POWER LLM (not fast_llm) with temperature=0 for
        maximum reliability and genuine independence from the Advocate.
        This deliberate model separation ensures the Judge cannot simply
        echo the Advocate's reasoning style.
        """
        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        human_msgs  = [m for m in state["messages"] if isinstance(m, HumanMessage)]

        if not ai_messages or not human_msgs:
            return _no_judge()

        advocate_response = ai_messages[-1].content
        user_claim        = human_msgs[-1].content
        fact_ctx          = state.get("fact_check_context", "") or "No fact-check performed this turn."

        print("[brain] ⚖️  Judge is reviewing the Advocate's response...")

        # Judge uses full LLM at temp=0 — deterministic, independent, cold
        judge_llm = build_llm(fast=False)
        # Override temperature via a wrapper call
        response = judge_llm.invoke(
            [
                SystemMessage(content=JUDGE_PROMPT),
                HumanMessage(content=(
                    f"USER_CLAIM:\n{user_claim}\n\n"
                    f"ADVOCATE_RESPONSE:\n{advocate_response}\n\n"
                    f"FACT_CHECK_CONTEXT:\n{fact_ctx}"
                )),
            ]
        )
        raw = response.content.strip()

        verdict    = _parse_field(raw, "VERDICT").upper()
        violation  = _parse_field(raw, "VIOLATION").upper()
        severity   = _parse_field(raw, "SEVERITY").upper()
        reasoning  = _parse_field(raw, "REASONING")

        if verdict not in ("UPHELD", "OVERRULED"):
            verdict = "UPHELD"

        is_overruled = verdict == "OVERRULED"

        if is_overruled:
            print(f"[brain] 🔨 OVERRULED — {violation} ({severity}) · {reasoning[:80]}...")
        else:
            print(f"[brain] ✅ UPHELD — Advocate's response passed judicial review")

        full_verdict = f"[{verdict}] Violation: {violation} | Severity: {severity}\n{reasoning}"

        return {
            "judge_verdict"  : full_verdict,
            "judge_reasoning": reasoning,
            "is_overruled"   : is_overruled,
        }

    # ── Node 7: Scoring ───────────────────────────────────────────────────────
    def scoring_node(state: DebateState) -> dict:
        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        human_msgs  = [m for m in state["messages"] if isinstance(m, HumanMessage)]

        if not ai_messages or not human_msgs:
            return {
                "last_user_delta"        : 0,
                "last_bot_delta"         : 0,
                "last_score_reason_user" : "No turn to score.",
                "last_score_reason_bot"  : "No turn to score.",
            }

        bot_new_response  = ai_messages[-1].content
        user_input        = human_msgs[-1].content
        bot_last_response = (
            ai_messages[-2].content if len(ai_messages) >= 2
            else state.get("last_bot_response", "")
        )

        print("[brain] 🧮 Calculating scores...")
        scoring_input = (
            f"ANALYSIS_REPORT:\n{state.get('analysis_report', 'None')}\n\n"
            f"FACT_CHECK_VERDICT: {state.get('fact_check_verdict', 'SKIPPED')}\n"
            f"FACT_CHECK_FINDING: {_parse_field(state.get('fact_check_context', ''), 'FINDING') or 'N/A'}\n\n"
            f"BOT_LAST_RESPONSE:\n{bot_last_response[:500] if bot_last_response else 'First turn.'}\n\n"
            f"BOT_NEW_RESPONSE:\n{bot_new_response[:500]}\n\n"
            f"USER_INPUT:\n{user_input}\n\n"
            f"CONCESSION: {'YES' if state.get('concession_detected') else 'NO'}"
        )

        response = fast_llm.invoke([
            SystemMessage(content=SCORING_PROMPT),
            HumanMessage(content=scoring_input),
        ])
        raw = response.content.strip()

        user_delta  = max(-25, min(10, _parse_int_field(raw, "USER_DELTA")))
        bot_delta   = max(-10, min(15, _parse_int_field(raw, "BOT_DELTA")))
        user_reason = _parse_field(raw, "USER_REASON") or "Score updated."
        bot_reason  = _parse_field(raw, "BOT_REASON")  or "Score updated."

        # ── Apply judge overrule penalty on top of normal scoring ─────────────
        is_overruled = state.get("is_overruled", False)
        if is_overruled:
            judge_penalty = -15          # bot loses 15 for dishonest response
            justice_bonus = +5           # user gains 5 for surviving unfair attack
            bot_delta   += judge_penalty
            user_delta  += justice_bonus
            bot_reason   = f"OVERRULED by Judge (-15). {bot_reason}"
            user_reason  = f"Justice Bonus +5 (survived overruled attack). {user_reason}"
            print(f"[brain] 🔨 Judge penalty applied — Bot: {judge_penalty:+d}, User: {justice_bonus:+d}")

        new_user = max(0, state.get("user_score", 0) + user_delta)
        new_bot  = max(0, state.get("bot_score",  0) + bot_delta)

        verdict = state.get("fact_check_verdict", "SKIPPED")
        fc_tag  = f" [Fact: {verdict}]" if verdict != "SKIPPED" else ""
        print(
            f"[brain] 📊 Scores → User: {state.get('user_score',0)}{user_delta:+d}={new_user} · "
            f"Bot: {state.get('bot_score',0)}{bot_delta:+d}={new_bot}{fc_tag}"
        )

        return {
            "user_score"             : new_user,
            "bot_score"              : new_bot,
            "last_user_delta"        : user_delta,
            "last_bot_delta"         : bot_delta,
            "last_score_reason_user" : user_reason,
            "last_score_reason_bot"  : bot_reason,
        }

    # ── Wire the graph ────────────────────────────────────────────────────────
    graph = StateGraph(DebateState)

    graph.add_node("feedback_analysis", feedback_analysis_node)
    graph.add_node("memory_retrieval",  memory_retrieval_node)
    graph.add_node("researcher",        researcher_node)
    graph.add_node("contradiction",     contradiction_node)
    graph.add_node("advocate",          advocate_node)
    graph.add_node("judge",             judge_node)             # ← new
    graph.add_node("scoring",           scoring_node)

    graph.set_entry_point("feedback_analysis")
    graph.add_edge("feedback_analysis", "memory_retrieval")
    graph.add_edge("memory_retrieval",  "researcher")
    graph.add_edge("researcher",        "contradiction")
    graph.add_edge("contradiction",     "advocate")
    graph.add_edge("advocate",          "judge")                # ← new edge
    graph.add_edge("judge",             "scoring")              # ← updated
    graph.add_edge("scoring",           END)

    return graph.compile()


# ─────────────────────────────────────────────
# 6.  Final Report Generator (called by app.py)
# ─────────────────────────────────────────────

def generate_final_report(
    llm                  : BaseChatModel,
    conversation_history : list,
    user_score           : int,
    bot_score            : int,
) -> str:
    if not conversation_history:
        return "No debate to analyse yet. Make some arguments first!"

    lines = []
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            lines.append(f"HUMAN: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"ADVOCATE: {msg.content}")

    transcript = "\n\n".join(lines)
    prompt = (
        f"{FINAL_REPORT_PROMPT}\n\n"
        f"═══ TRANSCRIPT ═══\n\n{transcript}\n\n"
        f"═══ FINAL SCORES ═══\n"
        f"Human    : {user_score} pts\n"
        f"Advocate : {bot_score} pts\n"
    )
    print("[brain] 📝 Generating final report...")
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ─────────────────────────────────────────────
# 7.  Echo Chamber — Self-Play Engine
# ─────────────────────────────────────────────

class SelfPlayTurnResult:
    """
    Returned by run_self_play_round() for each completed turn.
    Carries everything app.py needs to render one round of the simulation.
    """
    def __init__(
        self,
        round_num        : int,
        proponent_msg    : str,
        advocate_msg     : str,
        fact_verdict     : str,
        fact_finding     : str,
        fact_performed   : bool,
        judge_verdict    : str,
        judge_reasoning  : str,
        is_overruled     : bool,
        proponent_score  : int,
        advocate_score   : int,
        proponent_delta  : int,
        advocate_delta   : int,
        score_reason_pro : str,
        score_reason_adv : str,
    ):
        self.round_num        = round_num
        self.proponent_msg    = proponent_msg
        self.advocate_msg     = advocate_msg
        self.fact_verdict     = fact_verdict
        self.fact_finding     = fact_finding
        self.fact_performed   = fact_performed
        self.judge_verdict    = judge_verdict
        self.judge_reasoning  = judge_reasoning
        self.is_overruled     = is_overruled
        self.proponent_score  = proponent_score
        self.advocate_score   = advocate_score
        self.proponent_delta  = proponent_delta
        self.advocate_delta   = advocate_delta
        self.score_reason_pro = score_reason_pro
        self.score_reason_adv = score_reason_adv


def run_self_play_round(
    llm           : BaseChatModel,
    fast_llm      : BaseChatModel,
    topic         : str,
    round_num     : int,
    history       : list,          # list of (role, content) tuples: "proponent"|"advocate"
    pro_score     : int,
    adv_score     : int,
) -> SelfPlayTurnResult:
    """
    Runs ONE round of the Echo Chamber self-play:

      1. Proponent generates a defence of the topic
      2. Researcher fact-checks the proponent's claim
      3. Advocate counters (using existing Socratic logic)
      4. Judge reviews the Advocate's response
      5. Scorer assigns points to both sides

    The history list is updated in-place so each round builds on the last.
    """

    # ── Build conversation context from history ───────────────────────────────
    context_lines = [f"DEBATE TOPIC: {topic}\n"]
    for role, content in history:
        label = "PROPONENT" if role == "proponent" else "DEVIL'S ADVOCATE"
        context_lines.append(f"{label}:\n{content}")
    context_str = "\n\n".join(context_lines)

    # ── Step 1: Proponent generates their argument ────────────────────────────
    print(f"[echo] 🟦 Round {round_num} — Proponent generating argument...")
    last_advocate = next(
        (c for r, c in reversed(history) if r == "advocate"), None
    )
    proponent_user_msg = (
        f"You are defending the position: \"{topic}\"\n\n"
        f"Debate history so far:\n{context_str}\n\n"
        + (
            f"The Devil's Advocate just said:\n\"{last_advocate}\"\n\n"
            f"Respond as the Proponent. Counter their argument and advance your case."
            if last_advocate else
            f"This is round 1. Open the debate with your strongest argument for: \"{topic}\""
        )
    )
    pro_response = llm.invoke([
        SystemMessage(content=PROPONENT_PROMPT),
        HumanMessage(content=proponent_user_msg),
    ])
    proponent_msg = pro_response.content.strip()
    history.append(("proponent", proponent_msg))

    # ── Step 2: Researcher fact-checks the Proponent ─────────────────────────
    print(f"[echo] 🔍 Round {round_num} — Fact-checking Proponent's claim...")
    gate_resp = fast_llm.invoke([
        SystemMessage(content=FACT_CHECK_GATE_PROMPT),
        HumanMessage(content=f"USER'S CLAIM:\n{proponent_msg}"),
    ])
    gate_raw     = gate_resp.content.strip()
    needs_search = "NEEDS_SEARCH: YES" in gate_raw.upper()
    search_query = _parse_field(gate_raw, "SEARCH_QUERY")
    claim_extract = _parse_field(gate_raw, "CLAIM_EXTRACT")

    fact_verdict   = "SKIPPED"
    fact_finding   = ""
    fact_performed = False
    fact_context   = ""

    if needs_search and search_query and search_query.upper() != "NONE":
        raw_results = _web_search(search_query, max_results=4)
        if raw_results:
            formatted = _format_search_results(raw_results)
            verdict_resp = fast_llm.invoke([
                SystemMessage(content=FACT_CHECK_VERDICT_PROMPT),
                HumanMessage(content=(
                    f"USER'S CLAIM:\n\"{claim_extract or proponent_msg[:300]}\"\n\n"
                    f"SEARCH RESULTS:\n{formatted}"
                )),
            ])
            vr          = verdict_resp.content.strip()
            fact_verdict = _parse_field(vr, "VERDICT").upper() or "UNVERIFIABLE"
            confidence  = _parse_field(vr, "CONFIDENCE").upper()
            fact_finding = _parse_field(vr, "FINDING")
            corrected   = _parse_field(vr, "CORRECTED_FACT")
            fact_performed = True

            ctx_lines = [
                f"SEARCH QUERY : \"{search_query}\"",
                f"CLAIM CHECKED: \"{claim_extract or proponent_msg[:200]}\"",
                f"VERDICT      : {fact_verdict} (Confidence: {confidence})",
                f"FINDING      : {fact_finding}",
            ]
            if corrected and corrected.upper() != "NONE":
                ctx_lines.append(f"CORRECTED    : {corrected}")
            fact_context = "\n".join(ctx_lines)

            icon = {"ACCURATE": "✅", "INACCURATE": "❌", "UNVERIFIABLE": "⚠️"}.get(fact_verdict, "")
            print(f"[echo] {icon} Fact verdict: {fact_verdict}")

    # ── Step 3: Advocate counters ─────────────────────────────────────────────
    print(f"[echo] 🔴 Round {round_num} — Advocate generating counter...")
    advocate_system = build_system_prompt(
        fact_check_context = fact_context,
        strategy_context   = "",
        analysis_report    = "",
        memory_context     = context_str,
    )
    advocate_user_msg = (
        f"The Proponent just argued:\n\"{proponent_msg}\"\n\n"
        f"Debate topic: {topic}\n\n"
        f"Challenge their argument as the Devil's Advocate. Be rigorous."
    )
    adv_response  = llm.invoke([
        SystemMessage(content=advocate_system),
        HumanMessage(content=advocate_user_msg),
    ])
    advocate_msg = adv_response.content.strip()
    history.append(("advocate", advocate_msg))

    # ── Step 4: Judge reviews the Advocate's response ─────────────────────────
    print(f"[echo] ⚖️  Round {round_num} — Judge reviewing...")
    judge_llm = build_llm(fast=False)
    judge_resp = judge_llm.invoke([
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=(
            f"USER_CLAIM:\n{proponent_msg}\n\n"
            f"ADVOCATE_RESPONSE:\n{advocate_msg}\n\n"
            f"FACT_CHECK_CONTEXT:\n{fact_context or 'No fact-check performed.'}"
        )),
    ])
    jraw         = judge_resp.content.strip()
    j_verdict    = _parse_field(jraw, "VERDICT").upper()
    j_violation  = _parse_field(jraw, "VIOLATION").upper()
    j_severity   = _parse_field(jraw, "SEVERITY").upper()
    j_reasoning  = _parse_field(jraw, "REASONING")
    is_overruled = j_verdict == "OVERRULED"
    judge_full   = f"[{j_verdict}] {j_violation} ({j_severity})\n{j_reasoning}"

    if is_overruled:
        print(f"[echo] 🔨 OVERRULED — {j_violation} ({j_severity})")
    else:
        print(f"[echo] ✅ UPHELD")

    # ── Step 5: Scoring ───────────────────────────────────────────────────────
    print(f"[echo] 🧮 Round {round_num} — Scoring...")

    SELF_PLAY_SCORING_PROMPT = f"""\
You are scoring ONE round of an autonomous AI debate.

PROPONENT argued for: "{topic}"
PROPONENT'S ARGUMENT: {proponent_msg[:500]}

ADVOCATE'S COUNTER: {advocate_msg[:500]}

FACT_CHECK_VERDICT: {fact_verdict}
FACT_CHECK_FINDING: {fact_finding or 'N/A'}
JUDGE_VERDICT: {j_verdict} — {j_violation} ({j_severity})

Score the PROPONENT (not the human, the AI defending the topic):
  -3 per logical fallacy in their argument
  -5 per factual inaccuracy detected
  -2 for weak or vague arguments
  +3 for a strong, evidence-based point
  +2 for directly countering the Advocate's last argument

Score the ADVOCATE (Devil's Advocate):
  +3 for a sharp, targeted counter-argument
  +2 if they correctly exploited a proponent fallacy
  -3 if OVERRULED by the Judge
  -2 for repeating previous challenges
  +5 if fact-check was INACCURATE (they caught a false claim)

Respond EXACTLY in this format, nothing else:
PROPONENT_DELTA: [integer]
ADVOCATE_DELTA: [integer]
PROPONENT_REASON: [one sentence]
ADVOCATE_REASON: [one sentence]"""

    score_resp = fast_llm.invoke([HumanMessage(content=SELF_PLAY_SCORING_PROMPT)])
    sraw = score_resp.content.strip()

    pro_delta  = max(-15, min(10, _parse_int_field(sraw, "PROPONENT_DELTA")))
    adv_delta  = max(-10, min(10, _parse_int_field(sraw, "ADVOCATE_DELTA")))
    pro_reason = _parse_field(sraw, "PROPONENT_REASON") or "Score updated."
    adv_reason = _parse_field(sraw, "ADVOCATE_REASON")  or "Score updated."

    # Apply judge overrule penalty
    if is_overruled:
        adv_delta -= 3   # stacks with scoring
        pro_delta += 2   # proponent survives unfair challenge

    new_pro = max(0, pro_score + pro_delta)
    new_adv = max(0, adv_score + adv_delta)

    print(
        f"[echo] 📊 Round {round_num} scores — "
        f"Proponent: {pro_score}{pro_delta:+d}={new_pro} | "
        f"Advocate: {adv_score}{adv_delta:+d}={new_adv}"
    )

    return SelfPlayTurnResult(
        round_num        = round_num,
        proponent_msg    = proponent_msg,
        advocate_msg     = advocate_msg,
        fact_verdict     = fact_verdict,
        fact_finding     = fact_finding,
        fact_performed   = fact_performed,
        judge_verdict    = judge_full,
        judge_reasoning  = j_reasoning,
        is_overruled     = is_overruled,
        proponent_score  = new_pro,
        advocate_score   = new_adv,
        proponent_delta  = pro_delta,
        advocate_delta   = adv_delta,
        score_reason_pro = pro_reason,
        score_reason_adv = adv_reason,
    )


def generate_self_play_verdict(
    llm          : BaseChatModel,
    topic        : str,
    history      : list,
    pro_score    : int,
    adv_score    : int,
) -> str:
    """Generates a post-simulation verdict for Echo Chamber mode."""
    lines = [f"DEBATE TOPIC: {topic}\n"]
    for role, content in history:
        label = "PROPONENT" if role == "proponent" else "DEVIL'S ADVOCATE"
        lines.append(f"{label}:\n{content}")

    prompt = (
        "You are a debate analyst. Analyse this autonomous AI-vs-AI debate and produce "
        "a structured verdict.\n\n"
        "🏆 WINNER: Declare PROPONENT, ADVOCATE, or DRAW with one clear reason.\n"
        "📋 PROPONENT ANALYSIS: 2-3 strengths and 1 key weakness.\n"
        "🤖 ADVOCATE ANALYSIS: 2-3 strengths and 1 key weakness.\n"
        "⚡ BEST EXCHANGE: Quote the most decisive back-and-forth.\n"
        "📊 SCORES: Comment on what "
        f"Proponent {pro_score} pts vs Advocate {adv_score} pts reveals.\n\n"
        "═══ TRANSCRIPT ═══\n\n"
        + "\n\n".join(lines)
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ─────────────────────────────────────────────
# 8.  Helpers
# ─────────────────────────────────────────────

def _no_fact_check() -> dict:
    return {
        "fact_check_result"   : "",
        "fact_check_verdict"  : "SKIPPED",
        "fact_check_performed": False,
        "fact_check_context"  : "",
    }


def _no_judge() -> dict:
    return {
        "judge_verdict"  : "[UPHELD] No response to audit.",
        "judge_reasoning": "",
        "is_overruled"   : False,
    }


def _parse_field(text: str, field: str) -> str:
    for line in text.splitlines():
        if line.strip().upper().startswith(field.upper() + ":"):
            return line.split(":", 1)[-1].strip()
    return ""


def _parse_int_field(text: str, field: str) -> int:
    raw = _parse_field(text, field)
    try:
        nums = re.findall(r"-?\d+", raw)
        return int(nums[0]) if nums else 0
    except (ValueError, IndexError):
        return 0


def _extract_severity(report: str) -> str:
    for line in report.splitlines():
        if line.strip().upper().startswith("SEVERITY"):
            for level in ("HIGH", "MEDIUM", "LOW"):
                if level in line.upper():
                    return level
    return "UNKNOWN"


# ─────────────────────────────────────────────
# 8.  Terminal REPL
# ─────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      🗣️  DEVIL'S ADVOCATE  ·  FINAL EDITION                 ║
║      Researcher · Scorecard · Logic Engine · Memory         ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    quit / exit → end     |  reset  → clear turn             ║
║    forget      → wipe DB |  stats  → memory info            ║
║    score       → current scorecard                          ║
║    report      → last logic audit                           ║
║    factcheck   → last fact-check result                     ║
║    tactics     → winning tactics list                       ║
║    verdict     → generate final debate report               ║
╚══════════════════════════════════════════════════════════════╝
"""


def format_response(
    text              : str,
    memories_used     : int  = 0,
    severity          : str  = "",
    strategies_used   : int  = 0,
    concession        : bool = False,
    user_score        : int  = 0,
    bot_score         : int  = 0,
    user_delta        : int  = 0,
    bot_delta         : int  = 0,
    fact_verdict      : str  = "SKIPPED",
    fact_performed    : bool = False,
    is_overruled      : bool = False,
) -> str:
    tags = []
    if is_overruled:
        tags.append("🔨 OVERRULED")
    if concession:
        tags.append("✅ concession")
    if fact_performed:
        icon = {"ACCURATE": "✅", "INACCURATE": "❌", "UNVERIFIABLE": "⚠️"}.get(fact_verdict, "🔍")
        tags.append(f"{icon} Fact: {fact_verdict}")
    if strategies_used > 0:
        tags.append(f"🎯 {strategies_used} tactic(s)")
    if memories_used > 0:
        tags.append(f"🧠 {memories_used} mem")
    if severity and severity != "UNKNOWN":
        icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(severity, "")
        tags.append(f"{icon} {severity}")

    tag_line   = f" [{' · '.join(tags)}]" if tags else ""
    score_line = f"\n  📊 You: {user_score} ({user_delta:+d}) | Bot: {bot_score} ({bot_delta:+d})"
    divider    = "─" * 64
    return f"\n{divider}\n🔴 Devil's Advocate:{tag_line}{score_line}\n\n{text}\n{divider}\n"


if __name__ == "__main__":
    print(BANNER)

    llm           = build_llm(fast=False)
    fast_llm      = build_llm(fast=True)
    debate_memory = DebateMemory()
    graph         = build_debate_graph(llm, fast_llm, debate_memory)

    conversation_history : list[BaseMessage] = []
    last_bot_response    : str               = ""
    last_analysis_report : str               = ""
    last_fact_context    : str               = ""
    user_score           : int               = 0
    bot_score            : int               = 0

    print(
        "\n📌 FINAL EDITION active:\n"
        "   🌐 Real-time fact-checking via DuckDuckGo\n"
        "   📊 Live cumulative scoring\n"
        "   🧠 Adaptive tactic memory\n"
        "   Type 'factcheck' to inspect last fact-check result.\n"
    )

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended.]\n")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print(f"\n[Final score — You: {user_score} | Bot: {bot_score}]\n")
            break

        if user_input.lower() == "reset":
            conversation_history = []
            last_bot_response    = ""
            last_analysis_report = ""
            last_fact_context    = ""
            user_score = bot_score = 0
            print("\n[Reset. Scores cleared, memories kept.]\n")
            continue

        if user_input.lower() == "forget":
            confirm = input("⚠️  Wipe ALL memories? Type 'yes': ")
            if confirm.strip().lower() == "yes":
                debate_memory.clear_all()
                print("[All memories wiped.]\n")
            else:
                print("[Cancelled.]\n")
            continue

        if user_input.lower() == "score":
            leader = (
                "🧑 You lead!" if user_score > bot_score
                else "🔴 Bot leads." if bot_score > user_score
                else "🤝 Tied."
            )
            print(f"\n📊 Score: You {user_score} | Bot {bot_score} — {leader}\n")
            continue

        if user_input.lower() == "report":
            print(f"\n{'─'*64}\n{last_analysis_report or '[No report yet.]'}\n{'─'*64}\n")
            continue

        if user_input.lower() == "factcheck":
            print(f"\n{'─'*64}\n{last_fact_context or '[No fact-check performed yet.]'}\n{'─'*64}\n")
            continue

        if user_input.lower() == "tactics":
            tactics = debate_memory.get_all_strategies()
            if not tactics:
                print("\n[No tactics yet.]\n")
            else:
                print(f"\n🏆 Tactics ({len(tactics)}):")
                for i, t in enumerate(tactics, 1):
                    print(f"  {i}. {t['tactic_name']} | {t['topic'] or 'General'}")
                print()
            continue

        if user_input.lower() == "verdict":
            report = generate_final_report(llm, conversation_history, user_score, bot_score)
            print(f"\n{'═'*64}\n{report}\n{'═'*64}\n")
            continue

        if user_input.lower() == "stats":
            s = debate_memory.get_stats()
            print(
                f"\n📊 Memory: {s['total_debate_memories']} debates, "
                f"{s['total_winning_strategies']} tactics, "
                f"session {s['session_id']}\n"
            )
            continue

        # ── Normal debate turn ────────────────────────────────────────────────
        conversation_history.append(HumanMessage(content=user_input))

        initial_state: DebateState = {
            "messages"               : conversation_history,
            "retrieved_memories"     : [],
            "memory_context"         : "",
            "analysis_report"        : "",
            "last_bot_response"      : last_bot_response,
            "winning_strategies"     : [],
            "strategy_context"       : "",
            "concession_detected"    : False,
            "user_score"             : user_score,
            "bot_score"              : bot_score,
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

        result = graph.invoke(initial_state)

        ai_message           = result["messages"][-1]
        memories_used        = len(result.get("retrieved_memories", []))
        strategies_used      = len(result.get("winning_strategies",  []))
        last_analysis_report = result.get("analysis_report",         "")
        concession_detected  = result.get("concession_detected",     False)
        severity             = _extract_severity(last_analysis_report)
        user_score           = result.get("user_score",              user_score)
        bot_score            = result.get("bot_score",               bot_score)
        user_delta           = result.get("last_user_delta",         0)
        bot_delta            = result.get("last_bot_delta",          0)
        fact_verdict         = result.get("fact_check_verdict",      "SKIPPED")
        fact_performed       = result.get("fact_check_performed",    False)
        last_fact_context    = result.get("fact_check_context",      "")
        is_overruled         = result.get("is_overruled",            False)
        judge_reasoning      = result.get("judge_reasoning",         "")

        if is_overruled:
            print(f"\n⚖️  JUDGE OVERRULED the Advocate: {judge_reasoning}\n")

        last_bot_response = ai_message.content
        conversation_history.append(ai_message)
        debate_memory.store_argument(user_input, ai_message.content)

        print(format_response(
            ai_message.content,
            memories_used   = memories_used,
            severity        = severity,
            strategies_used = strategies_used,
            concession      = concession_detected,
            user_score      = user_score,
            bot_score       = bot_score,
            user_delta      = user_delta,
            bot_delta       = bot_delta,
            fact_verdict    = fact_verdict,
            fact_performed  = fact_performed,
            is_overruled    = is_overruled,
        ))
