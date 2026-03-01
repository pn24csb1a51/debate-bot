# 🔴 Devil's Advocate
### *A Multi-Agent Strategic Reasoning Engine with Constitutional Governance*

> A production-grade AI debate system that stress-tests human reasoning in real time. Seven specialised LangGraph nodes work in concert to fact-check claims, audit logic, detect fallacies, score arguments, and overrule unfair AI responses — all while building a persistent memory of what rhetorical tactics work best against *you specifically*.

---

## Table of Contents

- [Overview](#overview)
- [Cognitive Architecture — The 7-Node Pipeline](#cognitive-architecture)
- [Elite Technical Features](#elite-technical-features)
- [Visual Analytics — The Weakness Radar](#visual-analytics)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scoring System](#scoring-system)

---

## Overview

Devil's Advocate is not a chatbot. It is a **multi-agent reasoning engine** designed to challenge the logical integrity of every argument a user makes. Each user message triggers a full 7-stage agentic pipeline: the claim is fact-checked against live web data, audited for logical fallacies and self-contradictions, challenged by a Socratic AI opponent, reviewed by a neutral judicial agent, and scored — all before a response is rendered.

The system accumulates long-term memory across sessions, learning which rhetorical tactics cause *this specific user* to concede. It then deploys those tactics with increasing precision over time.

A secondary **Echo Chamber** mode runs two AI agents against each other autonomously, generating synthetic debate transcripts that serve as rich training data for argument quality analysis.

---

## Cognitive Architecture

### The 7-Node LangGraph Pipeline

Every user message flows through a deterministic directed acyclic graph compiled by **LangGraph**. Nodes execute sequentially; each node reads from and writes back to a shared `DebateState` TypedDict. The full pipeline is:

```
feedback_analysis → memory_retrieval → researcher
                                           │
                                    contradiction
                                           │
                                        advocate
                                           │
                                         judge
                                           │
                                        scoring → END
```

---

### Node Descriptions

| # | Node | Role | LLM Used |
|---|---|---|---|
| 1 | **`feedback_analysis_node`** | Detects whether the user's latest message constitutes a concession of the previous argument. If yes, logs the winning tactic to persistent vector memory. | Fast LLM |
| 2 | **`memory_retrieval_node`** | Queries ChromaDB for semantically similar past arguments and retrieves hybrid-ranked winning strategies for the current topic. | *(No LLM — vector search only)* |
| 3 | **`researcher_node`** | Two-stage fact-checking gate. First determines if the claim contains a verifiable statistic or factual assertion. If yes, executes a live DuckDuckGo web search and passes the results to a verdict LLM that classifies the claim as `ACCURATE`, `INACCURATE`, or `UNVERIFIABLE`. | Fast LLM × 2 |
| 4 | **`contradiction_node`** | Formal logic audit. Compares the current claim against the user's full argument history to detect direct contradictions. Independently identifies logical fallacies (Ad Hominem, Strawman, Hasty Generalisation, Slippery Slope, etc.). Assigns a severity rating of `LOW`, `MEDIUM`, or `HIGH`. | Fast LLM |
| 5 | **`advocate_node`** | The primary Socratic opponent. Builds a structured response using all prior context: fact-check findings, winning tactics, contradiction report, and argument history. Follows a strict response protocol: Steelman → Fact Check → Contradiction → Assumption Exposure → Challenge → Socratic Question. | Full LLM |
| 6 | **`judge_node`** | A constitutionally independent neutral agent that audits the Advocate's response for three violation types: **Bullying** (aggression without logic), **Strawman** (misrepresentation of the user's claim), and **Fact Suppression** (ignoring the Researcher's findings). Runs on the full model at `temperature=0` for maximum determinism. Can issue an `OVERRULED` verdict that penalises the bot's score. | Full LLM (temp=0) |
| 7 | **`scoring_node`** | Evaluates the full turn and assigns cumulative point deltas to both sides based on: fallacies found, contradictions detected, factual accuracy, concession status, judge verdict, and rhetorical quality. Applies the Judge's overrule penalty (`-15 bot / +5 user justice bonus`) on top of the base scoring. | Fast LLM |

---

## Elite Technical Features

### 1. Bandit Strategy Learning (Adaptive Tactic Memory)

**File:** `memory.py`

The system maintains a dedicated ChromaDB vector collection called `strategy_feedback` that stores every rhetorical tactic that has successfully caused the user to concede, stall, or admit a logical flaw. This collection is more than a log — it functions as a **contextual bandit**, guiding future tactic selection.

**How it works:**

When a concession is detected by `feedback_analysis_node`, the following metadata is written to ChromaDB via an **upsert pattern** (deterministic document ID derived from MD5 hash of tactic name):

```python
{
    "tactic_name"       : "Socratic Questioning",
    "topic"             : "AI job displacement",
    "user_concession"   : "ok fair point, I hadn't considered that...",
    "bot_argument_used" : "Your premise assumes full substitution...",
    "timestamp"         : "2025-01-15T14:32:00",
    "session_id"        : "abc-123"
}
```

On each subsequent turn, `memory_retrieval_node` retrieves candidate tactics using semantic similarity search, then re-ranks them using a **hybrid scoring formula:**

```
hybrid_score = (cosine_similarity × 0.6) + (success_rate × 0.4)
```

This means a tactic with 80% historical win rate on similar topics will outrank a tactic that is merely topically relevant but has never worked. The system gets harder to debate the longer you use it.

---

### 2. Constitutional Governance (Judicial Review Node)

**File:** `brain.py` — `judge_node`, `JUDGE_PROMPT`

The Judge is a second, fully independent AI agent with a deliberate adversarial relationship to the Advocate. Three design decisions enforce genuine independence:

- **Separate system prompt** — The Judge's persona explicitly states: *"You are NOT the Devil's Advocate. You do NOT take sides."* It receives no access to the Advocate's tactic memory or strategy context.
- **Temperature=0 on full model** — The Judge uses deterministic inference on the highest-capability model, while the Advocate uses `temperature=0.7` for rhetorical variety. They reason differently by design.
- **Strict false-positive guard** — The prompt explicitly defines what does *not* constitute a violation (tough-but-logical challenges, slight re-framing, expanding beyond the fact-check). The Judge does not overrule for style.

**Violation taxonomy:**

| Violation | Definition |
|---|---|
| **BULLYING** | Aggressive or dismissive language that replaces logical argumentation rather than accompanying it |
| **STRAWMAN** | Attacking a weakened misrepresentation of the user's actual claim |
| **FACT_SUPPRESSION** | Ignoring the Researcher node's `INACCURATE` finding and arguing as if the user's false claim were valid |

**Score impact on `OVERRULED`:** Bot `-15 pts`, User `+5 pts` (Justice Bonus).

---

### 3. Active RAG — Real-Time Fact-Checking

**File:** `brain.py` — `researcher_node`, `_web_search()`

The Researcher implements a **three-stage Active Retrieval Augmented Generation** pipeline that fires conditionally, not on every turn. This prevents unnecessary latency on opinion-based arguments.

**Stage A — Gate (LLM classification):**
A fast LLM call decides whether the claim contains a checkable factual assertion. Pure opinions (`"freedom is important"`), hypotheticals, and philosophical statements skip directly to the contradiction node. Only claims containing statistics, named studies, historical events, or specific causal assertions trigger a search.

**Stage B — Search (DuckDuckGo, no API key required):**
```python
from duckduckgo_search import DDGS
results = list(DDGS().text(query, max_results=4))
```
Returns title, URL, and excerpt for the top 4 results. Rate-limit safe with graceful fallback.

**Stage C — Verdict (LLM evaluation):**
A second fast LLM call reads the search results against the extracted claim and returns a structured verdict:

```
VERDICT: ACCURATE | INACCURATE | UNVERIFIABLE
CONFIDENCE: HIGH | MEDIUM | LOW
FINDING: [2-3 sentence summary of what the web actually says]
CORRECTED_FACT: [The real figure, if INACCURATE]
```

The verdict is injected into both the `analysis_report` (for scoring) and the Advocate's system prompt (so it leads with the corrected data). A verified `INACCURATE` verdict costs the user **-10 points** and awards the bot **+3 points**.

---

### 4. Echo Chamber — Autonomous Agent Self-Play

**File:** `brain.py` — `run_self_play_round()`, `SelfPlayTurnResult`, `app.py` — `render_echo_chamber()`

The Echo Chamber is a fully autonomous debate simulation mode. Given a topic, two AI agents — the **Proponent** (defends the thesis) and the **Devil's Advocate** (attacks it) — debate for a configurable number of rounds (2–8) with no human input.

**Architecture:**

Each round is a complete mini-pipeline:
```
Proponent → Researcher (fact-checks Proponent) → Advocate → Judge → Scorer
```

The Proponent is driven by `PROPONENT_PROMPT`, a separate system persona focused on: direct counter-rebuttal, evidence citation, strategic escalation, and offensive Socratic challenges. The Advocate uses the identical `BASE_SYSTEM_PROMPT` as in human mode — ensuring the same quality of challenge against both human and AI opponents.

**Streamlit integration** uses a **rerun-per-round** pattern rather than threading: each Streamlit rerun processes exactly one round, appends it to `echo_messages` in session state, then calls `st.rerun()`. This produces a live-updating transcript with no concurrency overhead.

**Primary use case:** Synthetic debate data generation. Running 50 simulations on a topic produces a rich annotated corpus of argument quality, fact accuracy, fallacy distribution, and judicial overrule patterns — without any human participants.

---

## Visual Analytics

### The Weakness Radar (Plotly Spider Chart)

**File:** `app.py` — `render_weakness_radar()`, `_update_weakness_profile()`

After every debate turn, five cumulative weakness axes are updated by parsing the structured outputs of the pipeline nodes:

| Axis | Data Source | Increment Logic |
|---|---|---|
| **Fallacies** | `contradiction_node` analysis report | `+1` per `FALLACY:` line detected |
| **Contradictions** | `contradiction_node` analysis report | `+1` per `PAST:` conflict block |
| **Factual Errors** | `researcher_node` verdict | `+1` when `fact_check_verdict == "INACCURATE"` |
| **Severity** | `_extract_severity()` on analysis report | `+2` for HIGH, `+1` for MEDIUM per turn |
| **Bullying** | `judge_node` verdict | `+1` per `OVERRULED` ruling |

Values are normalised to a 0–10 scale for rendering. The chart uses `plotly.graph_objects.Scatterpolar` with a translucent red fill area, styled to match the application's dark editorial aesthetic. Raw counts are displayed below the chart as an inline legend.

In **Echo Chamber** mode, a separate **Proponent Weakness Radar** is generated from the `SelfPlayTurnResult` objects after all rounds complete, showing where the AI defender's logic degraded over the simulation.

---

## Technical Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful multi-node agentic pipeline with typed state |
| **LLM Abstraction** | [LangChain](https://github.com/langchain-ai/langchain) | Unified interface across all LLM providers |
| **Primary LLM** | [Groq](https://groq.com) / OpenAI / Anthropic / Google | Pluggable via `LLM_PROVIDER` env var |
| **Fast LLM** | `llama-3.1-8b-instant` (Groq) | Concession detection, fact-checking gate, scoring |
| **Vector Memory** | [ChromaDB](https://www.trychroma.com) | Persistent local embeddings for argument + tactic memory |
| **Embeddings** | `all-MiniLM-L6-v2` (default ChromaDB) | Free, offline sentence embeddings |
| **Web Search** | [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) | Zero-cost real-time fact retrieval, no API key |
| **UI Framework** | [Streamlit](https://streamlit.io) | Web interface with real-time rerun-per-round streaming |
| **Visualisation** | [Plotly](https://plotly.com/python/) | Interactive radar chart for logical weakness profiling |
| **Language** | Python 3.10+ | |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A valid API key for at least one supported LLM provider (Groq recommended — fastest and free tier available)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/devils-advocate.git
cd devils-advocate

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

```
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-google-genai>=2.0.0
langchain-anthropic>=0.3.0
langchain-groq>=0.2.0
langgraph>=0.2.0
chromadb>=0.5.0
python-dotenv>=1.0.0
streamlit>=1.38.0
duckduckgo-search>=6.2.0
plotly>=5.18.0
```

---

## Configuration

Create a `.env` file in the project root:

```env
# ── LLM Provider ─────────────────────────────────────────────
# Options: groq | openai | anthropic | google
LLM_PROVIDER=groq

# ── Groq (recommended) ───────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant

# ── OpenAI (optional) ────────────────────────────────────────
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4o
# OPENAI_FAST_MODEL=gpt-4o-mini

# ── Anthropic (optional) ─────────────────────────────────────
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# ANTHROPIC_MODEL=claude-opus-4-6
# ANTHROPIC_FAST_MODEL=claude-haiku-4-5-20251001

# ── Google Gemini (optional) ─────────────────────────────────
# GOOGLE_API_KEY=your_google_api_key_here
# GOOGLE_MODEL=gemini-2.0-flash
```

> **Note:** No API key is required for DuckDuckGo web search. ChromaDB stores its vector database locally at `./chroma_db/` with no external connection.

---

## Usage

### Launch the Web Interface

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Live Debate Mode

Navigate to the **💬 Live Debate** tab. Type any claim and press Enter:

```
"Artificial intelligence will eliminate more jobs than it creates"
"Social media platforms should be regulated as public utilities"
"Universal Basic Income is economically unviable"
```

The sidebar updates in real time with the live scorecard, weakness radar, judicial oversight panel, and logic audit after every turn.

### Echo Chamber Mode

Navigate to the **🤖 Echo Chamber** tab:

1. Enter a debate topic
2. Select the number of rounds (2–8)
3. Click **▶ START SIMULATION**

Watch Proponent and Advocate debate autonomously, round by round, with live transcript rendering. A post-simulation verdict and Proponent Weakness Radar are generated on completion.

### Terminal Mode

```bash
python brain.py
```

Runs the full pipeline in a REPL for development and testing. Supports commands: `score`, `report`, `factcheck`, `tactics`, `verdict`, `stats`, `reset`, `forget`.

---

## Project Structure

```
devils-advocate/
│
├── brain.py          # Core agentic pipeline — all 7 LangGraph nodes,
│                     # self-play engine, prompt constants, REPL
│
├── app.py            # Streamlit web UI — Live Debate + Echo Chamber tabs,
│                     # sidebar scorecard, weakness radar, judicial panel
│
├── memory.py         # ChromaDB dual-collection memory system —
│                     # debate history + bandit strategy feedback
│
├── requirements.txt  # Python dependencies
├── .env              # API keys and model configuration (not committed)
└── chroma_db/        # Local vector database (auto-created, not committed)
```

---

## Scoring System

### Live Debate (Human vs Advocate)

| Event | User Δ | Bot Δ |
|---|---|---|
| Logical fallacy detected (per fallacy) | **-3** | — |
| Direct self-contradiction found | **-5** | — |
| MEDIUM severity reasoning | **-2** | — |
| HIGH severity reasoning (stacks) | **-4** | — |
| Factual claim verified INACCURATE | **-10** | **+3** |
| Factual claim UNVERIFIABLE | **-3** | — |
| User concession detected | — | **+5** |
| Fallacy correctly named in response | — | **+2** |
| Judge OVERRULES bot (Justice Bonus) | **+5** | **-15** |
| Bot repeats prior argument | — | **-3** |
| Bot ignores user's core claim | — | **-2** |

Scores are strictly cumulative and floor at 0. The maximum per-turn deduction for the user is capped at **-25 points**.

### Echo Chamber (Proponent vs Advocate)

The self-play scorer uses a separate prompt calibrated for AI-vs-AI argument quality rather than human rhetorical patterns. Both agents are scored independently each round, with the same Judge and Researcher nodes auditing the exchange.

---

## Architectural Notes

**Why LangGraph over a simple chain?** The pipeline requires conditional branching (researcher skips non-factual claims), shared mutable state across nodes (scores must accumulate), and the ability to add nodes without touching existing logic. LangGraph's compiled `StateGraph` provides all of this with full type safety via `TypedDict`.

**Why two separate LLMs?** The fast model (`llama-3.1-8b-instant`) handles deterministic classification tasks — concession detection, fact-check gating, scoring — where speed matters and the task is structured. The full model handles generative, rhetorically demanding responses. This dual-model architecture cuts turn latency by approximately 40% compared to using the full model for everything.

**Why ChromaDB local embeddings?** Keeping the vector database local means the system accumulates persistent user-specific memory across sessions with zero external API calls and no data leaving the machine. The `all-MiniLM-L6-v2` model provides sufficient semantic quality for argument retrieval at zero cost.

---

*Built with LangGraph · LangChain · ChromaDB · Streamlit · Groq*
