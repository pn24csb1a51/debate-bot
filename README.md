# 🧠 Cognitive OS — Multi-Mode Reasoning Engine

> *A multi-agent, memory-augmented reasoning system built on LangGraph, ChromaDB, and Streamlit. Five cognitive modes. One persistent brain.*

---

## What It Is

Cognitive OS is not a chatbot. It is a **reasoning environment** — a set of five specialised AI modes, each with its own logic pipeline, memory collection, and cognitive purpose. Every session you have is remembered. Every weakness in your reasoning is tracked. The system gets smarter about *you* the more you use it.

---

## Architecture Overview

```
app.py  (Streamlit router)
│
├── ui/debate_ui.py        →  brain.py (LangGraph)  +  memory.py (DebateMemory)
├── ui/execution_ui.py     →  core/llm_config.py    +  core/memory_manager.py
├── ui/ripple_ui.py        →  core/llm_config.py    +  core/memory_manager.py
├── ui/wealth_ui.py        →  core/llm_config.py    +  core/memory_manager.py
└── ui/brain_map_ui.py     →  core/memory_manager.py (all 4 collections)
         │
         └── core/
               ├── llm_config.py       — provider-agnostic LLM factory
               └── memory_manager.py   — 4 isolated ChromaDB collections
```

---

## The Five Modes

### 🔴 Devil's Advocate
**The original debate engine. Full LangGraph pipeline.**

A 4-node LangGraph pipeline that stress-tests your reasoning using Socratic challenges, logic auditing, and adaptive tactic learning.

| Node | Role |
|---|---|
| `feedback_analysis_node` | Detects if the user conceded the last turn. Logs the winning tactic to memory. |
| `memory_retrieval_node` | Queries both debate history and winning strategies from ChromaDB |
| `contradiction_node` | Fast-LLM logic audit — finds fallacies, contradictions, assigns severity |
| `advocate_node` | Builds the final Socratic challenge using all prior context |

**Additional features:**

- **⚖️ Judge Node** — independent LLM call after every bot reply, checks for `BULLYING`, `STRAWMAN`, or `FACT_SUPPRESSION`. Violations: Bot -15 pts, You +5 pts.
- **📊 Live Scoreboard** — You vs Bot point system updated every turn
- **📈 Weakness Radar** — Plotly spider chart: Fallacies · Contradictions · Factual Errors · Severity · Bullying
- **🤖 Echo Chamber** — Two LLMs argue any thesis autonomously for 2–8 rounds
- **🏆 Tactics Tab** — Every tactic the bot learned makes you concede

**Scoring table:**

| Event | Points |
|---|---|
| Logical fallacy detected | You **-3** |
| Contradiction with past claim | You **-5** |
| Fact checked INACCURATE | You **-10** |
| Fact UNVERIFIABLE | You **-3** |
| Judge overrules bot | Bot **-15**, You **+5** |
| Concession detected | You **-10**, Bot **+10** |
| Clean argument (no flaws) | You **+3** |

---

### 📅 Execution Architect
**Energy-aware task categorisation and daily planning.**

Set your energy level (1–10 via `select_slider`) and paste your task list. The engine pre-categorises tasks into **High Focus (Deep Work)** vs **Low Focus (Admin)** before sending to the LLM, then streams a ranked execution plan matched to your cognitive state.

- Three energy bands: High (7–10) · Medium (4–6) · Low (1–3)
- Past tasks saved with energy metadata — bottom panel shows completion patterns at the same energy level
- Optional blockers/constraints field feeds into the plan

---

### 🔮 Ripple Engine
**Structured 3-tier consequence mapping with blindspot detection.**

Describe a decision. The engine maps every consequence across three orders, then searches memory for repeating blindspots before generating the analysis.

- **Repeating Pattern Detection** — ChromaDB searched first; red warning shown if you are making the same mistake again
- Forced structured output: `## 1ST ORDER` → `## 2ND ORDER` → `## 3RD ORDER` → `## BLINDSPOTS` → `## And then what?`
- Three colour-coded expandable cards: 🟢 1st · 🟡 2nd · 🔴 3rd
- Each order saved as a separate memory entry with `order=1/2/3` for filtered future queries

---

### 📉 Wealth Logic
**Financial reasoning and opportunity cost auditor.**

Describe a financial scenario. The engine audits reasoning across four lenses: opportunity cost, risk-adjusted return, time horizon, and hidden assumptions.

- **Live Opportunity Cost Calculator** — Path A (your decision) vs Path B (12% index benchmark), compound interest projected side-by-side
- Risk level · asset class · time horizon controls inject directly into LLM context
- Never gives investment advice — audits *reasoning quality* only

---

### 🧠 Brain Map
**Semantic network visualisation of all memory.**

Pulls every entry from all 4 ChromaDB collections and renders them as an interactive 2D network graph. Similar nodes are connected by edges.

- **TF-IDF cosine similarity** (scikit-learn, bigrams, 500 features) across all node texts
- **Fruchterman-Reingold spring layout** in pure NumPy — same-mode nodes cluster naturally
- Edge opacity and width scale with similarity strength
- Hover any node to read full metadata: what you wrote, energy/domain/risk, timestamp
- **Edge Similarity Threshold** slider (0.05–0.70) and **Max Edges** slider (10–300)
- Colour coding: 🔴 `#e63946` · 📅 `#f4a261` · 🔮 `#a8dadc` · 📉 `#2a9d8f` · 🧠 `#c77dff`

---

## Memory Architecture

Four completely isolated ChromaDB collections — no data bleeds between modes.

| Collection | Mode | Key metadata fields |
|---|---|---|
| `debate_memory` | Devil's Advocate | `user_arg`, `bot_response`, `severity`, `tactic_used` |
| `strategy_feedback` | Devil's Advocate | `tactic_name`, `topic`, `user_concession` |
| `execution_strategies` | Execution Architect | `task`, `energy_level`, `category`, `completion_status` |
| `ripple_consequences` | Ripple Engine | `decision`, `consequence`, `order`, `domain`, `blindspot` |
| `wealth_reasoning` | Wealth Logic | `scenario`, `risk_level`, `time_horizon`, `asset_class` |

`core/memory_manager.py` is the single routing layer — self-healing on `get_stats()` if a collection is deleted externally.

---

## LLM Provider Support

Change one line in `.env` — zero code changes required.

```env
LLM_PROVIDER=groq        # groq | openai | anthropic | google
GROQ_API_KEY=your_key
```

| Provider | Full model | Fast model |
|---|---|---|
| Groq | llama-3.3-70b-versatile | llama-3.1-8b-instant |
| OpenAI | gpt-4o | gpt-4o-mini |
| Anthropic | claude-opus-4-6 | claude-haiku-4-5-20251001 |
| Google | gemini-2.0-flash | gemini-2.0-flash |

Fast model used for: concession detection · contradiction analysis · Judge evaluation.
Full model used for: all user-facing responses.

---

## Project Structure

```
debate-bot/
├── app.py                      # Streamlit router — entry point
├── brain.py                    # LangGraph 4-node debate pipeline
├── memory.py                   # DebateMemory — dual ChromaDB collections
├── requirements.txt
├── .env                        # LLM_PROVIDER + API keys
├── .streamlit/
│   └── config.toml             # Dark theme
├── chroma_db/                  # Persistent ChromaDB data (auto-created)
├── core/
│   ├── __init__.py
│   ├── llm_config.py           # Provider-agnostic LLM factory + system prompts
│   └── memory_manager.py       # 4-collection routing layer (self-healing)
└── ui/
    ├── __init__.py
    ├── debate_ui.py            # 🔴 Devil's Advocate — full feature mode
    ├── execution_ui.py         # 📅 Execution Architect
    ├── ripple_ui.py            # 🔮 Ripple Engine
    ├── wealth_ui.py            # 📉 Wealth Logic
    ├── brain_map_ui.py         # 🧠 Brain Map — semantic network
    └── placeholder_ui.py       # Base template
```

---

## Installation

```bash
# 1. Navigate to project
cd "C:\ai project 3\debate-bot"

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
venv\Scripts\python.exe -m pip install scikit-learn

# 4. Create .env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here

# 5. Run
streamlit run app.py
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit 1.38+ |
| LLM Orchestration | LangChain + LangGraph |
| LLM Providers | Groq / OpenAI / Anthropic / Google |
| Vector Database | ChromaDB (persistent, local) |
| Semantic Similarity | scikit-learn TF-IDF + cosine similarity |
| Visualisation | Plotly graph_objects |
| Graph Layout | Custom NumPy Fruchterman-Reingold |
| Web Search | DuckDuckGo Search |
| Environment | python-dotenv |
