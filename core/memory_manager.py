"""
core/memory_manager.py
======================
Central memory layer for the Cognitive OS.

Manages four completely isolated ChromaDB collections — one per operating mode.
No query, write, or delete operation can touch a collection it was not
explicitly routed to. Cross-collection contamination is architecturally
impossible: each collection is a separate ChromaDB namespace with its own
embedding space, metadata schema, and access methods.

Collection map
──────────────
  MODE               COLLECTION NAME          PURPOSE
  devils_advocate    debate_memory            Argument history + winning tactics
  execution          execution_strategies     Task/energy patterns + completions
  ripple             ripple_consequences      Decision trees + consequence chains
  wealth             wealth_reasoning         Financial logic + opportunity costs

Usage
─────
    from core.memory_manager import MemoryManager, format_debate_memories

    mm = MemoryManager()

    # Write
    mm.save_debate_exchange(user_input, bot_response)
    mm.save_task(task, energy_level="high", outcome="completed")
    mm.save_decision(decision, consequences, blindspots)
    mm.save_financial_entry(scenario, reasoning, opportunity_cost, risk_level)

    # Query — always scoped to one collection
    mm.query(mode="devils_advocate", text="inflation causes unemployment")
    mm.query(mode="wealth",          text="real estate vs index funds")

    # Route directly to ChromaDB collection object
    col = mm.get_collection("ripple")
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

Mode = Literal["devils_advocate", "execution", "ripple", "wealth"]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DEFAULT_TOP_K  = int(os.getenv("MEMORY_TOP_K", "4"))

# Relevance threshold per collection — tuned to each use case's semantic
# density. Debate arguments need tight matching; task patterns benefit
# from broader recall.
_RELEVANCE_THRESHOLD: dict[str, float] = {
    "debate_memory"        : 0.28,   # tight — argument matching is precise
    "execution_strategies" : 0.20,   # broad — energy patterns recur loosely
    "ripple_consequences"  : 0.22,   # medium — consequence chains vary
    "wealth_reasoning"     : 0.25,   # medium — financial scenarios cluster
}

# Collection name constants — centralised so a typo can never silently
# create a ghost collection at runtime.
_COL_DEBATE    = "debate_memory"
_COL_EXECUTION = "execution_strategies"
_COL_RIPPLE    = "ripple_consequences"
_COL_WEALTH    = "wealth_reasoning"

# Maps the UI mode string to internal collection name.
# This is the ONLY place where that mapping is defined.
_MODE_TO_COLLECTION: dict[str, str] = {
    "devils_advocate": _COL_DEBATE,
    "execution"      : _COL_EXECUTION,
    "ripple"         : _COL_RIPPLE,
    "wealth"         : _COL_WEALTH,
}


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Single entrypoint for all persistent memory operations across the
    Cognitive OS. One instance is created at app startup and injected
    into every LangGraph pipeline that needs memory.

    Isolation guarantee
    -------------------
    ChromaDB's get_or_create_collection with distinct names creates
    fully separate HNSW index spaces. Queries against one collection
    physically cannot reach documents in another — there is no shared
    index or shared embedding namespace between collections.

    This class enforces an additional software-level guarantee: every
    public method is explicitly scoped to exactly one collection. There
    is no method that accepts a collection name as a free-form string
    from untrusted input — routing always goes through _MODE_TO_COLLECTION.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id   = session_id or str(uuid.uuid4())[:8]
        self.client       = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embedding_fn = DefaultEmbeddingFunction()

        # Initialise all four collections at startup.
        # Using get_or_create ensures idempotency across restarts.
        # cosine is the correct distance metric for semantic text matching.
        self._debate    = self._init_collection(_COL_DEBATE)
        self._execution = self._init_collection(_COL_EXECUTION)
        self._ripple    = self._init_collection(_COL_RIPPLE)
        self._wealth    = self._init_collection(_COL_WEALTH)

        # Internal registry — the ONLY place where collection objects live.
        # All routing goes through this dict, never around it.
        self._registry: dict[str, chromadb.Collection] = {
            _COL_DEBATE    : self._debate,
            _COL_EXECUTION : self._execution,
            _COL_RIPPLE    : self._ripple,
            _COL_WEALTH    : self._wealth,
        }

        self._log_startup()

    # =========================================================================
    # ROUTING — public interface
    # =========================================================================

    def get_collection(self, mode: str) -> chromadb.Collection:
        """
        Returns the ChromaDB collection object for the given mode.
        This is the SINGLE routing method — every other method calls this.

        Args:
            mode: One of "devils_advocate" | "execution" | "ripple" | "wealth"

        Returns:
            The ChromaDB Collection bound to that mode.

        Raises:
            ValueError if an unknown mode string is passed. Silent fallback
            would cause data corruption, so we fail loudly.
        """
        col_name = _MODE_TO_COLLECTION.get(mode)
        if col_name is None:
            valid = list(_MODE_TO_COLLECTION.keys())
            raise ValueError(
                f"[MemoryManager] Unknown mode '{mode}'. Valid modes: {valid}"
            )
        return self._registry[col_name]

    def query(
        self,
        mode    : str,
        text    : str,
        top_k   : int = DEFAULT_TOP_K,
        filters : Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic similarity search scoped strictly to one mode's collection.

        Args:
            mode    : Which collection to search — hard-routed, no bleed-over.
            text    : The query string (embedded on the fly).
            top_k   : Max results before threshold filtering.
            filters : Optional ChromaDB where clause for metadata filtering,
                      e.g. {"energy_level": {"$eq": "high"}}

        Returns:
            List of result dicts: {id, document, metadata, relevance}.
            Only results above the collection-specific relevance threshold
            are returned. Documents below threshold are silently dropped.
        """
        col       = self.get_collection(mode)
        col_name  = _MODE_TO_COLLECTION[mode]
        threshold = _RELEVANCE_THRESHOLD[col_name]
        total     = col.count()

        if total == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_texts": [text],
            "n_results"  : min(top_k, total),
            "include"    : ["metadatas", "documents", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters

        results = col.query(**query_kwargs)

        hits = []
        for doc, meta, dist in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas",  [[]])[0],
            results.get("distances",  [[]])[0],
        ):
            relevance = round(1 - (dist / 2), 4)
            if relevance >= threshold:
                hits.append({
                    "document" : doc,
                    "metadata" : meta,
                    "relevance": relevance,
                })

        if hits:
            print(f"[memory:{mode}] Retrieved {len(hits)} result(s)")
        return hits

    def delete_entry(self, mode: str, entry_id: str) -> None:
        """Deletes a single entry by ID from the correct collection only."""
        col = self.get_collection(mode)
        col.delete(ids=[entry_id])
        print(f"[memory:{mode}] Deleted entry {entry_id}")

    def clear_collection(self, mode: str) -> None:
        """
        Wipes all data from one collection without touching the other three.
        The collection is deleted and re-created — ChromaDB has no truncate.
        Safe to call even if the collection was already deleted externally
        (e.g. by DebateMemory.clear_all() in brain.py / memory.py).
        """
        col_name = _MODE_TO_COLLECTION[mode]
        try:
            self.client.delete_collection(col_name)
        except Exception:
            # Already gone — that's fine, we'll re-create below
            pass
        new_col = self._init_collection(col_name)
        self._registry[col_name] = new_col
        attr_map = {
            _COL_DEBATE   : "_debate",
            _COL_EXECUTION: "_execution",
            _COL_RIPPLE   : "_ripple",
            _COL_WEALTH   : "_wealth",
        }
        setattr(self, attr_map[col_name], new_col)
        print(f"[memory:{mode}] Collection '{col_name}' cleared")

    def get_stats(self) -> dict[str, int]:
        """
        Returns document counts for all four collections.
        Self-healing: if a collection reference is stale (deleted externally
        by DebateMemory.clear_all()), it silently re-creates the collection
        instead of raising NotFoundError.
        """
        def _safe_count(attr: str, col_name: str) -> int:
            col = getattr(self, attr)
            try:
                return col.count()
            except Exception:
                # Collection was deleted externally — re-create and re-register
                new_col = self._init_collection(col_name)
                setattr(self, attr, new_col)
                self._registry[col_name] = new_col
                return 0

        return {
            "devils_advocate": _safe_count("_debate",    _COL_DEBATE),
            "execution"      : _safe_count("_execution", _COL_EXECUTION),
            "ripple"         : _safe_count("_ripple",    _COL_RIPPLE),
            "wealth"         : _safe_count("_wealth",    _COL_WEALTH),
        }

    # =========================================================================
    # MODE 1 — DEVIL'S ADVOCATE  (debate_memory)
    # =========================================================================
    # Metadata schema:
    #   user_arg      str   — the user's original claim
    #   bot_response  str   — the advocate's reply (truncated to 500 chars)
    #   tactic_used   str   — rhetorical tactic deployed by the bot
    #   severity      str   — LOW | MEDIUM | HIGH (from contradiction node)
    #   concession    str   — "yes" | "no"
    #   type          str   — "exchange" | "tactic_win"
    #   session_id    str
    #   timestamp     ISO 8601

    def save_debate_exchange(
        self,
        user_arg     : str,
        bot_response : str,
        tactic_used  : str = "",
        severity     : str = "",
        concession   : bool = False,
    ) -> str:
        """Saves one full debate turn to the debate collection."""
        entry_id = str(uuid.uuid4())
        document = (
            f"USER CLAIM: {user_arg.strip()} | "
            f"BOT RESPONSE: {bot_response.strip()[:400]} | "
            f"TACTIC: {tactic_used} | SEVERITY: {severity}"
        )
        self._debate.add(
            documents=[document],
            metadatas=[{
                "type"        : "exchange",
                "user_arg"    : user_arg.strip(),
                "bot_response": bot_response.strip()[:500],
                "tactic_used" : tactic_used,
                "severity"    : severity,
                "concession"  : "yes" if concession else "no",
                "session_id"  : self.session_id,
                "timestamp"   : _now(),
            }],
            ids=[entry_id],
        )
        print(f"[memory:debate] Exchange #{self._debate.count()} saved")
        return entry_id

    def save_winning_tactic(
        self,
        tactic_name    : str,
        topic          : str,
        concession_text: str,
        bot_argument   : str = "",
    ) -> str:
        """
        Saves a tactic that caused a concession. Tagged as type=tactic_win
        so it can be retrieved separately from regular exchange history
        without any risk of the two record types blending in query results.
        """
        entry_id = str(uuid.uuid4())
        document = (
            f"WINNING TACTIC: {tactic_name} | TOPIC: {topic} | "
            f"USER CONCEDED: {concession_text.strip()} | "
            f"ARGUMENT: {bot_argument.strip()[:300]}"
        )
        self._debate.add(
            documents=[document],
            metadatas=[{
                "type"           : "tactic_win",
                "tactic_name"    : tactic_name,
                "topic"          : topic,
                "concession_text": concession_text.strip()[:300],
                "bot_argument"   : bot_argument.strip()[:300],
                "session_id"     : self.session_id,
                "timestamp"      : _now(),
            }],
            ids=[entry_id],
        )
        print(f"[memory:debate] Tactic win saved: '{tactic_name}'")
        return entry_id

    def query_debate_history(self, text: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Past argument exchanges relevant to the current claim."""
        return self.query(
            "devils_advocate", text, top_k,
            filters={"type": {"$eq": "exchange"}},
        )

    def query_winning_tactics(self, topic: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Past winning tactics relevant to the current topic."""
        return self.query(
            "devils_advocate", topic, top_k,
            filters={"type": {"$eq": "tactic_win"}},
        )

    # =========================================================================
    # MODE 2 — EXECUTION ARCHITECT  (execution_strategies)
    # =========================================================================
    # Metadata schema:
    #   task               str   — the task description
    #   energy_level       str   — "high" | "medium" | "low"
    #   time_of_day        str   — "morning" | "afternoon" | "evening" | "night"
    #   estimated_minutes  int   — estimated duration
    #   actual_minutes     int   — actual duration (0 if pending)
    #   completion_status  str   — "completed" | "partial" | "abandoned" | "pending"
    #   category           str   — "deep_work" | "admin" | "creative" | "comms" | "other"
    #   session_id         str
    #   timestamp          ISO 8601

    def save_task(
        self,
        task              : str,
        energy_level      : str,
        time_of_day       : str = "",
        estimated_minutes : int = 0,
        actual_minutes    : int = 0,
        completion_status : str = "pending",
        category          : str = "other",
    ) -> str:
        """Saves a task with its energy context and completion outcome."""
        _validate("energy_level",      energy_level,
                  ["high", "medium", "low"])
        _validate("completion_status", completion_status,
                  ["completed", "partial", "abandoned", "pending"])

        entry_id = str(uuid.uuid4())
        document = (
            f"TASK: {task.strip()} | ENERGY: {energy_level} | "
            f"TIME: {time_of_day} | CATEGORY: {category} | "
            f"STATUS: {completion_status}"
        )
        self._execution.add(
            documents=[document],
            metadatas=[{
                "task"              : task.strip(),
                "energy_level"      : energy_level,
                "time_of_day"       : time_of_day,
                "estimated_minutes" : estimated_minutes,
                "actual_minutes"    : actual_minutes,
                "completion_status" : completion_status,
                "category"          : category,
                "session_id"        : self.session_id,
                "timestamp"         : _now(),
            }],
            ids=[entry_id],
        )
        print(f"[memory:execution] Task saved: '{task[:50]}' [{energy_level} energy]")
        return entry_id

    def query_tasks_by_energy(
        self,
        energy_level : str,
        context      : str = "",
        top_k        : int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """Past tasks at a given energy level, optionally filtered by context."""
        _validate("energy_level", energy_level, ["high", "medium", "low"])
        return self.query(
            "execution", context or f"tasks at {energy_level} energy", top_k,
            filters={"energy_level": {"$eq": energy_level}},
        )

    def query_completed_tasks(
        self, context: str = "", top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """Successfully completed tasks — used to surface positive patterns."""
        return self.query(
            "execution", context or "completed tasks", top_k,
            filters={"completion_status": {"$eq": "completed"}},
        )

    def update_task_completion(
        self,
        entry_id       : str,
        status         : str,
        actual_minutes : int = 0,
    ) -> None:
        """
        Updates completion status after a task is executed.
        ChromaDB has no partial update — fetch, mutate, upsert.
        """
        _validate("status", status, ["completed", "partial", "abandoned", "pending"])
        existing = self._execution.get(
            ids=[entry_id], include=["metadatas", "documents"]
        )
        if not existing["ids"]:
            print(f"[memory:execution] Entry {entry_id} not found")
            return
        meta = existing["metadatas"][0]
        doc  = existing["documents"][0]
        meta["completion_status"] = status
        meta["actual_minutes"]    = actual_minutes
        self._execution.upsert(documents=[doc], metadatas=[meta], ids=[entry_id])
        print(f"[memory:execution] Task {entry_id[:8]}... updated to: {status}")

    # =========================================================================
    # MODE 3 — RIPPLE ENGINE  (ripple_consequences)
    # =========================================================================
    # Metadata schema:
    #   decision     str   — the decision being evaluated
    #   consequence  str   — the specific consequence mapped
    #   order        int   — depth: 1=direct, 2=secondary, 3=tertiary
    #   domain       str   — "financial" | "social" | "career" | "health" | "other"
    #   probability  str   — "certain" | "likely" | "possible" | "unlikely"
    #   valence      str   — "positive" | "negative" | "neutral" | "ambiguous"
    #   blindspot    str   — hidden assumption or overlooked factor (empty if none)
    #   session_id   str
    #   timestamp    ISO 8601

    def save_decision(
        self,
        decision    : str,
        consequence : str,
        order       : int  = 1,
        domain      : str  = "other",
        probability : str  = "possible",
        valence     : str  = "neutral",
        blindspot   : str  = "",
    ) -> str:
        """
        Saves one consequence node of a decision tree.
        Call multiple times with different order values to build the full
        ripple map: order=1 (direct), 2 (secondary), 3 (tertiary).
        """
        _validate("order",       str(order),  ["1", "2", "3"])
        _validate("probability", probability, ["certain", "likely", "possible", "unlikely"])
        _validate("valence",     valence,     ["positive", "negative", "neutral", "ambiguous"])

        entry_id = str(uuid.uuid4())
        document = (
            f"DECISION: {decision.strip()} | "
            f"ORDER-{order} CONSEQUENCE: {consequence.strip()} | "
            f"DOMAIN: {domain} | PROBABILITY: {probability} | "
            f"VALENCE: {valence} | "
            f"BLINDSPOT: {blindspot.strip() or 'none'}"
        )
        self._ripple.add(
            documents=[document],
            metadatas=[{
                "decision"   : decision.strip(),
                "consequence": consequence.strip(),
                "order"      : order,
                "domain"     : domain,
                "probability": probability,
                "valence"    : valence,
                "blindspot"  : blindspot.strip(),
                "session_id" : self.session_id,
                "timestamp"  : _now(),
            }],
            ids=[entry_id],
        )
        print(f"[memory:ripple] Order-{order} consequence saved for: '{decision[:50]}'")
        return entry_id

    def query_past_decisions(
        self,
        current_decision : str,
        top_k            : int = DEFAULT_TOP_K,
        order_filter     : Optional[int] = None,
    ) -> list[dict]:
        """Past decisions similar to the current one. Optional order filter."""
        filters = None
        if order_filter is not None:
            filters = {"order": {"$eq": order_filter}}
        return self.query("ripple", current_decision, top_k, filters=filters)

    def query_blindspots(self, context: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Past blindspots relevant to the current decision context."""
        results = self.query("ripple", context, top_k)
        return [r for r in results if r["metadata"].get("blindspot")]

    def query_negative_consequences(
        self, decision: str, top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """Historically negative consequences from similar decisions."""
        return self.query(
            "ripple", decision, top_k,
            filters={"valence": {"$eq": "negative"}},
        )

    # =========================================================================
    # MODE 4 — WEALTH LOGIC  (wealth_reasoning)
    # =========================================================================
    # Metadata schema:
    #   scenario          str   — the financial scenario or question
    #   reasoning         str   — the AI's full analysis (truncated to 600 chars)
    #   decision_made     str   — what the user decided (if known)
    #   opportunity_cost  str   — what was given up by choosing this path
    #   risk_level        str   — "low" | "medium" | "high" | "critical"
    #   time_horizon      str   — "immediate" | "short" | "medium" | "long"
    #   asset_class       str   — "equity" | "real_estate" | "crypto" | "cash"
    #                             | "business" | "education" | "other"
    #   outcome           str   — "positive" | "negative" | "neutral" | "unknown"
    #   session_id        str
    #   timestamp         ISO 8601

    def save_financial_entry(
        self,
        scenario         : str,
        reasoning        : str,
        opportunity_cost : str = "",
        risk_level       : str = "medium",
        time_horizon     : str = "medium",
        asset_class      : str = "other",
        decision_made    : str = "",
        outcome          : str = "unknown",
    ) -> str:
        """Saves a financial reasoning session with full context metadata."""
        _validate("risk_level",   risk_level,
                  ["low", "medium", "high", "critical"])
        _validate("time_horizon", time_horizon,
                  ["immediate", "short", "medium", "long"])
        _validate("outcome",      outcome,
                  ["positive", "negative", "neutral", "unknown"])

        entry_id = str(uuid.uuid4())
        document = (
            f"SCENARIO: {scenario.strip()} | "
            f"REASONING: {reasoning.strip()[:400]} | "
            f"OPPORTUNITY COST: {opportunity_cost.strip() or 'not assessed'} | "
            f"RISK: {risk_level} | HORIZON: {time_horizon} | "
            f"ASSET: {asset_class}"
        )
        self._wealth.add(
            documents=[document],
            metadatas=[{
                "scenario"        : scenario.strip(),
                "reasoning"       : reasoning.strip()[:600],
                "opportunity_cost": opportunity_cost.strip(),
                "risk_level"      : risk_level,
                "time_horizon"    : time_horizon,
                "asset_class"     : asset_class,
                "decision_made"   : decision_made.strip(),
                "outcome"         : outcome,
                "session_id"      : self.session_id,
                "timestamp"       : _now(),
            }],
            ids=[entry_id],
        )
        print(
            f"[memory:wealth] Financial entry saved: "
            f"'{scenario[:50]}' [risk={risk_level}]"
        )
        return entry_id

    def query_financial_history(
        self, scenario: str, top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """Past financial scenarios semantically similar to the current one."""
        return self.query("wealth", scenario, top_k)

    def query_by_risk(
        self, risk_level: str, context: str = "", top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """Past decisions at a given risk level."""
        _validate("risk_level", risk_level, ["low", "medium", "high", "critical"])
        return self.query(
            "wealth", context or f"{risk_level} risk financial decisions", top_k,
            filters={"risk_level": {"$eq": risk_level}},
        )

    def query_by_asset_class(
        self, asset_class: str, context: str = "", top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """Past reasoning for a specific asset class."""
        return self.query(
            "wealth", context or f"{asset_class} investment reasoning", top_k,
            filters={"asset_class": {"$eq": asset_class}},
        )

    def query_opportunity_costs(
        self, scenario: str, top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """Past entries where opportunity cost was explicitly assessed."""
        results = self.query("wealth", scenario, top_k)
        return [r for r in results if r["metadata"].get("opportunity_cost")]

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _init_collection(self, name: str) -> chromadb.Collection:
        """Creates or opens a named collection. Never touches any other."""
        return self.client.get_or_create_collection(
            name               = name,
            embedding_function = self.embedding_fn,
            metadata           = {"hnsw:space": "cosine"},
        )

    def _log_startup(self) -> None:
        stats = self.get_stats()
        print(
            f"\n[MemoryManager] ChromaDB ready — {CHROMA_DB_PATH}\n"
            f"  Devil's Advocate : {stats['devils_advocate']:>4} entries\n"
            f"  Execution        : {stats['execution']:>4} entries\n"
            f"  Ripple Engine    : {stats['ripple']:>4} entries\n"
            f"  Wealth Logic     : {stats['wealth']:>4} entries\n"
            f"  Session ID       : {self.session_id}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _validate(field: str, value: str, allowed: list[str]) -> None:
    """
    Hard enum validation for metadata fields.
    Raises ValueError rather than silently storing invalid metadata
    that would corrupt future filter queries.
    """
    if value not in allowed:
        raise ValueError(
            f"[MemoryManager] Invalid '{field}': '{value}'. "
            f"Allowed: {allowed}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Format helpers — called by graph nodes to build LLM prompt context
# ─────────────────────────────────────────────────────────────────────────────

def format_debate_memories(results: list[dict]) -> str:
    """Formats debate history for injection into the Advocate system prompt."""
    if not results:
        return ""
    lines = ["RELEVANT PAST ARGUMENTS:"]
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        lines.append(
            f"\n[Memory {i}] Relevance: {r['relevance']:.0%} | {m.get('timestamp','')[:10]}\n"
            f"  User claimed : \"{m.get('user_arg', '')}\""
            f"\n  Bot responded: \"{m.get('bot_response', '')[:200]}...\""
        )
    return "\n".join(lines)


def format_winning_tactics(results: list[dict]) -> str:
    """Formats tactic wins for injection into the Advocate system prompt."""
    if not results:
        return ""
    lines = ["PROVEN WINNING TACTICS:"]
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        lines.append(
            f"\n[Tactic {i}] '{m.get('tactic_name','')}' "
            f"| Topic: {m.get('topic','General')} "
            f"| Relevance: {r['relevance']:.0%}\n"
            f"  Concession: \"{m.get('concession_text','')[:150]}\""
        )
    return "\n".join(lines)


def format_task_context(results: list[dict]) -> str:
    """Formats past task data for the Execution Architect prompt."""
    if not results:
        return ""
    lines = ["RELEVANT PAST TASKS:"]
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        lines.append(
            f"\n[Task {i}] Energy: {m.get('energy_level','')} | "
            f"Status: {m.get('completion_status','')} | "
            f"Category: {m.get('category','')}\n"
            f"  Task: \"{m.get('task','')}\""
        )
    return "\n".join(lines)


def format_ripple_context(results: list[dict]) -> str:
    """Formats past consequence data for the Ripple Engine prompt."""
    if not results:
        return ""
    lines = ["RELEVANT PAST CONSEQUENCES:"]
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        bs = f"\n  Blindspot: \"{m.get('blindspot','')}\""  if m.get("blindspot") else ""
        lines.append(
            f"\n[{i}] Order-{m.get('order',1)} | Domain: {m.get('domain','')} | "
            f"Prob: {m.get('probability','')} | Valence: {m.get('valence','')}\n"
            f"  Decision : \"{m.get('decision','')[:120]}\"\n"
            f"  Outcome  : \"{m.get('consequence','')[:200]}\""
            + bs
        )
    return "\n".join(lines)


def format_financial_context(results: list[dict]) -> str:
    """Formats past financial reasoning for the Wealth Logic prompt."""
    if not results:
        return ""
    lines = ["RELEVANT PAST FINANCIAL REASONING:"]
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        lines.append(
            f"\n[Entry {i}] Risk: {m.get('risk_level','')} | "
            f"Horizon: {m.get('time_horizon','')} | "
            f"Asset: {m.get('asset_class','')} | "
            f"Outcome: {m.get('outcome','unknown')}\n"
            f"  Scenario  : \"{m.get('scenario','')[:150]}\"\n"
            f"  Opp. Cost : \"{m.get('opportunity_cost','not assessed')}\"" 
        )
    return "\n".join(lines)
