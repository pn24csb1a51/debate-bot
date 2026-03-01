"""
memory.py  ·  Elite Upgrade – Bandit Scoring + Hybrid Strategy Ranking
=======================================================================
Collection 1: 'debate_memory'     — every (user_claim, bot_response) pair
Collection 2: 'strategy_feedback' — tactics with full performance tracking

Elite upgrades:
  ① Bandit Scoring   — tracks total_attempts, success_count, success_rate
                        per tactic so performance compounds over time
  ② Hybrid Ranking   — final score = (similarity × 0.6) + (success_rate × 0.4)
                        balances topical relevance WITH proven effectiveness

ChromaDB limitation note:
  ChromaDB does not support in-place metadata updates on existing documents.
  The bandit pattern is implemented by storing one record per tactic NAME
  using a deterministic ID (hash of tactic name) so upsert = overwrite.
  On each new attempt the full updated metadata is re-written.
"""

import hashlib
import os
import uuid
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CHROMA_DB_PATH      = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DEBATE_COLLECTION   = "debate_memory"
STRATEGY_COLLECTION = "strategy_feedback"
TOP_K_RESULTS       = int(os.getenv("MEMORY_TOP_K",    "3"))
TOP_K_STRATEGIES    = int(os.getenv("STRATEGY_TOP_K",  "3"))

# Hybrid ranking weights
SIMILARITY_WEIGHT   = float(os.getenv("SIMILARITY_WEIGHT", "0.6"))
SUCCESS_RATE_WEIGHT = float(os.getenv("SUCCESS_RATE_WEIGHT", "0.4"))


def _tactic_id(tactic_name: str) -> str:
    """
    Deterministic ID based on tactic name (lowercased + stripped).
    Ensures the same tactic always maps to the same ChromaDB document,
    enabling the upsert/overwrite pattern for bandit tracking.
    """
    key = tactic_name.lower().strip()
    return "tactic_" + hashlib.md5(key.encode()).hexdigest()[:16]


class DebateMemory:
    """
    Manages two persistent ChromaDB collections.

    Collection 1 — debate_memory:
        Standard episodic memory of every debate exchange.

    Collection 2 — strategy_feedback (ELITE):
        One document per unique tactic name.
        Tracks: total_attempts, success_count, success_rate.
        On each use: total_attempts += 1.
        On concession detected: success_count += 1, success_rate recalculated.
        Retrieval uses a HYBRID score blending cosine similarity + success_rate.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id   = session_id or str(uuid.uuid4())[:8]
        self.client       = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embedding_fn = DefaultEmbeddingFunction()

        self.debate_collection = self.client.get_or_create_collection(
            name=DEBATE_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.strategy_collection = self.client.get_or_create_collection(
            name=STRATEGY_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        print(
            f"[memory] 🗄️  ChromaDB ready\n"
            f"         📚 Debate memories   : {self.debate_collection.count()}\n"
            f"         🏆 Tracked tactics   : {self.strategy_collection.count()}\n"
            f"         🔑 Session           : {self.session_id}"
        )

    # ═══════════════════════════════════════════════════════
    # COLLECTION 1 — Debate Memory  (unchanged interface)
    # ═══════════════════════════════════════════════════════

    def store_argument(self, user_input: str, ai_response: str) -> str:
        """Saves a (user_argument, bot_response) pair as one memory unit."""
        memory_id = str(uuid.uuid4())
        document  = (
            f"USER CLAIM: {user_input.strip()} "
            f"| BOT RESPONSE: {ai_response.strip()}"
        )
        self.debate_collection.add(
            documents=[document],
            metadatas=[{
                "user_arg"    : user_input.strip(),
                "bot_response": ai_response.strip()[:500],
                "timestamp"   : datetime.now().isoformat(),
                "session_id"  : self.session_id,
            }],
            ids=[memory_id],
        )
        print(f"[memory] 💾 Stored debate memory #{self.debate_collection.count()}")
        return memory_id

    def query_past_arguments(
        self,
        user_input: str,
        top_k: int = TOP_K_RESULTS,
    ) -> list[dict]:
        """Returns top-k semantically relevant past debate exchanges."""
        total = self.debate_collection.count()
        if total == 0:
            return []

        results = self.debate_collection.query(
            query_texts=[user_input],
            n_results=min(top_k, total),
            include=["metadatas", "distances"],
        )
        memories = []
        for meta, dist in zip(
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            relevance = round(1 - (dist / 2), 3)
            if relevance >= 0.25:
                memories.append({
                    "user_arg"    : meta.get("user_arg", ""),
                    "bot_response": meta.get("bot_response", ""),
                    "timestamp"   : meta.get("timestamp", ""),
                    "session_id"  : meta.get("session_id", ""),
                    "relevance"   : relevance,
                })
        if memories:
            print(f"[memory] 🔍 Retrieved {len(memories)} relevant past debate exchange(s)")
        return memories

    def format_memories_for_prompt(self, memories: list[dict]) -> str:
        """Formats debate memories for system prompt injection."""
        if not memories:
            return ""
        lines = ["📚 RELEVANT PAST ARGUMENTS FROM THIS USER:"]
        for i, mem in enumerate(memories, 1):
            lines.append(
                f"\n[Memory {i}] (Relevance: {mem['relevance']:.0%} | "
                f"Date: {mem['timestamp'][:10]})\n"
                f"  • User previously claimed : \"{mem['user_arg']}\"\n"
                f"  • You responded with      : \"{mem['bot_response'][:200]}…\""
            )
        lines.append(
            "\nUse these to identify CONTRADICTIONS, PATTERNS, or REPEATED FALLACIES."
        )
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════
    # COLLECTION 2 — Strategy Feedback  (ELITE BANDIT MODEL)
    # ═══════════════════════════════════════════════════════

    def record_tactic_attempt(
        self,
        tactic_name      : str,
        topic            : str = "",
        bot_argument_used: str = "",
    ) -> None:
        """
        Call this every time the bot USES a tactic (win or not).
        Increments total_attempts by 1.
        Uses deterministic ID so the same tactic document is always updated.
        """
        doc_id   = _tactic_id(tactic_name)
        existing = self._get_tactic_record(doc_id)

        if existing:
            attempts     = existing["total_attempts"] + 1
            successes    = existing["success_count"]
            success_rate = round(successes / attempts, 4)
            document     = self._build_tactic_document(
                tactic_name, topic or existing["topic"],
                bot_argument_used or existing["bot_argument_used"],
                existing["first_used"],
            )
            # Overwrite with updated counts
            self.strategy_collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[{
                    **existing,
                    "total_attempts": attempts,
                    "success_rate"  : success_rate,
                    "last_used"     : datetime.now().isoformat(),
                }],
            )
            print(
                f"[memory] 📈 Tactic '{tactic_name}' attempt #{attempts} "
                f"· Success rate: {success_rate:.0%}"
            )
        else:
            # First time this tactic is seen — create the record
            now      = datetime.now().isoformat()
            document = self._build_tactic_document(
                tactic_name, topic, bot_argument_used, now
            )
            self.strategy_collection.add(
                documents=[document],
                metadatas=[{
                    "tactic_name"      : tactic_name,
                    "topic"            : topic,
                    "bot_argument_used": bot_argument_used[:300],
                    "total_attempts"   : 1,
                    "success_count"    : 0,
                    "success_rate"     : 0.0,
                    "first_used"       : now,
                    "last_used"        : now,
                    "session_id"       : self.session_id,
                }],
                ids=[doc_id],
            )
            print(f"[memory] 🆕 New tactic registered: '{tactic_name}'")

    def log_winning_strategy(
        self,
        tactic_name          : str,
        user_concession_text : str,
        bot_argument_used    : str = "",
        topic                : str = "",
    ) -> None:
        """
        Call this when a concession IS detected.
        Increments both total_attempts AND success_count, recalculates success_rate.
        Also stores the concession text for qualitative review.
        """
        doc_id   = _tactic_id(tactic_name)
        existing = self._get_tactic_record(doc_id)

        if existing:
            attempts     = existing["total_attempts"] + 1
            successes    = existing["success_count"]  + 1
            success_rate = round(successes / attempts, 4)
            document     = self._build_tactic_document(
                tactic_name,
                topic or existing["topic"],
                bot_argument_used or existing["bot_argument_used"],
                existing["first_used"],
            )
            self.strategy_collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[{
                    **existing,
                    "total_attempts"      : attempts,
                    "success_count"       : successes,
                    "success_rate"        : success_rate,
                    "last_concession_text": user_concession_text[:300],
                    "last_used"           : datetime.now().isoformat(),
                }],
            )
            print(
                f"[memory] 🏆 Tactic '{tactic_name}' won! "
                f"Successes: {successes}/{attempts} "
                f"({success_rate:.0%} success rate)"
            )
        else:
            # First encounter AND it's already a win
            now      = datetime.now().isoformat()
            document = self._build_tactic_document(
                tactic_name, topic, bot_argument_used, now
            )
            self.strategy_collection.add(
                documents=[document],
                metadatas=[{
                    "tactic_name"         : tactic_name,
                    "topic"               : topic,
                    "bot_argument_used"   : bot_argument_used[:300],
                    "last_concession_text": user_concession_text[:300],
                    "total_attempts"      : 1,
                    "success_count"       : 1,
                    "success_rate"        : 1.0,
                    "first_used"          : now,
                    "last_used"           : now,
                    "session_id"          : self.session_id,
                }],
                ids=[doc_id],
            )
            print(f"[memory] 🏆 New tactic '{tactic_name}' registered as immediate win!")

    def query_winning_strategies(
        self,
        current_topic: str,
        top_k: int = TOP_K_STRATEGIES,
    ) -> list[dict]:
        """
        ── ELITE HYBRID RANKING ──────────────────────────────────────────────
        Retrieves tactics sorted by a HYBRID score:

            score = (cosine_similarity × SIMILARITY_WEIGHT)
                  + (success_rate      × SUCCESS_RATE_WEIGHT)

        Default weights: similarity=0.6, success_rate=0.4

        This ensures the bot selects tactics that are:
          • Topically relevant to the current debate (similarity)
          • AND historically proven against this user (success_rate)

        A tactic with 0.90 similarity but 0% success rate scores:
            (0.90 × 0.6) + (0.00 × 0.4) = 0.54

        A tactic with 0.70 similarity but 80% success rate scores:
            (0.70 × 0.6) + (0.80 × 0.4) = 0.74  ← ranked higher
        ─────────────────────────────────────────────────────────────────────
        """
        total = self.strategy_collection.count()
        if total == 0:
            return []

        # Fetch more candidates than needed — re-rank after hybrid scoring
        fetch_k = min(max(top_k * 3, 9), total)

        results = self.strategy_collection.query(
            query_texts=[current_topic],
            n_results=fetch_k,
            include=["metadatas", "distances"],
        )

        candidates = []
        for meta, dist in zip(
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            similarity   = round(1 - (dist / 2), 4)
            success_rate = float(meta.get("success_rate", 0.0))
            hybrid_score = round(
                (similarity   * SIMILARITY_WEIGHT) +
                (success_rate * SUCCESS_RATE_WEIGHT),
                4,
            )

            if similarity >= 0.15:   # loose pre-filter; hybrid score handles the rest
                candidates.append({
                    "tactic_name"      : meta.get("tactic_name",       ""),
                    "topic"            : meta.get("topic",              ""),
                    "bot_argument_used": meta.get("bot_argument_used",  ""),
                    "total_attempts"   : int(meta.get("total_attempts",  0)),
                    "success_count"    : int(meta.get("success_count",   0)),
                    "success_rate"     : success_rate,
                    "similarity"       : similarity,
                    "hybrid_score"     : hybrid_score,
                    "timestamp"        : meta.get("last_used",          ""),
                })

        # Sort by hybrid score descending and return top_k
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top = candidates[:top_k]

        if top:
            print(
                f"[memory] 🎯 Top tactic: '{top[0]['tactic_name']}' "
                f"hybrid={top[0]['hybrid_score']:.2f} "
                f"(sim={top[0]['similarity']:.2f}, "
                f"win%={top[0]['success_rate']:.0%})"
            )
        return top

    def format_strategies_for_prompt(self, strategies: list[dict]) -> str:
        """Formats ranked strategies for system prompt injection."""
        if not strategies:
            return ""
        lines = [
            "🏆 HIGH-IMPACT TACTICS — ranked by hybrid score (relevance + win rate):",
            "Deploy these tactics EARLY. They are proven to work on this user.\n",
        ]
        for i, s in enumerate(strategies, 1):
            win_pct = f"{s['success_rate']:.0%}"
            lines.append(
                f"[Tactic {i}] '{s['tactic_name']}'\n"
                f"  • Hybrid score   : {s['hybrid_score']:.2f}  "
                f"(relevance={s['similarity']:.2f}, win rate={win_pct})\n"
                f"  • Used           : {s['total_attempts']}× · "
                f"Won: {s['success_count']}× ({win_pct})\n"
                f"  • Argument used  : \"{s['bot_argument_used'][:150]}…\""
            )
        lines.append(
            "\nIf deploying one of these, state the tactic explicitly in "
            "your ④ CHALLENGE section."
        )
        return "\n".join(lines)

    def get_all_strategies(self) -> list[dict]:
        """Returns all tracked tactics sorted by success_rate desc — for UI display."""
        total = self.strategy_collection.count()
        if total == 0:
            return []
        results = self.strategy_collection.get(include=["metadatas"])
        tactics = [
            {
                "tactic_name"   : m.get("tactic_name",    ""),
                "topic"         : m.get("topic",           ""),
                "total_attempts": int(m.get("total_attempts", 0)),
                "success_count" : int(m.get("success_count",  0)),
                "success_rate"  : float(m.get("success_rate", 0.0)),
                "timestamp"     : m.get("last_used",       ""),
            }
            for m in results.get("metadatas", [])
        ]
        tactics.sort(key=lambda x: x["success_rate"], reverse=True)
        return tactics

    # ═══════════════════════════════════════════════════════
    # UTILS
    # ═══════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        return {
            "total_debate_memories"   : self.debate_collection.count(),
            "total_winning_strategies": self.strategy_collection.count(),
            "session_id"              : self.session_id,
            "db_path"                 : CHROMA_DB_PATH,
        }

    def clear_all(self) -> None:
        for name in (DEBATE_COLLECTION, STRATEGY_COLLECTION):
            self.client.delete_collection(name)
        self.debate_collection = self.client.get_or_create_collection(
            name=DEBATE_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.strategy_collection = self.client.get_or_create_collection(
            name=STRATEGY_COLLECTION,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        print("[memory] 🗑️  All debate memories and strategies cleared.")

    # ── Private helpers ───────────────────────────────────

    def _get_tactic_record(self, doc_id: str) -> dict | None:
        """Fetches an existing tactic record by its deterministic ID."""
        try:
            result = self.strategy_collection.get(
                ids=[doc_id],
                include=["metadatas"],
            )
            metas = result.get("metadatas", [])
            return metas[0] if metas else None
        except Exception:
            return None

    @staticmethod
    def _build_tactic_document(
        tactic_name      : str,
        topic            : str,
        bot_argument_used: str,
        first_used       : str,
    ) -> str:
        """Builds the searchable document string for a tactic record."""
        return (
            f"TACTIC: {tactic_name} | "
            f"TOPIC: {topic} | "
            f"BOT USED: {bot_argument_used[:200]} | "
            f"FIRST USED: {first_used[:10]}"
        )
