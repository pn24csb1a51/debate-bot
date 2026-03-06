"""
ui/brain_map_ui.py
==================
Brain Map — Semantic Network of All Memory

Renders an interactive 2D network graph where every saved entry across
all 4 mode collections is a node. Edges are drawn between nodes whose
text content is semantically similar (TF-IDF cosine similarity > threshold).

Architecture
------------
  1. EXTRACT  — Pull all documents from all 4 ChromaDB collections via .get()
  2. VECTORISE — TF-IDF on the raw document text to build a similarity matrix
  3. LAYOUT   — Force-directed spring layout computed in pure NumPy (no networkx)
  4. RENDER   — Plotly graph_objects: scatter for nodes, lines for edges,
                rich hover tooltips showing metadata per node

Collections → colours
  devils_advocate  →  #e63946  (Red)
  execution        →  #f4a261  (Orange)
  ripple           →  #a8dadc  (Cyan)
  wealth           →  #2a9d8f  (Teal)

Integration
-----------
  from ui.brain_map_ui import render
  render()    ← called by app.py router, no arguments needed
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# Design constants
# ─────────────────────────────────────────────────────────────────────────────
ACCENT = "#c77dff"          # Brain Map's own accent colour (purple)
BG     = "#080808"

MODE_META: dict[str, dict] = {
    "devils_advocate": {
        "colour"  : "#e63946",
        "label"   : "Devil's Advocate",
        "icon"    : "🔴",
        "text_key": "user_arg",        # metadata field with primary text
        "fallback": "document",
    },
    "execution": {
        "colour"  : "#f4a261",
        "label"   : "Execution",
        "icon"    : "📅",
        "text_key": "task",
        "fallback": "document",
    },
    "ripple": {
        "colour"  : "#a8dadc",
        "label"   : "Ripple Engine",
        "icon"    : "🔮",
        "text_key": "decision",
        "fallback": "document",
    },
    "wealth": {
        "colour"  : "#2a9d8f",
        "label"   : "Wealth Logic",
        "icon"    : "📉",
        "text_key": "scenario",
        "fallback": "document",
    },
}

# Edge drawn when cosine similarity exceeds this value
DEFAULT_EDGE_THRESHOLD = 0.20
# Max edges to draw (keeps the graph readable)
MAX_EDGES = 120
# Spring layout iterations
SPRING_ITERATIONS = 80


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    mm = st.session_state.memory_manager

    _render_header()

    # ── Controls ──────────────────────────────────────────────────────────────
    col_thresh, col_max, col_btn = st.columns([2, 2, 1])
    with col_thresh:
        threshold = st.slider(
            "🔗 Edge Similarity Threshold",
            min_value=0.05, max_value=0.70,
            value=DEFAULT_EDGE_THRESHOLD, step=0.05,
            help="Higher = only very similar nodes are connected",
            key="_bm_threshold",
        )
    with col_max:
        max_edges = st.slider(
            "📊 Max Edges",
            min_value=10, max_value=300,
            value=MAX_EDGES, step=10,
            key="_bm_max_edges",
        )
    with col_btn:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        refresh = st.button(
            "🔄 Refresh Map",
            key="_bm_refresh",
            use_container_width=True,
            type="primary",
        )

    # ── Extract nodes from all collections ────────────────────────────────────
    nodes = _extract_all_nodes(mm)

    if not nodes:
        st.info(
            "📭 No memories found. Use the other modes first — "
            "every debate, task, decision, and financial analysis you save "
            "will appear here as a node in the network."
        )
        return

    # ── Build graph ───────────────────────────────────────────────────────────
    positions  = _spring_layout(nodes)
    edges      = _compute_edges(nodes, threshold, max_edges)
    fig        = _build_figure(nodes, positions, edges)

    # ── Stats bar ─────────────────────────────────────────────────────────────
    _render_stats_bar(nodes, edges)

    st.divider()

    # ── Plotly chart ──────────────────────────────────────────────────────────
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar"   : True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "scrollZoom"       : True,
    })

    # ── Legend ────────────────────────────────────────────────────────────────
    _render_legend(nodes)

    # ── Node table (expandable) ───────────────────────────────────────────────
    with st.expander("📋 All nodes — raw data", expanded=False):
        _render_node_table(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_all_nodes(mm) -> list[dict]:
    """
    Fetches every document from all 4 ChromaDB collections.
    Returns a flat list of node dicts, each containing:
        id, mode, colour, label, text, document, metadata, timestamp
    """
    nodes: list[dict] = []

    for mode, meta in MODE_META.items():
        try:
            col = mm.get_collection(mode)
            count = col.count()
            if count == 0:
                continue

            results = col.get(
                include=["documents", "metadatas"],
                limit=200,      # cap per collection
            )

            ids       = results.get("ids",       [])
            docs      = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            for entry_id, doc, m in zip(ids, docs, metadatas):
                # Primary display text: prefer the metadata field, fall back to document
                text_key  = meta["text_key"]
                text      = (m or {}).get(text_key, "")
                if not text or len(text.strip()) < 3:
                    text = (doc or "")[:200]

                # Build a rich label for hover tooltip
                tooltip = _build_tooltip(mode, text, m or {})

                nodes.append({
                    "id"       : entry_id,
                    "mode"     : mode,
                    "colour"   : meta["colour"],
                    "icon"     : meta["icon"],
                    "mode_label": meta["label"],
                    "text"     : text.strip()[:120],
                    "document" : (doc or "")[:300],
                    "metadata" : m or {},
                    "tooltip"  : tooltip,
                    "timestamp": (m or {}).get("timestamp", "")[:10],
                })
        except Exception as e:
            st.warning(f"Could not load {mode} collection: {e}")

    return nodes


def _build_tooltip(mode: str, text: str, meta: dict) -> str:
    """Builds the hover tooltip string for a node."""
    icon  = MODE_META[mode]["icon"]
    label = MODE_META[mode]["label"]
    ts    = meta.get("timestamp", "")[:10]

    lines = [f"{icon} {label}", ""]

    if mode == "devils_advocate":
        lines.append(f"Claim: {text[:100]}")
        sev = meta.get("severity", "")
        if sev:
            lines.append(f"Severity: {sev}")

    elif mode == "execution":
        lines.append(f"Task: {text[:100]}")
        energy = meta.get("energy_level", "")
        cat    = meta.get("category", "")
        status = meta.get("completion_status", "")
        if energy: lines.append(f"Energy: {energy}")
        if cat:    lines.append(f"Category: {cat}")
        if status: lines.append(f"Status: {status}")

    elif mode == "ripple":
        lines.append(f"Decision: {text[:100]}")
        order  = meta.get("order", "")
        domain = meta.get("domain", "")
        blind  = meta.get("blindspot", "")
        if order:  lines.append(f"Order: {order}")
        if domain: lines.append(f"Domain: {domain}")
        if blind:  lines.append(f"Blindspot: {blind[:80]}")

    elif mode == "wealth":
        lines.append(f"Scenario: {text[:100]}")
        risk    = meta.get("risk_level", "")
        horizon = meta.get("time_horizon", "")
        asset   = meta.get("asset_class", "")
        if risk:    lines.append(f"Risk: {risk}")
        if horizon: lines.append(f"Horizon: {horizon}")
        if asset:   lines.append(f"Asset: {asset}")

    if ts:
        lines.append(f"\n{ts}")

    return "<br>".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Semantic edge computation  (TF-IDF cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_edges(
    nodes     : list[dict],
    threshold : float,
    max_edges : int,
) -> list[tuple[int, int, float]]:
    """
    Computes similarity between all node pairs using TF-IDF on the
    combined text+document string. Returns a sorted list of
    (idx_a, idx_b, similarity_score) for pairs above the threshold.
    """
    if len(nodes) < 2:
        return []

    # Build corpus from both the primary text and the raw document
    corpus = [
        f"{n['text']} {n['document']}"
        for n in nodes
    ]

    try:
        vectoriser = TfidfVectorizer(
            max_features   = 500,
            stop_words     = "english",
            min_df         = 1,
            ngram_range    = (1, 2),
        )
        tfidf_matrix = vectoriser.fit_transform(corpus)
        sim_matrix   = cosine_similarity(tfidf_matrix)
    except Exception:
        return []

    edges: list[tuple[int, int, float]] = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                edges.append((i, j, score))

    # Sort by similarity descending, cap at max_edges
    edges.sort(key=lambda e: e[2], reverse=True)
    return edges[:max_edges]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Force-directed spring layout  (pure NumPy, no networkx)
# ─────────────────────────────────────────────────────────────────────────────

def _spring_layout(
    nodes      : list[dict],
    iterations : int = SPRING_ITERATIONS,
    seed       : int = 42,
) -> np.ndarray:
    """
    Fruchterman-Reingold force-directed layout implemented in pure NumPy.
    Returns an (N, 2) array of (x, y) positions in the range [-1, 1].

    Nodes repel each other (Coulomb-like), while edges attract connected
    nodes (spring force). Without networkx this gives a clean, readable
    layout that clusters similar nodes naturally.
    """
    n = len(nodes)
    rng = np.random.default_rng(seed)

    # Initialise on a circle + small noise (better than pure random)
    angles   = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pos      = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(float)
    pos     += rng.uniform(-0.1, 0.1, size=(n, 2))

    # Pre-compute adjacency from same-mode membership for attraction bonus
    modes = [nd["mode"] for nd in nodes]

    area       = 1.0
    k          = math.sqrt(area / max(n, 1)) * 1.5   # ideal spring length
    t          = 0.15                                 # initial temperature

    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]     # (N, N, 2) pairwise diff
        dist  = np.linalg.norm(delta, axis=2)         # (N, N)
        np.fill_diagonal(dist, 1e-6)                  # avoid /0

        # Repulsion: all pairs push apart
        rep_force = (k ** 2) / dist                   # scalar (N, N)
        rep_vec   = delta / dist[:, :, None]           # unit vectors
        disp      = (rep_force[:, :, None] * rep_vec).sum(axis=1)  # (N, 2)

        # Attraction: same-mode nodes pull together gently
        for i in range(n):
            for j in range(i + 1, n):
                if modes[i] == modes[j]:
                    d   = dist[i, j]
                    att = (d ** 2) / k * 0.5
                    v   = (pos[j] - pos[i]) / d
                    disp[i] += att * v
                    disp[j] -= att * v

        # Cap displacement by temperature and apply
        disp_len = np.linalg.norm(disp, axis=1, keepdims=True).clip(1e-6)
        pos     += disp / disp_len * np.minimum(disp_len, t)
        t       *= 0.92     # cool down

    # Normalise to [-1, 1]
    pos -= pos.min(axis=0)
    rng_ = pos.max(axis=0)
    rng_[rng_ == 0] = 1.0
    pos  = pos / rng_ * 2.0 - 1.0

    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Plotly figure assembly
# ─────────────────────────────────────────────────────────────────────────────

def _build_figure(
    nodes    : list[dict],
    positions: np.ndarray,
    edges    : list[tuple[int, int, float]],
) -> go.Figure:
    """
    Builds the Plotly graph:
      - One invisible scatter trace per edge (thin lines)
      - One scatter trace per mode group (coloured dots)
    """
    fig = go.Figure()

    # ── Edge traces ───────────────────────────────────────────────────────────
    for (i, j, score) in edges:
        xi, yi = positions[i]
        xj, yj = positions[j]
        # Opacity scales with similarity strength
        alpha = 0.08 + min(score, 0.8) * 0.35

        fig.add_trace(go.Scatter(
            x          = [xi, xj, None],
            y          = [yi, yj, None],
            mode       = "lines",
            line       = dict(
                color  = f"rgba(120,120,140,{alpha:.2f})",
                width  = 0.8 + score * 1.5,
            ),
            hoverinfo  = "skip",
            showlegend = False,
        ))

    # ── Node traces — one per mode so legend works ────────────────────────────
    for mode, meta in MODE_META.items():
        idx_list = [i for i, n in enumerate(nodes) if n["mode"] == mode]
        if not idx_list:
            continue

        xs        = [positions[i][0] for i in idx_list]
        ys        = [positions[i][1] for i in idx_list]
        tooltips  = [nodes[i]["tooltip"]   for i in idx_list]
        labels    = [nodes[i]["text"][:50] for i in idx_list]
        timestamps= [nodes[i]["timestamp"] for i in idx_list]

        fig.add_trace(go.Scatter(
            x           = xs,
            y           = ys,
            mode        = "markers+text",
            name        = f"{meta['icon']} {meta['label']} ({len(idx_list)})",
            marker      = dict(
                color   = meta["colour"],
                size    = 14,
                line    = dict(color="#1a1a1a", width=1.5),
                opacity = 0.90,
            ),
            text        = ["" for _ in idx_list],   # no inline text — too cluttered
            hovertemplate=(
                "<b style='font-family:IBM Plex Mono'>%{customdata[0]}</b>"
                "<extra></extra>"
            ),
            customdata  = [[t] for t in tooltips],
            showlegend  = True,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        height        = 620,
        margin        = dict(l=20, r=20, t=20, b=20),
        font          = dict(family="IBM Plex Mono", color="#444", size=11),
        legend        = dict(
            font       = dict(color="#666", size=11),
            bgcolor    = "rgba(0,0,0,0)",
            bordercolor= "#1a1a1a",
            borderwidth= 1,
            x=0.01, y=0.99,
        ),
        xaxis = dict(
            visible    = False,
            showgrid   = False,
            zeroline   = False,
        ),
        yaxis = dict(
            visible    = False,
            showgrid   = False,
            zeroline   = False,
        ),
        hovermode  = "closest",
        hoverlabel = dict(
            bgcolor    = "#111",
            bordercolor= "#333",
            font       = dict(family="IBM Plex Mono", size=12, color="#ddd"),
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Header, stats bar, legend, table
# ─────────────────────────────────────────────────────────────────────────────

def _render_header() -> None:
    st.markdown(
        f"""<div style="border-left:3px solid {ACCENT};padding:.85rem 1.35rem;
                        margin-bottom:1.4rem;background:rgba(255,255,255,.015);
                        border-radius:0 6px 6px 0;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.45rem;
                        font-weight:700;color:{ACCENT};margin-bottom:.2rem;">
                🧠&nbsp; BRAIN MAP</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                        color:#555;letter-spacing:.16em;text-transform:uppercase;
                        margin-bottom:.7rem;">Semantic Network · All Memory Collections</div>
            <div style="font-size:.88rem;color:#999;line-height:1.72;max-width:720px;">
                Every entry you have saved across all 4 modes — visualised as a
                semantic network. Nodes that share similar themes or keywords
                are connected by edges. <b style="color:{ACCENT}">Hover any dot</b>
                to read what you wrote.
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_stats_bar(nodes: list[dict], edges: list[tuple]) -> None:
    counts = {m: sum(1 for n in nodes if n["mode"] == m) for m in MODE_META}
    cols   = st.columns(5)

    stat_data = [
        ("Total Nodes",  str(len(nodes)),   ACCENT),
        ("🔴 Debate",    str(counts["devils_advocate"]), "#e63946"),
        ("📅 Execution", str(counts["execution"]),       "#f4a261"),
        ("🔮 Ripple",    str(counts["ripple"]),          "#a8dadc"),
        ("📉 Wealth",    str(counts["wealth"]),          "#2a9d8f"),
    ]

    for col, (label, value, colour) in zip(cols, stat_data):
        with col:
            st.markdown(
                f"""<div style="text-align:center;border:1px solid #1a1a1a;
                                border-radius:4px;padding:.5rem .4rem;
                                background:rgba(255,255,255,.015);">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:.52rem;
                                color:#2e2e2e;text-transform:uppercase;
                                letter-spacing:.1em;">{label}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{colour};
                                line-height:1.4;">{value}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;"
        f"color:#252525;margin:.5rem 0;'>"
        f"🔗 {len(edges)} semantic edges drawn &nbsp;·&nbsp; "
        f"Scroll to zoom &nbsp;·&nbsp; Drag to pan &nbsp;·&nbsp; Hover nodes for detail"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_legend(nodes: list[dict]) -> None:
    st.divider()
    cols = st.columns(4)
    for col, (mode, meta) in zip(cols, MODE_META.items()):
        count = sum(1 for n in nodes if n["mode"] == mode)
        with col:
            st.markdown(
                f"""<div style="border-left:3px solid {meta['colour']};
                                padding:.4rem .7rem;">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:.62rem;color:{meta['colour']};
                                font-weight:700;">{meta['icon']} {meta['label']}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:.58rem;color:#333;">{count} nodes</div>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_node_table(nodes: list[dict]) -> None:
    for n in nodes:
        meta  = n["metadata"]
        ts    = n["timestamp"] or "—"
        colour= n["colour"]
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
            f"color:#444;padding:.3rem 0;border-bottom:1px solid #111;'>"
            f"<span style='color:{colour}'>{n['icon']} {n['mode_label']}</span>"
            f"&nbsp;·&nbsp;{n['text'][:90]}"
            f"<span style='color:#222;float:right'>{ts}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
