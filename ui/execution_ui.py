"""
pages/execution_ui.py
Execution Architect — energy-aware task ranking.
render() is called directly by app.py router.
"""
import streamlit as st
from core.llm_config import EXECUTION_PROMPT, stream as llm_stream


def render() -> None:
    mm  = st.session_state.memory_manager
    llm = st.session_state.llm                  # available if needed directly

    # ── Title ─────────────────────────────────────────────────────────────────
    st.title("📅 Execution Architect")
    st.caption("Energy-aware task ranking and daily execution planning.")

    st.divider()

    # ── Energy slider ─────────────────────────────────────────────────────────
    energy = st.select_slider(
        "⚡ Energy Level",
        options = list(range(1, 11)),
        value   = 7,
        help    = "1–3 Low  ·  4–6 Medium  ·  7–10 High",
    )

    # Translate numeric value to band label and live description
    if energy <= 3:
        band_label   = "low"
        band_display = "🔴 Low — stick to admin, review, and passive tasks"
        band_colour  = "#e63946"
    elif energy <= 6:
        band_label   = "medium"
        band_display = "🟡 Medium — meetings, emails, structured thinking"
        band_colour  = "#f4a261"
    else:
        band_label   = "high"
        band_display = "🟢 High — deep work, creative tasks, hard decisions"
        band_colour  = "#2a9d8f"

    st.markdown(
        f"<p style='color:{band_colour};font-size:.9rem;margin-top:-.4rem;'>"
        f"{band_display}</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Task list ─────────────────────────────────────────────────────────────
    task_input = st.text_area(
        "📋 Task List",
        placeholder=(
            "One task per line, e.g.\n"
            "Write Q3 strategy doc\n"
            "Reply to client emails\n"
            "Fix the auth bug\n"
            "Review design mockups"
        ),
        height = 160,
    )

    # ── Analyse button ────────────────────────────────────────────────────────
    if st.button("▶ Analyse Tasks", type="primary"):

        tasks = [
            t.strip().lstrip("-•*").strip()
            for t in task_input.splitlines()
            if t.strip()
        ]

        if not tasks:
            st.warning("Add at least one task before analysing.")
            return

        # Build context block for the LLM
        context = (
            f"ENERGY LEVEL: {energy}/10 ({band_label.upper()})\n\n"
            f"TASKS:\n" + "\n".join(f"- {t}" for t in tasks) + "\n\n"
            f"INSTRUCTION: Rank these tasks in order of priority given the "
            f"user's energy level. Separate them into:\n"
            f"  HIGH FOCUS (Deep Work) — do first while energy holds\n"
            f"  LOW FOCUS  (Admin)     — do when energy dips\n"
            f"Give a one-line rationale for each task's placement."
        )

        st.divider()
        st.subheader("📊 Your Execution Plan")

        # Stream the response
        with st.spinner("Building plan..."):
            reply = st.write_stream(
                llm_stream(system_prompt=EXECUTION_PROMPT, user_input=context)
            )

        # Save every task to memory with energy metadata
        for task in tasks:
            mm.save_task(
                task         = task,
                energy_level = band_label,      # "high" | "medium" | "low"
                category     = "other",         # plan categorises further
            )

        st.success(f"✅ {len(tasks)} task(s) saved to execution memory.")

    st.divider()

    # ── Past task count ───────────────────────────────────────────────────────
    count = mm.get_stats().get("execution", 0)
    st.caption(f"🗄️ {count} task sessions in execution memory.")

    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("🗑️ Clear Memory", help="Wipes ONLY execution memory"):
            mm.clear_collection("execution")
            st.rerun()
