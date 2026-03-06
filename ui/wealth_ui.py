"""
pages/wealth_ui.py
==================
Wealth Logic — financial reasoning and opportunity cost auditor.

Layout
------
  Scenario input       : Text area — describe the financial decision/scenario
  Context controls     : Amount field, risk level selector, time horizon, asset class
  Opportunity cost calc: Side-by-side "Path A vs Path B" projected value display
  [Audit Reasoning]    : Sends full context to LLM, streams back analysis
  Past sessions panel  : Recent financial entries from memory

LLM  : stream() — Socratic financial reasoning audit
Save : mm.save_financial_entry() once per audit
"""
from __future__ import annotations

import streamlit as st

from core.llm_config import WEALTH_PROMPT, stream as llm_stream
from core.memory_manager import format_financial_context

# ─────────────────────────────────────────────────────────────────────────────
ACCENT   = "#2a9d8f"
MODE_KEY = "wealth"
ICON     = "📉"
TITLE    = "Wealth Logic"
TAGLINE  = "Financial reasoning and opportunity cost auditor"
DESC     = (
    "Describe a financial decision or scenario. The engine audits your reasoning "
    "across opportunity cost, risk-adjusted return, time horizon alignment, "
    "and hidden assumptions — without giving specific investment advice."
)

_RISK_LEVELS  = ["Low", "Medium", "High", "Critical"]
_HORIZONS     = ["Immediate (< 1 month)", "Short (1–12 months)",
                 "Medium (1–5 years)", "Long (5+ years)"]
_ASSET_CLASSES = ["Cash / Savings", "Equity / Stocks", "Real Estate",
                  "Crypto", "Business / Startup", "Education / Skills", "Other"]

_RISK_COLOURS = {
    "Low": "#2a9d8f", "Medium": "#f4a261",
    "High": "#e76f51", "Critical": "#e63946"
}
# ─────────────────────────────────────────────────────────────────────────────


def render() -> None:
    mm = st.session_state.memory_manager
    _header()
    _memory_bar(mm)
    st.divider()
    _controls_and_output(mm)


# ── Header ────────────────────────────────────────────────────────────────────

def _header() -> None:
    st.markdown(f"""
        <div style="border-left:3px solid {ACCENT};padding:.85rem 1.35rem;
                    margin-bottom:1.4rem;background:rgba(255,255,255,.015);
                    border-radius:0 6px 6px 0;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.45rem;
                        font-weight:700;color:{ACCENT};margin-bottom:.2rem;">
                {ICON}&nbsp; {TITLE.upper()}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                        color:#555;letter-spacing:.16em;text-transform:uppercase;
                        margin-bottom:.7rem;">{TAGLINE}</div>
            <div style="font-size:.88rem;color:#999;line-height:1.72;max-width:680px;">
                {DESC}</div>
        </div>""", unsafe_allow_html=True)


# ── Memory bar ────────────────────────────────────────────────────────────────

def _memory_bar(mm) -> None:
    count = mm.get_stats().get(MODE_KEY, 0)
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
            f"color:#444;text-transform:uppercase;padding-top:.4rem;'>"
            f"🗄️ Memory &nbsp;·&nbsp; <span style='color:{ACCENT}'>{count}</span>"
            f" financial sessions</div>", unsafe_allow_html=True)
    with c2:
        if st.button("🗑️ Reset", key=f"rst_{MODE_KEY}", use_container_width=True,
                     help="Wipes ONLY wealth memory"):
            st.session_state[f"_cfm_{MODE_KEY}"] = True

    if st.session_state.get(f"_cfm_{MODE_KEY}", False):
        st.warning(f"Delete all {count} sessions? Other modes are unaffected.")
        a, b, _ = st.columns([1, 1, 4])
        with a:
            if st.button("✅ Confirm", key="_do_rst_wealth", type="primary"):
                mm.clear_collection(MODE_KEY)
                st.session_state.update({
                    "_wealth_result": None, f"_cfm_{MODE_KEY}": False
                })
                st.rerun()
        with b:
            if st.button("❌ Cancel", key="_no_rst_wealth"):
                st.session_state[f"_cfm_{MODE_KEY}"] = False
                st.rerun()


# ── Controls + output ─────────────────────────────────────────────────────────

def _controls_and_output(mm) -> None:

    # ── Scenario input ────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;"
        "color:#444;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem;'>"
        "💬 Scenario</div>", unsafe_allow_html=True)

    scenario = st.text_area(
        label            = "##scenario",
        placeholder      = (
            "Describe your financial decision or scenario in plain language.\n\n"
            "e.g. 'I have ₹5L saved and I'm deciding between a fixed deposit or index funds'\n"
            "e.g. 'My company is offering ESOPs. Should I exercise them now?'\n"
            "e.g. 'I want to quit my job and freelance — I currently spend ₹80K/month'"
        ),
        height           = 110,
        key              = "_wealth_scenario",
        label_visibility = "collapsed",
    )

    # ── Financial context controls ────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        amount = st.number_input(
            "💰 Amount Involved (optional)",
            min_value   = 0,
            max_value   = 100_000_000,
            value       = 0,
            step        = 10_000,
            format      = "%d",
            key         = "_wealth_amount",
            help        = "Helps calibrate the opportunity cost calculator.",
        )
        currency = st.selectbox(
            "Currency",
            options=["₹ INR", "$ USD", "€ EUR", "£ GBP"],
            key="_wealth_currency",
            label_visibility="visible",
        )

    with col_b:
        risk_level = st.selectbox(
            "📊 Risk Level",
            options=_RISK_LEVELS,
            index=1,
            key="_wealth_risk",
            help="Your perceived risk of this decision",
        )
        risk_color = _RISK_COLOURS.get(risk_level, "#888")
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;"
            f"color:{risk_color};margin-top:.2rem;'>● {risk_level} risk</div>",
            unsafe_allow_html=True,
        )

        asset_class = st.selectbox(
            "🏷️ Asset Class",
            options=_ASSET_CLASSES,
            key="_wealth_asset",
        )

    with col_c:
        time_horizon = st.selectbox(
            "⏱️ Time Horizon",
            options=_HORIZONS,
            index=2,
            key="_wealth_horizon",
        )
        st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

        # Stated expected return — used in opportunity cost calc
        expected_return_pct = st.number_input(
            "📈 Expected Return %/yr (optional)",
            min_value  = -100.0,
            max_value  = 1000.0,
            value      = 0.0,
            step       = 0.5,
            format     = "%.1f",
            key        = "_wealth_expected_return",
            help       = "Used in the opportunity cost calculator below.",
        )

    # ── Opportunity Cost Calculator ───────────────────────────────────────────
    if amount > 0:
        _opportunity_cost_calculator(amount, currency, expected_return_pct, time_horizon)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Audit button ──────────────────────────────────────────────────────────
    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        run = st.button(
            "🔍 Audit Reasoning",
            key  = "_wealth_run",
            type = "primary",
            use_container_width=True,
        )
    with col_hint:
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:.6rem;"
            "color:#2e2e2e;padding-top:.5rem;'>Audits opportunity cost, "
            "risk-return, time horizon, and hidden assumptions.</div>",
            unsafe_allow_html=True,
        )

    if run:
        if not scenario.strip():
            st.warning("Describe a scenario before auditing.")
            return
        _run_audit(scenario.strip(), amount, currency, risk_level,
                   time_horizon, asset_class, mm)

    elif "_wealth_result" in st.session_state and st.session_state["_wealth_result"]:
        st.divider()
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
            "color:#2e2e2e;text-transform:uppercase;margin-bottom:.5rem;'>"
            "Last Audit</div>", unsafe_allow_html=True)
        st.markdown(st.session_state["_wealth_result"])

    # Past sessions
    _past_sessions(mm)


# ── Opportunity cost calculator ───────────────────────────────────────────────

def _opportunity_cost_calculator(
    amount: int, currency: str, expected_return: float, horizon: str
) -> None:
    """
    Side-by-side comparison: projected value of PATH A (stated decision)
    vs PATH B (benchmark: 12% index fund equivalent).
    Values are illustrative, not financial advice.
    """
    # Determine approximate years from horizon string
    years_map = {
        "Immediate": 0.08,   # ~1 month
        "Short":     0.75,
        "Medium":    3.0,
        "Long":      7.0,
    }
    horizon_years = next(
        (v for k, v in years_map.items() if k.lower() in horizon.lower()), 3.0
    )

    benchmark_rate = 12.0   # illustrative annual index fund return
    path_a_return  = expected_return if expected_return != 0 else 0.0
    path_b_return  = benchmark_rate

    path_a_value = amount * ((1 + path_a_return / 100) ** horizon_years)
    path_b_value = amount * ((1 + path_b_return / 100) ** horizon_years)
    opp_cost     = path_b_value - path_a_value

    curr_symbol = currency.split(" ")[0]

    col_a, col_mid, col_b = st.columns([2, 1, 2])

    with col_a:
        st.markdown(
            f"""
            <div style="border:1px solid #1e1e1e;border-radius:4px;padding:.75rem 1rem;
                        background:rgba(42,157,143,.05);">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.58rem;
                            color:#2e2e2e;text-transform:uppercase;letter-spacing:.1em;
                            margin-bottom:.4rem;">Path A · This Decision</div>
                <div style="font-size:1.1rem;font-weight:700;color:{ACCENT};">
                    {curr_symbol} {path_a_value:,.0f}
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;
                            color:#333;margin-top:.2rem;">
                    @ {path_a_return:+.1f}%/yr over {horizon_years:.1f}yr
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_mid:
        st.markdown(
            "<div style='text-align:center;padding-top:1.2rem;font-size:1rem;"
            "color:#2a2a2a;'>vs</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown(
            f"""
            <div style="border:1px solid #1e1e1e;border-radius:4px;padding:.75rem 1rem;
                        background:rgba(255,255,255,.02);">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.58rem;
                            color:#2e2e2e;text-transform:uppercase;letter-spacing:.1em;
                            margin-bottom:.4rem;">Path B · Index Benchmark</div>
                <div style="font-size:1.1rem;font-weight:700;color:#888;">
                    {curr_symbol} {path_b_value:,.0f}
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;
                            color:#333;margin-top:.2rem;">
                    @ {path_b_return:+.1f}%/yr over {horizon_years:.1f}yr (illustrative)
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Opportunity cost line
    opp_color  = "#e63946" if opp_cost > 0 else "#2a9d8f"
    opp_prefix = "−" if opp_cost > 0 else "+"
    st.markdown(
        f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                    color:{opp_color};margin:.55rem 0 .25rem;
                    border-left:2px solid {opp_color};padding-left:.6rem;">
            Opportunity cost of Path A vs Path B:
            <b>{opp_prefix}{curr_symbol} {abs(opp_cost):,.0f}</b>
            over {horizon_years:.1f} yr
            &nbsp;·&nbsp;
            <span style="color:#333;font-size:.58rem;">
            Illustrative only — not financial advice.</span>
        </div>
        """, unsafe_allow_html=True)


# ── Audit runner ──────────────────────────────────────────────────────────────

def _run_audit(
    scenario: str, amount: int, currency: str,
    risk_level: str, time_horizon: str, asset_class: str, mm
) -> None:
    past         = mm.query_financial_history(scenario, top_k=3)
    memory_block = format_financial_context(past)

    curr_symbol = currency.split(" ")[0]
    context = (
        f"SCENARIO: {scenario}\n"
        f"AMOUNT   : {curr_symbol} {amount:,} ({currency})\n"
        f"RISK     : {risk_level}\n"
        f"HORIZON  : {time_horizon}\n"
        f"ASSET    : {asset_class}"
    )
    if memory_block:
        context += f"\n\nRELEVANT PAST REASONING FROM MEMORY:\n{memory_block}"

    st.divider()
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.65rem;"
        f"color:{ACCENT};text-transform:uppercase;letter-spacing:.1em;"
        f"margin-bottom:.6rem;'>🔍 Reasoning Audit</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Auditing reasoning..."):
        analysis = st.write_stream(
            llm_stream(system_prompt=WEALTH_PROMPT, user_input=context)
        )

    st.session_state["_wealth_result"] = analysis

    # Save to memory
    mm.save_financial_entry(
        scenario         = scenario,
        reasoning        = analysis,
        risk_level       = risk_level.lower(),
        time_horizon     = _normalise_horizon(time_horizon),
        asset_class      = _normalise_asset(asset_class),
    )


def _past_sessions(mm) -> None:
    past = mm.query_financial_history("financial decision", top_k=4)
    if not past:
        return

    st.divider()
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
        "color:#2e2e2e;text-transform:uppercase;letter-spacing:.1em;"
        "margin-bottom:.6rem;'>📚 Past Sessions</div>", unsafe_allow_html=True)

    for r in past:
        m = r["metadata"]
        risk_col = _RISK_COLOURS.get(m.get("risk_level","").capitalize(), "#444")
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.62rem;"
            f"color:#333;padding:.3rem 0;border-bottom:1px solid #141414;'>"
            f"<span style='color:{risk_col}'>{m.get('risk_level','?').upper()}</span>"
            f"&nbsp;·&nbsp;{m.get('scenario','')[:80]}"
            f"&nbsp;<span style='color:#222;float:right'>{m.get('timestamp','')[:10]}</span>"
            f"</div>", unsafe_allow_html=True)


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalise_horizon(h: str) -> str:
    h = h.lower()
    if "immediate" in h: return "immediate"
    if "short"     in h: return "short"
    if "medium"    in h: return "medium"
    return "long"

def _normalise_asset(a: str) -> str:
    a = a.lower()
    if "cash"     in a: return "cash"
    if "equity"   in a or "stock" in a: return "equity"
    if "real"     in a: return "real_estate"
    if "crypto"   in a: return "crypto"
    if "business" in a or "startup" in a: return "business"
    if "education"in a or "skill"  in a: return "education"
    return "other"
