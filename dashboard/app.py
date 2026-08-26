"""CANE dashboard - overview and leaderboard.

Run from the repository root:

    .venv/Scripts/python.exe -m streamlit run dashboard/app.py

This page answers the assignment's second research question directly: among
LinUCB, DQN, Double DQN and PPO, which achieves the best engagement-fatigue
trade-off? It reads the result files each notebook wrote and renders whatever
exists, naming anything that does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.lib import charts, data, theme  # noqa: E402

st.set_page_config(page_title="CANE - Notification Pacing", page_icon="🔔",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  h1 {font-size: 1.85rem !important; letter-spacing: -0.02em;}
  .lede {color: #52514e; font-size: 1.02rem; line-height: 1.55; max-width: 70ch;}
  .kpi {border: 1px solid #e6e5e1; border-radius: 10px; padding: 0.85rem 1rem;
        background: #fff;}
  .kpi .k {font-size: 0.72rem; text-transform: uppercase; letter-spacing: .07em;
           color: #8a8983; font-weight: 600;}
  .kpi .v {font-size: 1.5rem; font-weight: 650; color: #0b0b0b; line-height: 1.25;}
  .kpi .s {font-size: 0.78rem; color: #52514e;}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi"><div class="k">{label}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True)


st.title("CANE - Context-Aware Notification Engine")
st.markdown(
    '<p class="lede">A reinforcement-learning agent decides, each hour, whether '
    'to stay silent or send one of two notification types to a simulated app '
    'user. It is scored on long-term engagement minus the fatigue every send '
    'creates. Four algorithms are compared on one shared environment against two '
    'non-learning baselines.</p>',
    unsafe_allow_html=True)

summary, missing = data.load_summary()

with st.sidebar:
    st.subheader("View")
    archetypes = ["All"] + [a for a in theme.ARCHETYPE_ORDER
                            if not summary.empty and a in set(summary["archetype"])]
    archetype = st.selectbox("User archetype", archetypes, index=0)
    st.caption(
        "The five archetypes have different receptive hours and are hidden from "
        "the agent, which must infer responsiveness from reward alone."
    )
    st.divider()
    st.subheader("Result files")
    for name, path in data.RESULT_FILES.items():
        ready = path.is_file()
        dot = theme.STATUS["good"] if ready else theme.TEXT_MUTED
        state = "ready" if ready else "not generated"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.5rem;'
            f'padding:.15rem 0;font-size:.86rem;">'
            f'<span style="width:8px;height:8px;border-radius:50%;'
            f'background:{dot};display:inline-block;flex:none;"></span>'
            f'<span style="font-weight:600;color:{theme.TEXT_PRIMARY};">{name}</span>'
            f'<span style="color:{theme.TEXT_MUTED};margin-left:auto;">{state}</span>'
            f'</div>', unsafe_allow_html=True)
    if missing:
        st.warning(f"{', '.join(missing)} still running. "
                   "The charts show every algorithm that has finished.")

if summary.empty:
    st.error(
        "No result files found. Run at least one notebook to generate results - "
        "see `RESUME.md` for the commands."
    )
    st.stop()

board = data.leaderboard(summary, archetype)

if not board.empty:
    best = board.iloc[0]
    learners = board[~board["agent"].isin(data.BASELINES)]
    best_learner = learners.iloc[0] if not learners.empty else None
    # Counted per (agent, archetype), not on the aggregate. An agent that is
    # silent on four archetypes and active on one still shows a non-zero mean
    # send rate overall, so aggregating first would hide exactly the collapse
    # this metric exists to surface.
    pairs = (summary.groupby(["agent", "archetype"])["sends_per_episode"]
             .mean().reset_index())
    learner_pairs = pairs[~pairs["agent"].isin(data.BASELINES)]
    silent = learner_pairs[learner_pairs["sends_per_episode"] < 0.05]

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Best overall", best["agent"], f"reward {best['reward']:.2f}")
    if best_learner is not None:
        kpi(c2, "Best learner", best_learner["agent"],
            f"reward {best_learner['reward']:.2f}")
    kpi(c3, "Never-send collapses", f"{len(silent)} of {len(learner_pairs)}",
        "learner x archetype pairs")
    n_seeds = int(summary[~summary["agent"].isin(data.BASELINES)]["seed"].nunique())
    kpi(c4, "Evaluation", "200 episodes",
        f"held out - {n_seeds} training seeds")

st.write("")
left, right = st.columns([1.05, 1])

with left:
    chart = charts.leaderboard_bars(
        board, f"Mean reward - {archetype if archetype != 'All' else 'all archetypes'}")
    if chart is not None:
        st.altair_chart(chart, width='stretch')
    st.caption(
        "Whiskers are 95% confidence intervals over training seeds. Reward 0.00 "
        "is not a missing value: it is exactly what an agent earns by never "
        "sending, since it then collects no clicks and pays no costs."
    )

with right:
    chart = charts.efficiency_frontier(board)
    if chart is not None:
        st.altair_chart(chart, width='stretch')
    st.caption(
        "Top-left is the goal: high conversion on few notifications. Bubble area "
        "is mean reward. Every point is labelled, so the comparison never rests "
        "on colour alone."
    )

st.divider()
st.subheader("Per-archetype breakdown")
st.markdown(
    '<p class="lede">Averaging across archetypes hides the result. A policy that '
    'looks degenerate on the mixed population is correct on some user types and '
    'clearly wrong on others.</p>', unsafe_allow_html=True)
chart = charts.reward_by_archetype(summary)
if chart is not None:
    # 'content': a faceted chart sizes each panel explicitly, so stretching it
    # would distort the panel widths rather than the surrounding whitespace.
    st.altair_chart(chart, width='content')

st.divider()
with st.expander("Results table (all agents, all archetypes)"):
    show = (summary.groupby(["archetype", "agent"])
            .agg(reward=("reward_mean", "mean"), ctr=("ctr", "mean"),
                 sends=("sends_per_episode", "mean"),
                 optout=("optout_rate", "mean"), seeds=("seed", "nunique"))
            .round(3).reset_index())
    st.dataframe(show, width='stretch', hide_index=True)
    st.caption(
        "A table view is provided because three palette hues fall below 3:1 "
        "contrast on a light surface; no reading depends on colour alone."
    )
