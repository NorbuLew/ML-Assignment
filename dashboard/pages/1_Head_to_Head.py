"""Head-to-head comparison of the four algorithms.

This page answers RQ2 -- which algorithm best manages the engagement/fatigue
trade-off -- and it answers it in the form the question is actually asked: not
"who has the largest mean" but "who beats whom, on the same episodes, by more
than the seed-to-seed noise".

The distinction matters here. Three of the four learners collapse to sending
nothing on some archetypes and score exactly 0.00 there, so a mean over all
five archetypes is a mean over a very lumpy distribution. Pairing the
comparison cell by cell is what keeps that from turning into a false ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.lib import charts, data, theme  # noqa: E402

st.set_page_config(page_title="Head to Head - CANE", page_icon="🔔",
                   layout="wide")

st.title("Head to head")
st.caption("Every agent below was evaluated on the same 200 held-out episodes "
           "(seeds 900,000-900,199). The comparison is paired over those cells.")

summary, missing = data.load_summary()

if summary.empty:
    st.warning("No result files yet. Run the notebooks first - see RESUME.md.")
    st.stop()

if missing:
    st.info(f"Still missing: {', '.join(missing)}. The page shows what exists; "
            "the comparison completes when every algorithm has written its CSV.")


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
all_agents = [a for a in theme.AGENT_ORDER if a in set(summary["agent"])]
learners = [a for a in all_agents if a not in data.BASELINES]

with st.sidebar:
    st.subheader("Compare")
    picked = st.multiselect("Agents", all_agents,
                            default=learners + ["Fixed-18:00"])
    arch_options = ["All"] + [a for a in theme.ARCHETYPE_ORDER
                              if a in set(summary["archetype"])]
    archetype = st.selectbox("Archetype", arch_options, index=0)

if len(picked) < 2:
    st.warning("Pick at least two agents to compare.")
    st.stop()

view = summary[summary["agent"].isin(picked)]
if archetype != "All":
    view = view[view["archetype"] == archetype]

board = data.leaderboard(view, archetype)


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------
def paired_matrix(df: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    """Mean reward difference and paired p-value for every ordered agent pair.

    Pairing is on (archetype, seed): the same user population and the same
    random draw for both agents. Cells present for only one of the two are
    dropped rather than filled, so a missing run can never read as a win.

    The deterministic baselines carry a single seed, because re-running a fixed
    schedule on fixed evaluation episodes reproduces itself exactly. Pairing
    those on (archetype, seed) would silently discard four fifths of the
    *learner's* data -- keeping only its seed-0 draw, which is an arbitrary one.
    So when either side has one seed, both collapse to per-archetype means
    first and the pairing is on archetype. Same test, but the learner is
    represented by its average rather than by one lucky or unlucky run.

    The test is a paired t-test on the per-cell differences. With five seeds it
    is underpowered by design, which is why the cell count travels with the
    p-value -- a bare star would overstate what 25 paired cells can support.
    """
    fine = (df.pivot_table(index=["archetype", "seed"], columns="agent",
                           values="reward_mean", aggfunc="mean"))
    coarse = (df.pivot_table(index="archetype", columns="agent",
                             values="reward_mean", aggfunc="mean"))
    seeds = df.groupby("agent")["seed"].nunique()

    rows = []
    for a in agents:
        for b in agents:
            if a == b or a not in fine.columns or b not in fine.columns:
                continue
            single = min(seeds.get(a, 0), seeds.get(b, 0)) < 2
            wide = coarse if single else fine
            pair = wide[[a, b]].dropna()
            if pair.empty:
                continue
            diff = (pair[a] - pair[b]).to_numpy(dtype=float)
            n = int(diff.size)
            # A paired t-test needs at least two cells and some spread; a
            # constant difference (both agents silent everywhere) has no
            # sampling distribution, so report the delta with p undefined.
            if n >= 2 and np.ptp(diff) > 0:
                p = float(stats.ttest_rel(pair[a], pair[b]).pvalue)
            else:
                p = float("nan")
            delta = float(diff.mean())
            star = "*" if (p == p and p < 0.05) else ""
            rows.append({"row": a, "col": b, "delta": delta, "p": p, "n": n,
                         "label": f"{delta:+.1f}{star}"})
    return pd.DataFrame(rows)


matrix = paired_matrix(view, picked)

left, right = st.columns([3, 2])

with left:
    chart = charts.head_to_head_matrix(matrix)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    st.caption("Read a row: positive (orange) means the row agent scored higher "
               "than the column agent on the shared cells. `*` marks p < 0.05 "
               "on a paired t-test.")

with right:
    st.subheader("RQ2 - who wins?")
    if board.empty:
        st.write("No rows for this selection.")
    else:
        top = board.iloc[0]
        st.metric(f"Highest mean reward{'' if archetype == 'All' else f' on {archetype}'}",
                  top["agent"], f"{top['reward']:+.2f} per episode")

        # The honest headline is not the top mean but whether that top mean is
        # separable from the runner-up on paired cells.
        if len(board) > 1:
            second = board.iloc[1]
            cell = matrix[(matrix["row"] == top["agent"]) &
                          (matrix["col"] == second["agent"])]
            if not cell.empty:
                c = cell.iloc[0]
                p = c["p"]
                if p != p:
                    verdict = "no variation across cells - not testable"
                elif p < 0.05:
                    verdict = f"separable from {second['agent']} (p = {p:.4f})"
                else:
                    verdict = (f"**not** separable from {second['agent']} "
                               f"(p = {p:.3f}) on {int(c['n'])} paired cells")
                st.write(f"Margin over the runner-up: **{c['delta']:+.2f}** - "
                         f"{verdict}.")

        st.dataframe(
            board.assign(
                reward=board["reward"].round(2),
                ctr=board["ctr"].round(3),
                sends=board["sends"].round(1),
                optout=board["optout"].round(3),
            )[["agent", "reward", "ctr", "sends", "optout"]],
            width="stretch", hide_index=True,
        )

st.divider()

# --------------------------------------------------------------------------
# The trade-off itself
# --------------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    frontier = charts.efficiency_frontier(board)
    if frontier is not None:
        st.altair_chart(frontier, width="stretch")
    st.caption("Top-left is the objective: high click-through on few sends. "
               "Bubble size is mean reward.")

with c2:
    st.subheader("Where the averages come from")
    st.write(
        "An agent sitting at the origin sends nothing and converts nothing. "
        "That scores exactly 0.00, which beats every negative score but is a "
        "local optimum, not the best available policy - a send pays whenever "
        "the click probability clears the break-even rate of 0.340, which "
        "every archetype does for 3-8 hours a day. The **Diagnosis** page "
        "carries that arithmetic and what it cost."
    )
    silent = board[(board["sends"] < 0.05) & (~board["agent"].isin(data.BASELINES))]
    if not silent.empty:
        st.warning("Sending nothing on this selection: "
                   + ", ".join(silent["agent"]))

st.divider()

by_arch = charts.reward_by_archetype(view)
if by_arch is not None and archetype == "All":
    st.altair_chart(by_arch, width="content")
    st.caption("The same agents, per archetype. A flat 0.00 bar is a silent "
               "policy, not a failed run.")
