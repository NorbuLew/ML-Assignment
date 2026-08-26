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

import altair as alt
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

st.divider()

# ---------------------------------------------------------------------------
# Convergence speed
# ---------------------------------------------------------------------------
st.header("Sample efficiency: which algorithm gets there first?")

st.markdown(
    "Final reward says which algorithm ends up best. It says nothing about "
    "which gets there first, and for a system that has to be trained against "
    "real users those are different questions -- an agent needing three times "
    "the episodes spends three times as long sending badly timed "
    "notifications to real people."
)


@st.cache_data(show_spinner=False)
def _curves(suffix: str):
    root = Path(__file__).resolve().parents[2] / "artifacts"
    c = root / f"learning_curves{suffix}.csv"
    v = root / f"convergence{suffix}.csv"
    return (pd.read_csv(c) if c.is_file() else pd.DataFrame(),
            pd.read_csv(v) if v.is_file() else pd.DataFrame())


# Two runs exist and they answer different questions. Defaults show what the
# shipped hyperparameters do, which is mostly nothing; tuned shows what the
# algorithms are capable of once the collapse is fixed. Showing only one of
# them would misrepresent whichever question the reader had in mind.
available = [(label, sfx) for label, sfx in
             [("Tuned configuration", "_tuned"), ("Shipped defaults", "")]
             if not _curves(sfx)[0].empty]

if not available:
    st.info("Run `python tools/learning_curves.py` to populate this section.")
else:
    labels = [lab for lab, _ in available]
    pick = st.radio("Configuration", labels, index=0, horizontal=True)
    suffix = dict(available)[pick]
    curves, conv = _curves(suffix)

    arch_opts = ["All"] + [a for a in theme.ARCHETYPE_ORDER
                           if a in set(curves["archetype"])]
    which = st.selectbox("Archetype", arch_opts, index=0, key="conv_arch")
    cur = curves if which == "All" else curves[curves["archetype"] == which]

    agents_here = [a for a in theme.AGENT_ORDER
                   if a in {x.upper() for x in cur["agent"]}]
    scale = alt.Scale(domain=agents_here,
                      range=[theme.AGENT_COLOUR[a] for a in agents_here])

    line = alt.Chart(cur).mark_line(strokeWidth=2).encode(
        x=alt.X("episode:Q", title="training episodes"),
        y=alt.Y("mean(reward):Q", title="greedy reward on held-out episodes"),
        color=alt.Color("agent:N", scale=scale,
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["agent", "episode", alt.Tooltip("mean(reward):Q",
                                                 format=".2f")],
    ).properties(height=320)
    st.altair_chart(line, width="stretch")
    st.caption(
        "Each point is the frozen policy evaluated greedily, so these curves "
        "track what was *learned* rather than the noisy exploring policy that "
        "generated it. Averaged over archetypes when 'All' is selected."
    )

    if not conv.empty:
        eff = (conv.dropna(subset=["converged_at"])
                   .groupby("agent")["converged_at"]
                   .agg(["mean", "count"])
                   .rename(columns={"mean": "episodes to converge",
                                    "count": "archetypes that converged"}))
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(eff.round(0), width="stretch")
        with c2:
            st.markdown(
                "Converged means the first checkpoint reaching 90% of that "
                "agent's **own** final reward and never dropping back. "
                "Measuring against each agent's own ceiling rather than a "
                "shared reward level is deliberate: a shared threshold "
                "flatters whichever algorithm scores highest and says nothing "
                "about speed.\n\n"
                "An agent missing from this table never rose above zero on any "
                "archetype, so it has no ceiling to converge to. Reporting a "
                "number for it would invent one."
            )

        st.warning(
            "**Read these figures as a lower bound, for two reasons.** The "
            "reward curves are noisy on a plateau, and 'never drops back' is "
            "only satisfied once the noise ends -- which pushes the reported "
            "episode toward the end of the budget even for an agent that "
            "effectively converged much earlier. And Double DQN was still "
            "improving at the final checkpoint (about +0.8 reward per 50 "
            "episodes over the last six), so 1500 episodes was not enough to "
            "find its ceiling. DQN had flattened. The honest conclusion is "
            "that both are slow to converge on this task and that the budget "
            "used everywhere else in this project, 600 episodes, is well short "
            "of convergence."
        )
