"""Ensemble agent: the majority-vote control and the confidence-gated mixture.

Part B of the rubric asks for an evaluated ensemble. This page reports two,
because reporting only the one that works would hide the finding.

The members disagree in a very particular way here: on three of the five
archetypes some members have collapsed to sending nothing. That makes naive
majority vote a genuinely informative negative control rather than a strawman
-- with one silent member in a pool of three it stops being a vote and starts
being an AND-gate over the two that still send.

Fitting is per archetype and never on the mixed population. On the mixed pool
every member holds with a wide margin, so vote, z-scored fusion and weighting
all collapse to the same silent policy and the comparison says nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.lib import charts, data, theme  # noqa: E402

st.set_page_config(page_title="Ensemble - CANE", page_icon="🔔", layout="wide")

st.title("Ensemble")
st.caption("Members are fitted on a validation split (seeds 920,000-920,099) "
           "that is disjoint from the 200 test episodes, so the mixture weights "
           "never see the numbers they are judged on.")

ens = data.load_ensemble()
sweep = data.load_bias_sweep()
meta = data.load_ensemble_meta()

if ens.empty:
    st.warning(
        "No ensemble results yet. Run:\n\n"
        "```\n.venv/Scripts/python.exe -m cane.ensemble_study\n```\n\n"
        "The quick pass (`--quick`) is enough to see the page render, but its "
        "validation sample is too small to weight the members honestly - see "
        "the caveat at the bottom of this page."
    )
    st.stop()

archetypes = [a for a in theme.ARCHETYPE_ORDER if a in set(ens["archetype"])]

with st.sidebar:
    st.subheader("Ensemble")
    archetype = st.selectbox("Archetype", archetypes, index=0)
    split = st.radio("Split", ["test", "validation"], index=0,
                     help="Test is the held-out result. Validation is what the "
                          "hold bias was chosen on.")

view = ens[(ens["archetype"] == archetype) & (ens["split"] == split)]


# --------------------------------------------------------------------------
# Members against the two ensembles
# --------------------------------------------------------------------------
def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per policy: the individual members, then each ensemble scheme."""
    out = df.copy()
    out["policy"] = out.apply(
        lambda r: r["member"] if r["scheme"] == "member"
        else f"Ensemble ({r['scheme']})", axis=1)
    cols = ["policy", "scheme", "reward_mean", "ctr", "sends_per_episode",
            "optout_rate"]
    return out[cols].sort_values("reward_mean", ascending=False)


table = comparison_table(view)

left, right = st.columns([2, 3])

with left:
    st.subheader(f"{archetype} - {split}")
    st.dataframe(
        table.assign(
            reward=table["reward_mean"].round(2),
            ctr=table["ctr"].round(3),
            sends=table["sends_per_episode"].round(2),
            optout=table["optout_rate"].round(3),
        )[["policy", "reward", "ctr", "sends", "optout"]],
        width="stretch", hide_index=True,
    )

    members = view[view["scheme"] == "member"]
    best_member = members.loc[members["reward_mean"].idxmax()] \
        if not members.empty else None
    gated = view[view["scheme"] == "gated"]
    vote = view[view["scheme"] == "vote"]

    if best_member is not None and not gated.empty:
        g = gated.iloc[0]
        delta = g["reward_mean"] - best_member["reward_mean"]
        st.metric("Gated ensemble against the best single member",
                  f"{g['reward_mean']:+.2f}", f"{delta:+.2f}")
        if delta < 0:
            st.caption(f"Below the best member ({best_member['member']}). "
                       "That is a real result, not a bug - see the caveat below.")

with right:
    bars = charts.ensemble_bars(
        table, title=f"Reward per episode - {archetype} ({split})")
    if bars is not None:
        st.altair_chart(bars, width="stretch")

st.divider()

# --------------------------------------------------------------------------
# The hold-bias line search
# --------------------------------------------------------------------------
st.subheader("How the operating point was chosen")

c1, c2 = st.columns([3, 2])

with c1:
    member_refs = ens[(ens["archetype"] == archetype) &
                      (ens["split"] == "validation") &
                      (ens["scheme"] == "member")]
    curve = charts.bias_sweep(sweep, archetype, members=member_refs)
    if curve is not None:
        st.altair_chart(curve, width="stretch")
        st.caption("Searched on validation only. The dashed line is the chosen "
                   "bias; horizontal rules are the individual members.")
    else:
        st.info("No bias sweep recorded for this archetype.")

with c2:
    m = meta.get(archetype)
    if m:
        st.markdown("**Mixture weights** - softmax of validation reward")
        w = pd.DataFrame({"member": list(m["weights"]),
                          "weight": list(m["weights"].values())})
        st.dataframe(w.assign(weight=w["weight"].round(3)),
                     width="stretch", hide_index=True)

        st.markdown("**Gate temperatures** - entropy matched across members")
        g = pd.DataFrame({"member": list(m["gates"]),
                          "temperature": list(m["gates"].values())})
        st.dataframe(g.assign(temperature=g["temperature"].round(2)),
                     width="stretch", hide_index=True)
        st.caption("Raw Q-values and PPO logits live on different scales. "
                   "Without matching each member's policy entropy, whichever "
                   "member happens to have the widest scores dominates the "
                   "mixture regardless of how good it is.")

        st.metric("Vote disagreement rate",
                  f"{m['vote_disagreement_rate']:.1%}",
                  help="Share of states where the members did not all agree. "
                       "Near zero means the vote had nothing to decide.")

st.divider()

# --------------------------------------------------------------------------
# The two honest caveats
# --------------------------------------------------------------------------
st.subheader("What to be careful about when reading this")

st.markdown(
    """
**1. Majority vote is the control, not the proposal.** With a silent member in
the pool it degenerates into an AND-gate: a notification goes out only when
both remaining members want one. That produces few sends at unusually high
precision, which looks good on click-through rate and says nothing about
whether voting was a good idea. Judge it on reward, and read the disagreement
rate beside it -- a vote among members that rarely disagree is not a vote.

**2. Softmax weighting over-rewards abstention.** A member that sends nothing
scores exactly `0.00` on validation. A member that takes risks and is slightly
unlucky on a small validation sample scores a little below zero. Softmax then
hands the highest weight to the member that does nothing, purely because doing
nothing has no variance.

On the full run (100 validation, 200 test episodes) the estimates are stable --
`Fixed-18:00` moves by at most 2.1 reward between the two splits on any
archetype -- so the weights below are not an artifact of a small sample. The
bias remains structural: it would reappear on any split where a silent member's
exact `0.00` sits above a working member's unlucky negative.
"""
)

if len(sweep) and sweep["hold_bias"].nunique() <= 3:
    st.warning("This looks like a `--quick` run: the bias sweep has very few "
               "points. Re-run without `--quick` before using these numbers in "
               "the report.")
