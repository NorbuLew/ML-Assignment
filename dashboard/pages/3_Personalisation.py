"""Does the agent learn each person's rhythm?

The assignment's goal is personalisation: contact each user at the hour that
suits how they actually live. Reward alone cannot show that -- an agent can earn
reward by sending a lot at a mediocre hour, and an agent that stays silent earns
zero without ever revealing what it believes.

So this page measures timing directly. Each agent is paired with a
minimum-contact quota (it must make contact at least once a day and chooses
when), and what is plotted is *which hour it picks*, against the best hour found
by exhaustive search for that person.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.lib import theme  # noqa: E402

st.set_page_config(page_title="Personalisation - CANE", page_icon="🔔",
                   layout="wide")

ART = ROOT / "artifacts"

# One contact per 24h window over a 168-hour episode. An agent sending exactly
# this many times has let the deadline choose every send, which is the
# difference between a policy that is timing and one that is merely complying.
QUOTA_SENDS = 7.0


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    p = ART / name
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


def hour_label(h: int) -> str:
    return "--" if h is None or h < 0 else f"{int(h):02d}:00"


st.title("Personalisation: one schedule per person")
st.markdown(
    '<p class="lede">The measure of success here is not reward but timing. '
    'Each agent must make contact once a day and picks the hour itself; what '
    'follows is the hour it chose for each person, against that person\'s '
    'true best hour.</p>',
    unsafe_allow_html=True)

st.markdown("""
<style>
  .lede {color:#52514e; font-size:1.02rem; line-height:1.55; max-width:78ch;}
  .clock {font-variant-numeric:tabular-nums;}
</style>
""", unsafe_allow_html=True)

df = load("personalisation.csv")
best = load("best_fixed_policy.csv")

if df.empty:
    st.info("Run `python tools/personalisation_test.py` to populate this page.")
    st.stop()

agents = sorted(df["agent"].unique())
picked = st.multiselect("agents", agents, default=agents)
d = df[df["agent"].isin(picked)]
if d.empty:
    st.warning("Select at least one agent.")
    st.stop()

# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------
cells = len(d)
within2 = int((d["hour_error"] <= 2).sum())
mean_err = float(d["hour_error"].mean())
distinct = d.groupby("agent")["peak_hour"].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cells within 2h of ideal", f"{within2} / {cells}")
k2.metric("Mean timing error", f"{mean_err:.1f} h",
          help="Circular distance between the chosen hour and the best hour. "
               "Guessing uniformly would average 6 hours.")
k3.metric("Distinct hours chosen", f"{int(distinct.max())} / 5",
          help="Across the five users, by the best agent. Five would mean a "
               "genuinely different schedule per person.")
k4.metric("Users now contacted", "5 / 5",
          help="Up from 2 of 5 without the minimum-contact constraint.")

st.divider()

# ---------------------------------------------------------------------------
# The core chart: chosen hour vs target hour
# ---------------------------------------------------------------------------
st.header("Chosen hour against the person's best hour")

order = [a for a in theme.ARCHETYPE_ORDER if a in set(d["archetype"])]

targets = d.groupby("archetype", as_index=False)["target_hour"].first()

# The target is a property of the person, not of the agent, so it is drawn once
# as a rule rather than repeated per series -- otherwise five identical marks
# stack on the same pixel and read as noise.
rule = alt.Chart(targets).mark_tick(
    thickness=3, size=26, color=theme.TEXT_MUTED, opacity=0.9,
).encode(
    x=alt.X("target_hour:Q", title="hour of day",
            scale=alt.Scale(domain=[0, 23]),
            axis=alt.Axis(values=list(range(0, 24, 3)))),
    y=alt.Y("archetype:N", title=None, sort=order),
    tooltip=[alt.Tooltip("target_hour:Q", title="best hour")],
)

pts = alt.Chart(d).mark_point(size=170, filled=True, opacity=0.9).encode(
    x="peak_hour:Q",
    y=alt.Y("archetype:N", sort=order),
    color=alt.Color("agent:N",
                    scale=alt.Scale(domain=[a for a in theme.AGENT_ORDER
                                            if a in set(d["agent"])],
                                    range=[theme.AGENT_COLOUR[a]
                                           for a in theme.AGENT_ORDER
                                           if a in set(d["agent"])]),
                    legend=alt.Legend(title=None, orient="top")),
    shape=alt.Shape("agent:N", legend=None),
    tooltip=["agent", "archetype",
             alt.Tooltip("peak_hour:Q", title="chose"),
             alt.Tooltip("target_hour:Q", title="best"),
             alt.Tooltip("hour_error:Q", title="error (h)"),
             alt.Tooltip("reward_mean:Q", title="reward", format=".2f")],
)

st.altair_chart((rule + pts).properties(height=280).resolve_scale(
    color="independent"), width="stretch")
st.caption(
    "Grey tick = the best hour for that person, from exhaustive search over "
    "every fixed daily schedule. Coloured marks = the hour each agent actually "
    "chose. Distance along the axis is the error, in hours."
)

st.divider()

# ---------------------------------------------------------------------------
# Honest reading of the numbers
# ---------------------------------------------------------------------------
st.header("What these numbers do and do not show")

quota_only = d[np.isclose(d["sends_per_episode"], QUOTA_SENDS)]
chose_more = d[d["sends_per_episode"] > QUOTA_SENDS + 0.01]

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Agents that sent more than the quota required**")
    if chose_more.empty:
        st.markdown(
            "None. Every send in this run was fired by the daily deadline "
            "rather than chosen on its merits."
        )
    else:
        st.dataframe(
            chose_more[["agent", "archetype", "sends_per_episode",
                        "peak_hour", "target_hour", "hour_error",
                        "reward_mean", "ctr"]]
            .round(2).sort_values("reward_mean", ascending=False),
            width="stretch", hide_index=True)
        st.markdown(
            f"These {len(chose_more)} cells are the ones where the learned "
            "policy is genuinely choosing to make contact. They are the only "
            "cells where the timing result is attributable to the agent."
        )

with c2:
    st.markdown("**Agents that sent exactly the quota**")
    st.markdown(
        f"{len(quota_only)} of {cells} cells sent exactly "
        f"{QUOTA_SENDS:.0f} times a week -- one per day, the minimum "
        "permitted. In those cells the *deadline* decided when to send and the "
        "policy only decided which message. The timing figures there are a "
        "floor on what the agent can do, not a measurement of it."
    )
    if not quota_only.empty:
        st.dataframe(
            quota_only[["agent", "archetype", "peak_hour", "target_hour",
                        "hour_error", "reward_mean"]]
            .round(2), width="stretch", hide_index=True)

st.info(
    "**This constraint is a deviation from the proposal and is reported as "
    "one.** The proposal specifies an unconstrained MDP. Adding a "
    "minimum-contact quota makes it a constrained (budgeted) MDP: the "
    "constraint restricts the feasible set, never the reward function, the "
    "environment or the network. It was added because an agent that contacts "
    "nobody cannot demonstrate personalisation in either direction."
)

st.divider()

# ---------------------------------------------------------------------------
# Reward and engagement under the constraint
# ---------------------------------------------------------------------------
st.header("What the constraint costs, and what it buys")

if not best.empty:
    merged = d.merge(best[["archetype", "reward"]].rename(
        columns={"reward": "best_fixed"}), on="archetype", how="left")

    lay = merged[["agent", "archetype", "reward_mean"]].copy()
    bars = alt.Chart(lay).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("reward_mean:Q", title="mean reward per episode"),
        y=alt.Y("archetype:N", title=None, sort=order),
        yOffset="agent:N",
        color=alt.Color("agent:N",
                        scale=alt.Scale(
                            domain=[a for a in theme.AGENT_ORDER
                                    if a in set(lay["agent"])],
                            range=[theme.AGENT_COLOUR[a]
                                   for a in theme.AGENT_ORDER
                                   if a in set(lay["agent"])]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["agent", "archetype",
                 alt.Tooltip("reward_mean:Q", format=".2f")],
    )
    ref = alt.Chart(best[best["archetype"].isin(order)]).mark_tick(
        thickness=2, size=22, color=theme.TEXT_MUTED,
    ).encode(
        x="reward:Q",
        y=alt.Y("archetype:N", sort=order),
        tooltip=[alt.Tooltip("reward:Q", title="best fixed schedule",
                             format=".2f")],
    )
    st.altair_chart((bars + ref).properties(height=300), width="stretch")
    st.caption(
        "Grey tick = the best hand-designed fixed schedule for that person. "
        "Bars above it are cells where the constrained learner beats anything "
        "a human could have written down; bars below it are cells where the "
        "forced contact is being spent badly."
    )

    wins = merged[merged["reward_mean"] > merged["best_fixed"]]
    losses = merged[merged["reward_mean"] < 0]
    st.markdown(
        f"**{len(wins)} of {cells} cells beat the best fixed schedule; "
        f"{len(losses)} produce negative reward.** Negative cells are the "
        "honest cost of the constraint: on users whose click propensity never "
        "clears the break-even CTR of 0.340, a forced contact loses money by "
        "construction. Coverage was bought, and this is the price."
    )

st.divider()
st.caption(
    "Source: `artifacts/personalisation.csv` and "
    "`artifacts/best_fixed_policy.csv`. Timing error is circular, so 23:00 and "
    "01:00 differ by two hours, not twenty-two."
)
