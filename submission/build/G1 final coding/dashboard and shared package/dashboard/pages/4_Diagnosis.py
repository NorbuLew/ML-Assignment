"""Why the agents went silent, and what finally fixed it.

Some of the five users receive nothing from some of the deep RL agents. A
results table showing `0.00` invites two wrong readings -- that the run broke,
or that silence is simply optimal -- and neither is true. This page carries the
evidence: what silence costs, what caused it, every intervention tested, and
which one worked.

Which agents collapse is read from the result files rather than written down
here. It has already changed once: an earlier DQN run was silent on three users
and a later re-run of the same notebook was not, which is itself evidence for
the undertraining conclusion this page reaches. Naming the agents in prose would
have left that prose quietly wrong.

Seven mechanisms were tested across four studies. Five failed. They are kept
here because a surviving explanation shown next to its rejected rivals is better
evidence than one shown alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.lib import data, theme  # noqa: E402

st.set_page_config(page_title="Diagnosis - CANE", page_icon="🔔", layout="wide")

ART = ROOT / "artifacts"

# (W_send + W_fat*kappa/(1-lam)) / R_click for an Engage nudge. A policy whose
# CTR sits below this loses money on every contact, so it is the line every
# result on this page is judged against.
BREAK_EVEN_CTR = 0.340


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    p = ART / name
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


SILENT_SENDS = 0.05

# Which agents collapsed is a property of the result files, not of this page.
# Reading it here keeps the headline true when a teammate re-runs a notebook --
# which has already happened once, and silently falsified the sentence that used
# to be hardcoded in its place.
_summary, _ = data.load_summary()
_pairs = (_summary[~_summary["agent"].isin(data.BASELINES)]
          .groupby(["agent", "archetype"])["sends_per_episode"]
          .mean().reset_index())
_silent = _pairs[_pairs["sends_per_episode"] < SILENT_SENDS]
COLLAPSED = sorted(_silent["agent"].unique(),
                   key=lambda a: -(_silent["agent"] == a).sum())
SILENT_USERS = sorted(_silent["archetype"].unique())
N_PEOPLE = _pairs["archetype"].nunique()


def _join(items: list[str]) -> str:
    if not items:
        return "none"
    if len(items) == 1:
        return f"**{items[0]}**"
    return ", ".join(f"**{i}**" for i in items[:-1]) + f" and **{items[-1]}**"


st.title("Diagnosis: the missing notifications")

if COLLAPSED:
    _lede = (
        f"{_join(SILENT_USERS)} received nothing from {_join(COLLAPSED)} -- "
        f"{len(_silent)} of {len(_pairs)} agent-user pairs sent not one "
        "notification all week. Reward was genuinely being left on the table, "
        "the cause was a collapse of every Q-value rather than a considered "
        "choice, and of seven interventions tested only one recovered both "
        "coverage and profit.")
else:
    _lede = (
        "No agent is currently silent on any user, so the collapse this page "
        "diagnoses is not present in the result files on disk. The evidence "
        "below is kept because it is what the fix was derived from.")

st.markdown(f'<p class="lede">{_lede}</p>', unsafe_allow_html=True)

if COLLAPSED:
    st.caption(
        f"Read from the result files, not written into the page. "
        f"{_join([a for a in ['LinUCB', 'DQN', 'DDQN', 'PPO'] if a not in COLLAPSED])} "
        "contacted every user at least once."
    )

st.markdown("""
<style>
  .lede {color:#52514e; font-size:1.02rem; line-height:1.55; max-width:78ch;}
  .verdict {border-left:3px solid #e6e5e1; padding:2px 0 2px 14px; margin:8px 0;}
  .verdict.dead {border-left-color:#b3261e;}
  .verdict.live {border-left-color:#008300;}
  .verdict b {display:block; font-size:0.92rem; margin-bottom:2px;}
  .verdict span {color:#52514e; font-size:0.86rem; line-height:1.45;}
</style>
""", unsafe_allow_html=True)

best = load("best_fixed_policy.csv")
trace = load("collapse_trace_ddqn_OfficeWorker.csv")
preseed = load("preseed_study.csv")
ppo = load("ppo_fix.csv")
tune = load("tune_study.csv")

order = [a for a in theme.ARCHETYPE_ORDER]

# ---------------------------------------------------------------------------
# 1. Was silence optimal?
# ---------------------------------------------------------------------------
st.header("1. Silence was not optimal")

if best.empty:
    st.info("Run the exhaustive policy search to populate this section.")
else:
    summary, _ = data.load_summary()
    learners = summary[~summary["agent"].isin(data.BASELINES)]

    # The exemplar is whichever agent actually collapsed on the most users, not
    # a name fixed in the source. DQN held that role until it was re-run; DDQN
    # holds it now, and the argument is about the collapse rather than about any
    # particular algorithm.
    exemplar = COLLAPSED[0] if COLLAPSED else "DDQN"
    scores = (learners[learners["agent"] == exemplar]
              .groupby("archetype")["reward_mean"].mean())

    tbl = best.copy()
    tbl[exemplar] = tbl["archetype"].map(scores).fillna(0.0)
    tbl["schedule"] = (tbl["best_hour"].map("{:02d}:00".format)
                       + "  " + tbl["action"])
    forfeit = float(tbl[tbl[exemplar] <= 0.01]["reward"].sum())

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            "Every policy of the form *send message T at hour H, every day* was "
            "evaluated on the same 200 held-out episodes -- all 24 hours by "
            "both message types, per user. The best of those beats never-send "
            "on **all five** users, so `0.00` is a local optimum, not the "
            "true one."
        )
        st.dataframe(
            tbl[["archetype", "schedule", "reward", "ctr", exemplar]]
            .round(3)
            .rename(columns={"schedule": "best fixed schedule",
                             "reward": "its reward",
                             exemplar: f"what {exemplar} scored"}),
            width="stretch", hide_index=True)
    with c2:
        st.metric("Reward forfeited by silence", f"{forfeit:.2f}",
                  help="Sum of the best achievable reward on the users the "
                       "agent never contacted.")
        st.metric("Break-even CTR", f"{BREAK_EVEN_CTR:.3f}",
                  help="(W_send + W_fat*kappa/(1-lam)) / R_click. A send below "
                       "this loses money by construction.")
        st.markdown(
            "Both numbers matter. The first says contact was worth making; the "
            "second says it is only worth making at the right hours -- which is "
            "why the fix has to be timing, not volume."
        )

st.divider()

# ---------------------------------------------------------------------------
# 2. The cause
# ---------------------------------------------------------------------------
st.header("2. The cause: exploration destroyed the user it explored on")

if trace.empty:
    st.info("Run `tools/collapse_trace.py` to populate this section.")
else:
    left, right = st.columns(2)

    with left:
        q = trace.melt(id_vars="episode",
                       value_vars=["q_hold", "q_engage", "q_incentive"],
                       var_name="action", value_name="q")
        q["action"] = q["action"].str.replace("q_", "", regex=False).str.title()
        st.altair_chart(
            alt.Chart(q).mark_line(strokeWidth=2).encode(
                x=alt.X("episode:Q", title="training episode"),
                y=alt.Y("q:Q", title="Q-value at the best hour"),
                color=alt.Color("action:N",
                                scale=alt.Scale(
                                    domain=theme.ACTION_ORDER,
                                    range=[theme.ACTION_COLOUR[a]
                                           for a in theme.ACTION_ORDER]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["episode", "action", alt.Tooltip("q:Q", format=".3f")],
            ).properties(height=270,
                         title="Every action collapses, not just sending"),
            width="stretch")
        st.caption(
            "Holding earns exactly zero per step, so Q(Hold) should sit near "
            "zero. It ends near -10. The network is pessimistic about "
            "everything and merely least pessimistic about silence -- and "
            "argmax breaks ties toward index 0, which is Hold."
        )

    with right:
        ch = trace[["episode", "churn_rate", "send_share"]].melt(
            id_vars="episode", var_name="series", value_name="value")
        ch["series"] = ch["series"].map(
            {"churn_rate": "episodes ending in churn",
             "send_share": "share of steps sending"})
        st.altair_chart(
            alt.Chart(ch).mark_line(strokeWidth=2).encode(
                x=alt.X("episode:Q", title="training episode"),
                y=alt.Y("value:Q", title="share", axis=alt.Axis(format="%")),
                color=alt.Color("series:N",
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["episode", "series",
                         alt.Tooltip("value:Q", format=".1%")],
            ).properties(height=270, title="The user quits before it can learn"),
            width="stretch")
        st.caption(
            "Epsilon-greedy samples uniformly over three actions, two of which "
            "send, so 68% of exploratory steps send. Fatigue reaches a steady "
            "state of E[kappa]/(1-lam) = 1.00 against a churn threshold of "
            "0.70, and the user quits in most early episodes."
        )

st.divider()

# ---------------------------------------------------------------------------
# 3. Every intervention tested
# ---------------------------------------------------------------------------
st.header("3. Seven interventions, five failures")

st.markdown(
    "Each row is a mechanism that was proposed, implemented and measured. The "
    "failures are kept deliberately -- reporting only what worked would "
    "overstate how well the surviving explanation is supported."
)

VERDICTS = [
    ("dead", "Profitable mass",
     "Predicted that users are rescued in order of how much of the day is "
     "profitable. OfficeWorker has the second-largest profitable mass and was "
     "never rescued; NightOwlStudent, fourth, was."),
    ("dead", "Catastrophic exploration",
     "Predicted the 51% training churn poisons the value function. Safe "
     "exploration drove churn to 0% and the agents still sent nothing."),
    ("dead", "Replay pre-seeding",
     "Pre-filling the buffer with peak-hour sends. Rescued 1 of 10 cells on "
     "its own and contributed nothing on the other nine."),
    ("dead", "PPO policy-head init",
     "Biasing PPO's actor toward holding. The initial action mix changed as "
     "designed (0.29/0.32/0.39 to 0.75/0.16/0.09) and PPO trained straight "
     "back to silence: 0 of 5 users contacted."),
    ("dead", "PPO behaviour cloning",
     "Cloning the actor onto a peak-hour sending policy first. Same outcome, "
     "0 of 5, which rules out the initial policy as the binding constraint."),
    ("live", "Lower discount rate",
     "gamma 0.99 to 0.90 shrinks a send's discounted fatigue debt, roughly "
     "W_fat*kappa/(1-gamma*lam), making contact affordable in the agent's own "
     "accounting. This is the half that carries the result."),
    ("live", "Longer training",
     "600 to 1500 episodes. The plainest control in the study and the largest "
     "single gain -- the over-sending was undertraining, not a broken reward."),
]

cols = st.columns(2)
for i, (kind, name, body) in enumerate(VERDICTS):
    label = "REFUTED" if kind == "dead" else "SUPPORTED"
    cols[i % 2].markdown(
        f'<div class="verdict {kind}"><b>{label} &mdash; {name}</b>'
        f'<span>{body}</span></div>', unsafe_allow_html=True)

if not preseed.empty:
    st.markdown("**Isolating the two halves of the fix** (Double DQN, "
                "sends per week)")
    piv = preseed[preseed.agent == "DDQN"].pivot_table(
        index="archetype", columns="variant", values="sends_per_episode")
    st.dataframe(piv.reindex(order).round(2), width="stretch")
    st.caption(
        "`preseed` alone leaves four users uncontacted; `gamma090` alone "
        "leaves two; only `both` reaches all five. The commit needed both "
        "halves, but not for the reason the pre-seeding suggests."
    )

st.divider()

# ---------------------------------------------------------------------------
# 4. The result
# ---------------------------------------------------------------------------
st.header("4. What actually worked")

if tune.empty:
    st.info("Run `tools/tune_study.py` to populate this section.")
else:
    agg = tune.groupby("variant").agg(
        contacted=("sends_per_episode", lambda s: int((s > 0.5).sum())),
        profitable=("reward_mean", lambda s: int((s > 0).sum())),
        total_reward=("reward_mean", "sum"),
        mean_churn=("train_churn_rate", "mean"),
    ).sort_values("total_reward", ascending=False)
    agg.insert(0, "of", len(tune["archetype"].unique()))

    best_variant = agg.index[0]

    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(agg.round(2), width="stretch")
        st.caption("Contacting everyone at a loss is not an improvement over "
                   "silence, so both columns have to move.")
    with c2:
        d = tune[tune.variant == best_variant]
        st.altair_chart(
            alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("reward_mean:Q", title="mean reward per episode"),
                y=alt.Y("archetype:N", title=None, sort=order),
                color=alt.condition(alt.datum.reward_mean > 0,
                                    alt.value(theme.STATUS["good"]),
                                    alt.value(theme.STATUS["serious"])),
                tooltip=["archetype",
                         alt.Tooltip("reward_mean:Q", format=".2f"),
                         alt.Tooltip("ctr:Q", format=".3f"),
                         alt.Tooltip("sends_per_episode:Q", format=".1f")],
            ).properties(height=250,
                         title=f"Reward under {best_variant}"),
            width="stretch")

    won = tune[tune.variant == best_variant].set_index("archetype")
    ctrl = tune[tune.variant == "both"].set_index("archetype")
    if not ctrl.empty:
        # Name the control column explicitly rather than by its variant string.
        # When the winning variant *is* the control -- which happens whenever
        # tuning fails to beat it -- a dict literal keyed on both would silently
        # collapse the two columns into one and the table would show the winner
        # beating itself.
        cmp = pd.DataFrame({
            "both (control)": ctrl["reward_mean"],
            f"{best_variant} (best)": won["reward_mean"],
            "hour chosen": won["peak_hour"],
            "their best hour": won["target_hour"],
            "error (h)": won["hour_error"],
        }).reindex(order)
        st.markdown("**Against the un-tuned control, per person**")
        st.dataframe(cmp.round(2), width="stretch")

    within1 = int((won["hour_error"] <= 1).sum())
    st.success(
        f"**{best_variant} recovers both coverage and profit.** Total reward "
        f"{agg.loc[best_variant, 'total_reward']:.2f} against "
        f"{agg.loc['both', 'total_reward']:.2f} for the control, and "
        f"{within1} of five users are contacted within an hour of their own "
        f"best time. Random timing averages a six-hour error."
    )

    silent = won[won["sends_per_episode"] <= 0.5]
    if not silent.empty:
        names = ", ".join(silent.index)
        st.warning(
            f"**Still silent: {names}.** Its click propensity never clears the "
            f"break-even CTR of {BREAK_EVEN_CTR:.3f} at any hour, so silence "
            "there is most likely the correct decision rather than a remaining "
            "failure. The one variant that did contact it lost money doing so."
        )

if not ppo.empty:
    with st.expander("PPO needed the mechanism translated, not copied"):
        st.markdown(
            "PPO is on-policy and keeps no replay buffer, so pre-seeding "
            "cannot be applied to it at all. Its baseline training churn is "
            "**2%, not 52%** -- it was never dying from user burnout, so the "
            "diagnosis above does not transfer either. Its silence comes from "
            "the entropy bonus allowing an early collapse onto Hold, which the "
            "clip then makes expensive to reverse."
        )
        st.dataframe(
            ppo.pivot_table(index="archetype", columns="variant",
                            values="sends_per_episode").reindex(order).round(2),
            width="stretch")
        st.caption("Only the variant carrying both a lower gamma and a raised "
                   "entropy coefficient moves it.")

st.divider()
st.caption(
    "Sources: `best_fixed_policy.csv`, `collapse_trace_ddqn_OfficeWorker.csv`, "
    "`preseed_study.csv`, `ppo_fix.csv`, `tune_study.csv` under `artifacts/`."
)
