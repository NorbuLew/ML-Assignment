"""Animated week-long simulation, all five user archetypes side by side.

This is the presentation page. Every other page reports what happened; this one
shows it happening -- 168 hourly steps, five different simulated users, the same
agent policy driving all of them, running at whatever speed the presenter wants.

The point it makes is the one the results tables make badly. A table saying
`0.00` on three archetypes reads as a broken run. Watching the same agent stay
silent through a night-shift worker's entire waking window, while a housewife
two lanes over racks up clicks, reads as a policy decision -- which is what it
actually is.

Traces are precomputed per (agent, archetype, seed) and cached, so pressing play
replays a fixed episode rather than re-simulating. That matters for a live demo:
the run you rehearsed is the run the audience sees.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cane  # noqa: E402
from cane.core import (  # noqa: E402
    ACTION_NAMES, ARCHETYPE_TABLE, IDX_FATIGUE, IDX_HOUR,
)
from cane.persistence import load_agent  # noqa: E402
from dashboard.lib import theme  # noqa: E402

st.set_page_config(page_title="Live simulation - CANE", page_icon="🔔",
                   layout="wide")

# A face per archetype. Purely presentational, and load-bearing anyway: in a
# five-lane layout the eye finds a glyph far faster than it reads a label.
FACES = {
    "OfficeWorker": "🧑‍💼",
    "NightOwlStudent": "🦉",
    "NightShiftWorker": "🌙",
    "NormalStudent": "🎓",
    "Housewife": "🏠",
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.markdown("""
<style>
  .block-container {padding-top: 2.0rem; max-width: 1500px;}
  .lede {color: #52514e; font-size: 1.02rem; line-height: 1.55; max-width: 76ch;}

  .lane {
    border: 1px solid #e6e5e1; border-radius: 12px; padding: 12px 14px 10px;
    background: #fcfcfb; height: 100%;
  }
  .lane-head {display: flex; align-items: baseline; gap: 8px; margin-bottom: 2px;}
  .lane-face {font-size: 1.45rem; line-height: 1;}
  .lane-name {font-weight: 600; font-size: 0.95rem; color: #0b0b0b;}
  .lane-clock {
    margin-left: auto; font-variant-numeric: tabular-nums;
    font-size: 0.78rem; color: #8a8983; white-space: nowrap;
  }
  .lane-name {overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}

  .meter {
    height: 7px; border-radius: 4px; background: #eeedea;
    overflow: hidden; margin: 8px 0 4px;
  }
  .meter > span {display: block; height: 100%; border-radius: 4px;}

  .lane-stats {
    display: flex; gap: 14px; font-size: 0.78rem; color: #52514e;
    font-variant-numeric: tabular-nums; margin-top: 6px;
  }
  .meter-row {
    display: flex; align-items: center; gap: 8px; margin: 9px 0 2px;
  }
  .meter-row .meter {flex: 1; margin: 0;}
  .meter-label {
    font-size: 0.7rem; color: #8a8983; font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .lane-stats b {color: #0b0b0b; font-weight: 600;}

  .feed {
    margin-top: 8px; min-height: 76px; display: flex;
    flex-direction: column; gap: 3px;
  }
  .evt {
    font-size: 0.74rem; padding: 3px 7px; border-radius: 5px;
    background: #f2f1ee; color: #52514e;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .evt.click {background: #e4f2e4; color: #005c00;}
  .evt.send  {background: #e3edfa; color: #14508f;}
  .evt.out   {background: #fae3e3; color: #8f1414;}
  .quiet {color: #b3b2ad; font-size: 0.74rem; font-style: italic;}
  .margin {
    margin-top: 5px; font-size: 0.76rem; font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .margin.up   {color: #008300;}
  .margin.down {color: #b3261e;}
  .margin.flat {color: #8a8983;}
</style>
""", unsafe_allow_html=True)

st.title("Live simulation")
_LEDE = ('<p class="lede">One week, hour by hour, five simulated users. Each '
         'lane is an independent episode in the same environment the reported '
         'numbers come from.</p>')
st.markdown(_LEDE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------
# One agent trained per archetype, written by tools/train_demo_agents.py. This
# is the code path the notebooks *measure* (run_all trains a fresh agent per
# archetype) but never save, so without these the dashboard can only ever show
# the mixed-population agent -- which learned to hold on every lane.
DEMO_DIR = ROOT / "Code" / "models" / "demo"

# Every algorithm gets an entry, not just the one that wins. A page that can
# only show the best result cannot show the comparison the project is about --
# and several of these are near-silent on several lanes, which is itself the
# finding. The Overview page counts exactly which; this menu does not claim it.
#
# Each entry is (label, filename template). The template is filled with the
# archetype; a set is offered only when all five of its files exist, so a
# half-finished training run never produces a menu entry that errors on click.
POLICY_SETS = [
    ("Min-contact Double DQN (5 of 5 contacted)", "ddqn_demomincontact_{}.pt"),
    ("Tuned Double DQN (4 of 5 contacted)", "ddqn_demotuned_{}.pt"),
    ("LinUCB (5 of 5 contacted)", "linucb_demo_{}.npz"),
    ("Double DQN (default settings)", "ddqn_demo_{}.pt"),
    ("DQN (default settings)", "dqn_demo_{}.pt"),
    ("PPO (default settings)", "ppo_demo_{}.pt"),
    ("Optimistic-init Double DQN", "ddqn_demooptimistic_{}.pt"),
]

# Fixed-18:00 is always offered, checkpoints or not. It needs no training, and
# it is the reference the learned agents are measured against -- a demo that
# can only show learned policies cannot show what they failed to beat.
BASELINE_LABEL = "Fixed-18:00 (no learning)"

lane_sets: dict[str, dict[str, Path]] = {}
options: dict[str, object] = {}
for label, template in POLICY_SETS:
    paths = {arch: DEMO_DIR / template.format(arch)
             for arch in theme.ARCHETYPE_ORDER}
    if all(q.is_file() for q in paths.values()):
        lane_sets[label] = paths
        options[label] = label
options[BASELINE_LABEL] = None

with st.sidebar:
    st.subheader("Simulation")
    # Specialists first when they exist: they are the only option in which the
    # five lanes behave differently, which is the whole point of the page.
    choice = st.selectbox("Policy", list(options), index=0)
    seed = st.number_input("Episode seed", min_value=900_000, max_value=999_999,
                           value=900_000, step=1,
                           help="900000-900199 are the held-out evaluation "
                                "episodes.")
    speed = st.select_slider(
        "Playback speed", options=["slow", "normal", "fast", "instant"],
        value="normal",
        help="Hours per second during playback. 'instant' jumps to the end.")
    st.caption("Traces are deterministic for a given policy and seed, so the "
               "run you rehearse is the run you present.")

DELAY = {"slow": 0.10, "normal": 0.045, "fast": 0.018, "instant": 0.0}[speed]


@st.cache_resource(show_spinner=False)
def get_policy(path_str: str | None):
    """Load a checkpoint, or hand back the non-learning baseline."""
    if path_str is None:
        return cane.FixedScheduleAgent(hour=18)
    return load_agent(path_str)


@st.cache_data(show_spinner="Simulating five users...")
def simulate(path_str: str | None, archetype: str, seed: int) -> pd.DataFrame:
    """One full episode, one row per hour.

    Cached on (policy, archetype, seed) so scrubbing the playhead or pressing
    play again is free -- the animation reads a frame out of this frame, it does
    not re-run the environment.
    """
    agent = get_policy(path_str)
    agent.reset()
    env = cane.CANEEnv(seed=7000, archetype=archetype)
    s, _ = env.reset(seed=int(seed))

    rows, total, done, t = [], 0.0, False, 0
    while not done and t < 168:
        a, _ = agent.act(s, greedy=True)
        nxt, r, term, trunc, info = env.step(a)
        total += r
        rows.append({
            "t": t,
            "hour": int(s[IDX_HOUR]),
            "day": t // 24,
            "fatigue": float(s[IDX_FATIGUE]),
            "action": ACTION_NAMES[int(a)],
            "clicked": bool(info.get("clicked", False)),
            "reward": float(r),
            "cumulative": total,
            "opted_out": bool(term),
        })
        s, done, t = nxt, term or trunc, t + 1
    return pd.DataFrame(rows)


path = options[choice]
is_specialist = path is not None

if is_specialist:
    lane_paths = lane_sets[path]
    traces = {arch: simulate(str(lane_paths[arch]), arch, int(seed))
              for arch in theme.ARCHETYPE_ORDER}
else:
    traces = {arch: simulate(None, arch, int(seed))
              for arch in theme.ARCHETYPE_ORDER}

# The reference is simulated whenever a learned policy is selected. A learned
# policy that holds all week is a flat line at zero; on its own that reads as no
# data, and against the baseline it reads as the result it actually is. When the
# baseline itself is the selection there is nothing to compare it against -- it
# would be drawn twice and every margin would read as +-0.00.
show_reference = is_specialist
ref = ({arch: simulate(None, arch, int(seed)) for arch in theme.ARCHETYPE_ORDER}
       if show_reference else {})

horizon = max(len(df) for df in traces.values())


# ---------------------------------------------------------------------------
# Transport controls
# ---------------------------------------------------------------------------
if "playhead" not in st.session_state:
    st.session_state.playhead = horizon - 1

if path and path.startswith("Tuned"):
    st.caption("Each lane is its own agent under the configuration that won "
               "the tuning study: a lower discount rate, a pre-seeded replay "
               "buffer, and 1500 training episodes. Watch where each lane "
               "places its sends -- three of the five land within an hour of "
               "that person's own best time.")
elif path and path.startswith("LinUCB"):
    st.caption("A contextual bandit, which optimises the immediate click and "
               "cannot represent the fatigue a send leaves behind. It contacts "
               "everyone, constantly, and loses money on all five -- the "
               "clearest argument in the project for why this task needs a "
               "sequential method.")
elif path:
    st.caption("Each lane is driven by an agent trained on that archetype "
               "alone -- the setting the results CSVs measure. At default "
               "settings these mostly learned to hold, which is the finding, "
               "not a fault. Switch to the tuned policy to see what changes.")
else:
    st.caption("The non-learning reference: one notification at 18:00 daily, "
               "to everyone, regardless of who they are.")

c_play, c_reset, c_scrub = st.columns([1, 1, 8])
play = c_play.button("Play week", type="primary", width="stretch")
if c_reset.button("Reset", width="stretch"):
    st.session_state.playhead = 0

playhead = c_scrub.slider(
    "Hour of the week", 0, horizon - 1,
    value=min(st.session_state.playhead, horizon - 1),
    help="Drag to scrub. Hour 0 is Monday 00:00.")
st.session_state.playhead = playhead


# ---------------------------------------------------------------------------
# Rendering one frame
# ---------------------------------------------------------------------------
def fatigue_colour(f: float) -> str:
    """Green through amber to red as fatigue climbs.

    Three stops rather than a continuous ramp: at 7px tall a gradient reads as
    one indeterminate colour, whereas three well-separated stops still say
    'fine / watch this / too far' at a glance from the back of a room.
    """
    if f < 0.33:
        return theme.STATUS["good"]
    if f < 0.66:
        return theme.STATUS["warning"]
    return theme.STATUS["serious"]


def lane_html(archetype: str, df: pd.DataFrame, t: int,
              ref_df: pd.DataFrame | None = None) -> str:
    """The whole lane as one HTML string.

    One string rather than a stack of Streamlit widgets because the animation
    rewrites all five lanes on every tick: a single markdown write per lane is
    fast enough to hold ~20fps, whereas five widget calls per lane is not.
    """
    face = FACES.get(archetype, "*")
    if df.empty:
        return f'<div class="lane"><div class="lane-name">{archetype}</div></div>'

    # A lane whose user opted out early freezes on its final hour rather than
    # disappearing -- the audience needs to see that it ended, not lose the lane.
    idx = min(t, len(df) - 1)
    row = df.iloc[idx]
    upto = df.iloc[: idx + 1]

    ended_early = bool(df.iloc[-1]["opted_out"]) and t >= len(df) - 1
    sends = int((upto["action"] != "Hold").sum())
    clicks = int(upto["clicked"].sum())
    reward = float(row["cumulative"])
    fat = float(row["fatigue"])

    day = DAYS[min(int(row["day"]), 6)]
    clock = f"{day} {int(row['hour']):02d}:00"

    # The margin is the point of the lane when the policy is silent: without it
    # a flat zero says nothing about whether zero was a good score.
    margin_html = ""
    if ref_df is not None and not ref_df.empty:
        r_idx = min(t, len(ref_df) - 1)
        r_cum = float(ref_df.iloc[r_idx]["cumulative"])
        delta = reward - r_cum
        cls = "up" if delta > 0.05 else ("down" if delta < -0.05 else "flat")
        margin_html = (f'<div class="margin {cls}">{delta:+.1f} '
                       f'vs Fixed-18:00</div>')

    # Recent events, newest first, capped at four lines so the lane height does
    # not jump around mid-animation.
    events = []
    for _, e in upto.iloc[::-1].iterrows():
        if len(events) >= 4:
            break
        stamp = f"{DAYS[min(int(e['day']), 6)]} {int(e['hour']):02d}:00"
        if e["clicked"]:
            events.append(('click', f"{stamp}  clicked {e['action'].lower()}"))
        elif e["action"] != "Hold":
            events.append(('send', f"{stamp}  sent {e['action'].lower()}"))
    if ended_early:
        events.insert(0, ('out', "opted out - episode ended"))

    feed = "".join(f'<div class="evt {k}">{txt}</div>' for k, txt in events)
    if not feed:
        feed = '<div class="quiet">no notifications yet</div>'

    return f"""
<div class="lane">
  <div class="lane-head">
    <span class="lane-face">{face}</span>
    <span class="lane-name">{archetype}</span>
    <span class="lane-clock">{clock}</span>
  </div>
  <div class="meter-row">
    <div class="meter"><span style="width:{max(fat, 0.012) * 100:.1f}%;
         background:{fatigue_colour(fat)};"></span></div>
    <span class="meter-label">fatigue {fat:.2f}</span>
  </div>
  <div class="lane-stats">
    <span>sent <b>{sends}</b></span>
    <span>clicks <b>{clicks}</b></span>
    <span>reward <b>{reward:+.1f}</b></span>
  </div>
  {margin_html}
  <div class="feed">{feed}</div>
</div>
"""


def race_chart(t: int):
    """Cumulative reward for all five lanes, drawn only up to hour `t`.

    Clipping to the playhead rather than drawing the whole week and moving a
    marker is what makes it read as a race: the lines grow.
    """
    frames = []
    for arch, df in traces.items():
        if df.empty:
            continue
        upto = df.iloc[: min(t, len(df) - 1) + 1][["t", "cumulative"]].copy()
        upto["archetype"] = arch
        frames.append(upto)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)

    order = [a for a in theme.ARCHETYPE_ORDER if a in set(d["archetype"])]
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeWidth=1, color=theme.TEXT_MUTED, strokeDash=[3, 3]).encode(y="y:Q")

    # Reference lines share one muted colour and carry no per-archetype legend
    # entry: they are context, and five more colours would compete with the
    # five that matter.
    ref_layer = None
    if show_reference:
        rframes = []
        for arch, df in ref.items():
            if df.empty:
                continue
            upto = df.iloc[: min(t, len(df) - 1) + 1][["t", "cumulative"]].copy()
            upto["archetype"] = arch
            rframes.append(upto)
        if rframes:
            rd = pd.concat(rframes, ignore_index=True)
            ref_layer = alt.Chart(rd).mark_line(
                strokeWidth=1.4, strokeDash=[4, 3], opacity=0.55,
                color=theme.TEXT_MUTED).encode(
                x=alt.X("t:Q", scale=alt.Scale(domain=[0, horizon - 1],
                                               nice=False)),
                y="cumulative:Q",
                detail="archetype:N",
                tooltip=[alt.Tooltip("archetype:N"),
                         alt.Tooltip("cumulative:Q", title="Fixed-18:00",
                                     format=".2f")])
    lines = alt.Chart(d).mark_line(strokeWidth=2.2).encode(
        strokeDash=alt.StrokeDash("archetype:N", sort=order,
                                  legend=None),
        x=alt.X("t:Q", title="hour of the week",
                scale=alt.Scale(domain=[0, horizon - 1], nice=False)),
        y=alt.Y("cumulative:Q", title="cumulative reward"),
        color=alt.Color("archetype:N", sort=order,
                        scale=alt.Scale(scheme="tableau10"),
                        legend=alt.Legend(title=None, orient="bottom",
                                          columns=5)),
        tooltip=[alt.Tooltip("archetype:N"), alt.Tooltip("t:Q", title="hour"),
                 alt.Tooltip("cumulative:Q", title="reward", format=".2f")],
    )
    layers = [zero] + ([ref_layer] if ref_layer is not None else []) + [lines]
    title = ("Cumulative reward - solid is the selected policy, "
             "dashed grey is Fixed-18:00" if show_reference
             else "Cumulative reward")
    return alt.layer(*layers).properties(height=280, title=title)


lane_slots = st.columns(5)
lane_boxes = [c.empty() for c in lane_slots]
chart_box = st.empty()


def draw(t: int):
    for box, arch in zip(lane_boxes, theme.ARCHETYPE_ORDER):
        box.markdown(lane_html(arch, traces[arch], t, ref.get(arch)),
                     unsafe_allow_html=True)
    chart = race_chart(t)
    if chart is not None:
        chart_box.altair_chart(chart, width="stretch")


if play:
    # A plain loop over placeholders, not st.rerun per frame. Reruns re-execute
    # the whole page each tick, which drops frames and makes a screen recording
    # stutter; overwriting placeholders in one pass does not.
    for t in range(0, horizon):
        draw(t)
        if DELAY:
            time.sleep(DELAY)
    st.session_state.playhead = horizon - 1
else:
    draw(playhead)


# ---------------------------------------------------------------------------
# What the week added up to
# ---------------------------------------------------------------------------
st.divider()

summary = []
for arch, df in traces.items():
    if df.empty:
        continue
    upto = df.iloc[: min(st.session_state.playhead, len(df) - 1) + 1]
    sent = int((upto["action"] != "Hold").sum())
    clicked = int(upto["clicked"].sum())
    summary.append({
        "archetype": arch,
        "sent": sent,
        "clicks": clicked,
        "CTR": round(clicked / sent, 3) if sent else 0.0,
        "peak propensity": round(float(np.max(ARCHETYPE_TABLE[arch])), 3),
        "reward": round(float(upto.iloc[-1]["cumulative"]), 2),
        "vs Fixed-18:00": (
            round(float(upto.iloc[-1]["cumulative"])
                  - float(ref[arch].iloc[min(len(upto), len(ref[arch])) - 1]
                          ["cumulative"]), 2)
            if show_reference and arch in ref and not ref[arch].empty else None),
        "ended early": bool(df.iloc[-1]["opted_out"]),
    })

s1, s2 = st.columns([3, 2])
with s1:
    st.subheader("Week so far")
    st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)

with s2:
    st.subheader("Reading the lanes")
    rewards = [round(r["reward"], 2) for r in summary]
    if len(set(rewards)) < len(rewards):
        st.caption(
            "Lanes sharing a cumulative reward draw as one line above. That is "
            "the data, not a rendering fault: a policy that ignores who it is "
            "talking to produces near-identical weeks."
        )

    silent = [r["archetype"] for r in summary if r["sent"] == 0]
    if silent:
        beat = [r for r in summary
                if r.get("vs Fixed-18:00") is not None
                and r["vs Fixed-18:00"] > 0]
        msg = ("Silent all week on: **" + ", ".join(silent) + "**.\n\n"
               "That is the policy choosing to hold, not a failed run.")
        if beat:
            msg += (f"\n\nAnd it is *winning*: on {len(beat)} of "
                    f"{len(summary)} archetypes, sending nothing scores higher "
                    "than the fixed 18:00 schedule, because every send that "
                    "does not get clicked costs send price plus fatigue.")
        msg += ("\n\nWhether holding is the *right* choice is answered on the "
                "**Diagnosis** page: a send pays only when the click "
                "probability clears the break-even rate of 0.340, and every "
                "archetype clears it for a few hours a day -- so silence is a "
                "local optimum, not the best available policy.")
        st.info(msg)
    else:
        st.info("Every lane received at least one notification this week.")
    st.caption(
        "Fatigue meters turn amber past 0.33 and red past 0.66. A lane that "
        "ends early hit the opt-out threshold - the user left, and the episode "
        "stopped there."
    )


# ---------------------------------------------------------------------------
# Pre-rendered fallback
# ---------------------------------------------------------------------------
# The live animation above is the thing to show. This is the same week rendered
# offline by tools/render_simulation_video.py, kept for two reasons: it plays at
# a fixed frame rate on a machine that is busy, and it is the one artifact that
# still works if the checkpoints are unavailable during a graded demo.
_VIDEOS = [
    ("Tuned Double DQN", ROOT / "artifacts" / "cane_simulation_tuned.mp4"),
    ("Optimistic-initialisation variant",
     ROOT / "artifacts" / "cane_simulation_optimistic.mp4"),
]
_available = [(lab, p) for lab, p in _VIDEOS if p.is_file()]

if _available:
    st.divider()
    with st.expander("Pre-rendered walkthrough (offline copy of this page)"):
        if len(_available) > 1:
            _pick = st.selectbox("Render", [lab for lab, _ in _available],
                                 index=0, key="video_pick")
            _path = dict(_available)[_pick]
        else:
            _pick, _path = _available[0]
        st.video(str(_path))
        st.caption(
            f"`artifacts/{_path.name}` - the same five lanes over the same "
            "week, rendered ahead of time. Use the live animation above by "
            "preference; this is the fallback."
        )
