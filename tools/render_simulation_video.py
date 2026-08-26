"""Render the week-long simulation to a video file.

Produces the same story the dashboard's Live Simulation page tells, as a
standalone MP4: no browser chrome, reproducible from a seed, and regenerable
whenever the results change. Screen-recording the dashboard captures a session;
this captures the experiment.

    python tools/render_simulation_video.py
    python tools/render_simulation_video.py --policy optimistic --out demo.mp4
    python tools/render_simulation_video.py --fps 15 --format gif

Layout, top to bottom:

* five archetype lanes -- fatigue meter, running counters, and a 168-hour strip
  where every send is a tick and every click is a filled marker
* a live statistics panel, refreshed each frame
* cumulative reward for all five lanes, growing, against the Fixed-18:00
  reference drawn dashed

Every number on screen is read from a real rollout through `cane.CANEEnv` with
the same greedy evaluation path the reported results use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cane  # noqa: E402
from cane.core import ACTION_NAMES, IDX_FATIGUE, IDX_HOUR  # noqa: E402
from cane.persistence import load_agent  # noqa: E402

ARCHETYPES = ["OfficeWorker", "NightOwlStudent", "NightShiftWorker",
              "NormalStudent", "Housewife"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Chosen for separation under the common colour-vision deficiencies, and kept
# consistent with the dashboard so the video and the app read as one artifact.
LANE_COLOUR = {
    "OfficeWorker":     "#2a78d6",
    "NightOwlStudent":  "#eb6834",
    "NightShiftWorker": "#b3261e",
    "NormalStudent":    "#00857a",
    "Housewife":        "#7a5bb5",
}

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

DEMO_DIR = ROOT / "Code" / "models" / "demo"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def build_policy(kind: str, archetype: str):
    """Return an agent for one lane.

    `specialist` and `optimistic` are trained here rather than loaded, because
    the notebooks never save a per-archetype agent -- they train one inside
    `run_all()`, report it, and discard it.
    """
    if kind == "fixed":
        return cane.FixedScheduleAgent(hour=18)

    if kind == "specialist":
        path = DEMO_DIR / f"ddqn_demo_{archetype}.pt"
        if not path.is_file():
            raise SystemExit(
                f"missing {path.relative_to(ROOT)} -- run "
                "tools/train_demo_agents.py first")
        return load_agent(str(path))

    if kind == "tuned":
        # The configuration that won tools/tune_study.py: a lower discount
        # rate, a pre-seeded replay buffer, and 1500 training episodes. The
        # only variant that recovered coverage *and* profit, so it is what the
        # presentation video should show.
        path = DEMO_DIR / f"ddqn_demotuned_{archetype}.pt"
        if not path.is_file():
            raise SystemExit(
                f"missing {path.relative_to(ROOT)} -- run "
                "tools/train_demo_agents.py --variant tuned first")
        return load_agent(str(path))

    if kind == "optimistic":
        # The one variant the exploration study found that makes a deep agent
        # send at all: a positive bias on the two send heads only. A uniform
        # lift does nothing, because argmax breaks ties toward index 0 (Hold).
        # Prefer the saved checkpoint so rendering does not retrain five agents
        # every time; fall back to training if it has not been produced yet.
        saved = DEMO_DIR / f"ddqn_demooptimistic_{archetype}.pt"
        if saved.is_file():
            return load_agent(str(saved))
        from cane.exploration_study import VARIANTS, make_agent
        return make_agent(0, VARIANTS["optimistic_send"], archetype)

    raise SystemExit(f"unknown policy {kind!r}")


def rollout(agent, archetype: str, seed: int) -> dict:
    """One week, one lane. Returns per-hour arrays."""
    agent.reset()
    env = cane.CANEEnv(seed=7000, archetype=archetype)
    s, _ = env.reset(seed=seed)

    hours, fatigue, actions, clicks, cumulative = [], [], [], [], []
    total, done, t = 0.0, False, 0
    while not done and t < 168:
        a, _ = agent.act(s, greedy=True)
        nxt, r, term, trunc, info = env.step(a)
        total += r
        hours.append(int(s[IDX_HOUR]))
        fatigue.append(float(s[IDX_FATIGUE]))
        actions.append(int(a))
        clicks.append(bool(info.get("clicked", False)))
        cumulative.append(total)
        s, done, t = nxt, term or trunc, t + 1

    n = len(hours)
    return {
        "archetype": archetype,
        "n": n,
        "hour": np.array(hours),
        "fatigue": np.array(fatigue),
        "action": np.array(actions),
        "clicked": np.array(clicks),
        "cumulative": np.array(cumulative),
        "opted_out": bool(done and n < 168),
    }


def train_lane(kind: str, archetype: str, seed: int, episodes: int) -> dict:
    """Build, train if the agent is fresh, and roll out one lane."""
    agent = build_policy(kind, archetype)
    needs_training = (
        kind == "optimistic"
        and not (DEMO_DIR / f"ddqn_demooptimistic_{archetype}.pt").is_file())
    if needs_training:
        env = cane.CANEEnv(seed=1000, archetype=archetype)
        cane.run_episodes(agent, env, n_episodes=episodes, learn=True,
                          greedy=False)
    return rollout(agent, archetype, seed)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(traces, ref, title, subtitle, horizon):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(SURFACE)

    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=15,
             fontweight="bold", color=INK)
    fig.text(0.5, 0.943, subtitle, ha="center", va="top", fontsize=9,
             color=INK2)

    gs = fig.add_gridspec(
        2, 2, left=0.135, right=0.995, top=0.865, bottom=0.075,
        height_ratios=[1.05, 1.0], width_ratios=[2.75, 1.0],
        hspace=0.34, wspace=0.10)

    ax_lanes = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[:, 1])
    ax_race = fig.add_subplot(gs[1, 0])

    # --- lanes -------------------------------------------------------------
    ax_lanes.set_xlim(0, horizon)
    ax_lanes.set_ylim(-0.6, len(ARCHETYPES) - 0.35)
    ax_lanes.invert_yaxis()
    ax_lanes.set_facecolor(SURFACE)
    for spine in ax_lanes.spines.values():
        spine.set_visible(False)
    ax_lanes.set_yticks(range(len(ARCHETYPES)))
    ax_lanes.set_yticklabels(ARCHETYPES, fontsize=9, color=INK,
                             fontweight="semibold")
    ax_lanes.set_xticks(range(0, horizon + 1, 24))
    ax_lanes.set_xticklabels(DAYS + [""], fontsize=8, color=MUTED)
    ax_lanes.tick_params(length=0)
    ax_lanes.set_title("Each lane: one user, one week.  Ticks are sends, "
                       "filled dots are clicks.",
                       fontsize=9, color=MUTED, loc="left", pad=6)
    for d in range(0, horizon + 1, 24):
        ax_lanes.axvline(d, color=GRID, lw=0.8, zorder=0)

    # Fatigue is drawn relative to the largest value reached anywhere in this
    # run. Absolute scaling would flatten every lane to a line, because a policy
    # that paces itself well never gets far from zero. The stats panel carries
    # the absolute number.
    peak_fat = max(
        [float(np.max(t["fatigue"])) for t in traces.values()] + [0.05])

    artists = {}
    for i, arch in enumerate(ARCHETYPES):
        colour = LANE_COLOUR[arch]
        # Fatigue is drawn as a filled trace along the lane rather than a bar:
        # the shape over the week is the interesting part, and a bar would only
        # ever show the final value.
        fat_fill = ax_lanes.fill_between([0], [i], [i], color=colour,
                                         alpha=0.18, lw=0)
        (fat_line,) = ax_lanes.plot([], [], color=colour, lw=1.3, alpha=0.9)
        sends = ax_lanes.scatter([], [], marker="|", s=90, linewidths=1.4,
                                 color=colour, alpha=0.85)
        hits = ax_lanes.scatter([], [], marker="o", s=26, color=colour,
                                edgecolors="white", linewidths=0.8, zorder=4)
        artists[arch] = {"fat_line": fat_line, "fat_fill": fat_fill,
                         "sends": sends, "hits": hits, "row": i,
                         "colour": colour}

    playhead_lane = ax_lanes.axvline(0, color=INK, lw=1.2, alpha=0.7, zorder=6)

    # --- stats panel -------------------------------------------------------
    ax_stats.axis("off")
    stats_text = ax_stats.text(
        0.0, 1.0, "", ha="left", va="top", fontsize=8.4,
        family="DejaVu Sans Mono", color=INK, linespacing=1.5,
        transform=ax_stats.transAxes)

    # --- race chart --------------------------------------------------------
    ax_race.set_facecolor(SURFACE)
    ax_race.set_xlim(0, horizon)
    lo = min(float(np.min(t["cumulative"])) for t in traces.values())
    hi = max(float(np.max(t["cumulative"])) for t in traces.values())
    if ref:
        lo = min(lo, min(float(np.min(t["cumulative"])) for t in ref.values()))
        hi = max(hi, max(float(np.max(t["cumulative"])) for t in ref.values()))
    pad = 0.12 * max(hi - lo, 1.0)
    ax_race.set_ylim(lo - pad, hi + pad)
    ax_race.axhline(0, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax_race.grid(True, color=GRID, lw=0.7)
    ax_race.set_axisbelow(True)
    for side in ("top", "right"):
        ax_race.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_race.spines[side].set_color(GRID)
    ax_race.tick_params(colors=MUTED, labelsize=8)
    ax_race.set_xlabel("hour of the week", fontsize=9, color=INK2)
    ax_race.set_ylabel("cumulative reward", fontsize=9, color=INK2)
    ax_race.set_xticks(range(0, horizon + 1, 24))

    # Archetypes frequently land on identical cumulative reward -- five lines
    # drew as two in testing. Dash patterns keep them tellable apart where they
    # coincide, without implying an ordering the way varying widths would.
    DASHES = [(0, ()), (0, (6, 2)), (0, (2, 2)), (0, (7, 2, 2, 2)),
              (0, (1, 2))]
    race_lines, ref_lines = {}, {}
    for k, arch in enumerate(ARCHETYPES):
        (ln,) = ax_race.plot([], [], color=LANE_COLOUR[arch], lw=2.0,
                             ls=DASHES[k % len(DASHES)], label=arch)
        race_lines[arch] = ln
        if ref:
            (rl,) = ax_race.plot([], [], color=MUTED, lw=1.1,
                                 ls=(0, (4, 3)), alpha=0.6)
            ref_lines[arch] = rl

    handles = [race_lines[a] for a in ARCHETYPES]
    labels = list(ARCHETYPES)
    if ref:
        handles.append(ref_lines[ARCHETYPES[0]])
        labels.append("Fixed-18:00 reference")
    ax_race.legend(handles, labels, loc="upper left", fontsize=7.6,
                   frameon=False, ncol=3, labelcolor=INK2,
                   handlelength=2.6, columnspacing=1.4)

    playhead_race = ax_race.axvline(0, color=INK, lw=1.2, alpha=0.7)

    return (fig, artists, stats_text, race_lines, ref_lines,
            playhead_lane, playhead_race, ax_lanes, peak_fat)


def make_frame_fn(traces, ref, artists, stats_text, race_lines, ref_lines,
                  playhead_lane, playhead_race, ax_lanes, policy_label,
                  peak_fat):
    def update(frame: int):
        t = frame
        lines = [f"{'LIVE STATISTICS':^30}", "-" * 30]

        any_trace = next(iter(traces.values()))
        idx = min(t, any_trace["n"] - 1)
        hour = int(any_trace["hour"][idx])
        # The policy label is long enough to run past the panel, so it gets
        # its own line rather than being truncated mid-word.
        lines += [f"  hour of week : {t:>3d} / 168",
                  f"  clock        : {DAYS[min(t // 24, 6)]} {hour:02d}:00",
                  "  policy       :",
                  f"    {policy_label[:26]}",
                  "", f"{'PER USER':^30}", "-" * 30]

        for arch in ARCHETYPES:
            tr = traces[arch]
            i = min(t, tr["n"] - 1)
            art = artists[arch]
            row = art["row"]

            fat = tr["fatigue"][: i + 1]
            xs = np.arange(len(fat))
            # Fatigue occupies the upper 0.42 of the lane's row.
            ys = row - (fat / peak_fat) * 0.42
            art["fat_line"].set_data(xs, ys)
            art["fat_fill"].remove()
            art["fat_fill"] = ax_lanes.fill_between(
                xs, np.full_like(ys, row), ys, color=art["colour"],
                alpha=0.18, lw=0)

            sent_idx = np.where(tr["action"][: i + 1] != 0)[0]
            art["sends"].set_offsets(
                np.column_stack([sent_idx, np.full(len(sent_idx), row + 0.22)])
                if len(sent_idx) else np.empty((0, 2)))

            hit_idx = np.where(tr["clicked"][: i + 1])[0]
            art["hits"].set_offsets(
                np.column_stack([hit_idx, np.full(len(hit_idx), row + 0.22)])
                if len(hit_idx) else np.empty((0, 2)))

            sends = int(len(sent_idx))
            clicks = int(len(hit_idx))
            reward = float(tr["cumulative"][i])
            ctr = clicks / sends if sends else 0.0
            lines.append(f"  {arch[:16]:<16}")
            lines.append(f"    sent {sends:>3d}  clk {clicks:>3d}  "
                         f"ctr {ctr:4.2f}")
            lines.append(f"    fatigue {tr['fatigue'][i]:4.2f}  "
                         f"reward {reward:+7.2f}")

            race_lines[arch].set_data(np.arange(i + 1),
                                      tr["cumulative"][: i + 1])
            if ref and arch in ref_lines:
                rt = ref[arch]
                j = min(t, rt["n"] - 1)
                ref_lines[arch].set_data(np.arange(j + 1),
                                         rt["cumulative"][: j + 1])

        if ref:
            total = sum(float(traces[a]["cumulative"][min(t, traces[a]["n"] - 1)])
                        for a in ARCHETYPES)
            base = sum(float(ref[a]["cumulative"][min(t, ref[a]["n"] - 1)])
                       for a in ARCHETYPES)
            lines += ["", "-" * 30,
                      f"  vs Fixed-18:00 : {total - base:+7.2f}"]

        stats_text.set_text("\n".join(lines))
        playhead_lane.set_xdata([t, t])
        playhead_race.set_xdata([t, t])
        return []

    return update


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="tuned",
                    choices=["fixed", "specialist", "optimistic", "tuned"],
                    help="which policy drives the five lanes")
    ap.add_argument("--seed", type=int, default=900_000)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES,
                    help="training episodes, for --policy optimistic")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold", type=float, default=2.0,
                    help="seconds to hold on the final frame")
    ap.add_argument("--format", default="mp4", choices=["mp4", "gif"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    label = {"fixed": "Fixed-18:00 (no learning)",
             "specialist": "Double DQN, per archetype",
             "optimistic": "Double DQN + optimistic send init",
             "tuned": "Double DQN, tuned"}[args.policy]

    print(f"policy   : {label}")
    print(f"seed     : {args.seed}")
    print("simulating five lanes...", flush=True)

    t0 = time.time()
    traces = {}
    for arch in ARCHETYPES:
        traces[arch] = train_lane(args.policy, arch, args.seed, args.episodes)
        tr = traces[arch]
        sends = int((tr["action"] != 0).sum())
        print(f"  {arch:17} sent {sends:3d}  clicks "
              f"{int(tr['clicked'].sum()):3d}  "
              f"reward {tr['cumulative'][-1]:+8.2f}", flush=True)

    ref = ({arch: rollout(cane.FixedScheduleAgent(hour=18), arch, args.seed)
            for arch in ARCHETYPES} if args.policy != "fixed" else {})

    horizon = max(t["n"] for t in traces.values())

    title = "CANE - Context-Aware Notification Engine"
    subtitle = (f"One week, hourly decisions, five user archetypes  |  "
                f"policy: {label}  |  episode seed {args.seed}")

    (fig, artists, stats_text, race_lines, ref_lines, ph_lane, ph_race,
     ax_lanes, peak_fat) = build_figure(traces, ref, title, subtitle, horizon)

    update = make_frame_fn(traces, ref, artists, stats_text, race_lines,
                           ref_lines, ph_lane, ph_race, ax_lanes, label,
                           peak_fat)

    hold_frames = int(args.hold * args.fps)
    frames = list(range(horizon)) + [horizon - 1] * hold_frames

    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"cane_simulation_{args.policy}.{args.format}")
    out.parent.mkdir(parents=True, exist_ok=True)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps,
                         blit=False, repeat=False)

    writer = (FFMpegWriter(fps=args.fps, bitrate=3200)
              if args.format == "mp4" else PillowWriter(fps=args.fps))
    print(f"\nrendering {len(frames)} frames to {out.name} ...", flush=True)
    anim.save(str(out), writer=writer, savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)

    size_mb = out.stat().st_size / 1e6
    # `out` may be relative (from --out) or absolute (the default), so resolve
    # before comparing; relative_to raises on a mismatch of the two kinds.
    shown = out.resolve()
    try:
        shown = shown.relative_to(ROOT)
    except ValueError:
        pass
    print(f"wrote {shown}  ({size_mb:.1f} MB, "
          f"{len(frames) / args.fps:.0f}s)  in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
