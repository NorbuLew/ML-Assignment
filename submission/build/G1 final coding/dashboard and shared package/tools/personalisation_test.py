"""Does the agent learn each person's rhythm?

The assignment's goal is personalisation: send at the hour that suits each
individual's way of living. An agent that stays silent cannot demonstrate that
one way or the other, so this pairs every agent with a minimum-contact quota --
it must contact the user at least `--min-sends` times a week, and it chooses
when -- and then measures *which hours it picks*.

Reward is reported, but the headline metric is personalisation:

* `peak_hour`     -- the hour the agent sends at most often
* `target_hour`   -- the best hour found by exhaustive search over fixed policies
* `hour_error`    -- circular distance between them, in hours
* `spread`        -- how concentrated the send-hour distribution is; a
                     personalised policy should be peaked, not uniform
* `distinctness`  -- whether the agent picks *different* hours for different
                     people, which is the thing that separates personalisation
                     from one global schedule

    python tools/personalisation_test.py
    python tools/personalisation_test.py --agents dqn ddqn --min-sends 7
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cane  # noqa: E402
from cane.agents_deep import DDQNAgent, DQNAgent, PPOAgent  # noqa: E402
from cane.core import IDX_HOUR  # noqa: E402
from cane.min_contact import MinContactAgent  # noqa: E402

OUT = ROOT / "artifacts"

# Best hour per archetype, from the exhaustive search over every
# "send <type> at hour H daily" policy on the held-out episodes.
TARGET_HOUR = {
    "OfficeWorker": 20,
    "NightOwlStudent": 23,
    "NightShiftWorker": 16,
    "NormalStudent": 16,
    "Housewife": 13,
}

MAKERS = {"dqn": DQNAgent, "ddqn": DDQNAgent, "ppo": PPOAgent}


def circular_hour_error(a: int, b: int) -> int:
    """Hours wrap, so 23:00 and 01:00 are two apart, not twenty-two."""
    d = abs(int(a) - int(b)) % 24
    return min(d, 24 - d)


def send_hours(agent, archetype: str, seeds) -> np.ndarray:
    """Histogram over hour-of-day of every send the greedy policy makes."""
    counts = np.zeros(24, dtype=int)
    env = cane.CANEEnv(seed=7000, archetype=archetype)
    for sd in seeds:
        if hasattr(agent, "reset"):
            agent.reset()
        s, _ = env.reset(seed=int(sd))
        done = False
        while not done:
            a, _ = agent.act(s, greedy=True)
            if a != 0:
                counts[int(s[IDX_HOUR])] += 1
            s, _, term, trunc, _ = env.step(a)
            done = term or trunc
    return counts


def run_cell(kind: str, archetype: str, seed: int, episodes: int,
             min_sends: int) -> dict:
    t0 = time.time()
    inner = MAKERS[kind](seed=seed)
    agent = MinContactAgent(inner, min_sends=min_sends, window=24,
                            horizon=168)

    train_env = cane.CANEEnv(seed=1000 + seed, archetype=archetype)
    for _ in range(episodes):
        agent.reset()
        s, _ = train_env.reset()
        done = False
        while not done:
            a, aux = agent.act(s, greedy=False)
            nxt, r, term, trunc, _ = train_env.step(a)
            agent.update(s, a, r, nxt, term, aux)
            s, done = nxt, term or trunc

    eval_env = cane.CANEEnv(seed=7000 + seed, archetype=archetype)
    m, _, _, _ = cane.run_episodes(agent, eval_env, seeds=cane.EVAL_SEEDS,
                                   learn=False, greedy=True)

    counts = send_hours(agent, archetype, cane.EVAL_SEEDS[:40])
    total = int(counts.sum())
    peak = int(np.argmax(counts)) if total else -1
    # Share of sends landing in a 3-hour window around the modal hour. A
    # personalised policy concentrates; a policy that has learned nothing about
    # timing spreads its sends across the day.
    if total:
        idx = [(peak - 1) % 24, peak, (peak + 1) % 24]
        concentration = float(counts[idx].sum()) / total
    else:
        concentration = 0.0

    return {
        "agent": kind.upper(), "archetype": archetype, "seed": seed,
        "reward_mean": float(m["reward_mean"]), "ctr": float(m["ctr"]),
        "sends_per_episode": float(m["sends_per_episode"]),
        "optout_rate": float(m["optout_rate"]),
        "peak_hour": peak, "target_hour": TARGET_HOUR[archetype],
        "hour_error": circular_hour_error(peak, TARGET_HOUR[archetype])
                      if peak >= 0 else 99,
        "concentration": round(concentration, 3),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", nargs="*", default=["dqn", "ddqn"],
                    choices=list(MAKERS))
    ap.add_argument("--archetypes", nargs="*", default=list(cane.ARCHETYPES))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--min-sends", type=int, default=1,
                    help="required contacts per 24-hour window")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    seeds = list(range(args.seeds))
    cells = [(k, a, s) for k in args.agents
             for a in args.archetypes for s in seeds]

    print("=" * 84)
    print("Personalisation test: does the agent learn each person's rhythm?")
    print("=" * 84)
    print(f"min contacts/day  : {args.min_sends}   (agent chooses which hour)")
    print(f"cells {len(cells)} | workers {jobs} | episodes {args.episodes}\n",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_cell, k, a, s, args.episodes, args.min_sends):
                (k, a, s) for k, a, s in cells}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i:2d}/{len(cells)}] {r['agent']:5} {r['archetype']:17} "
                  f"sends {r['sends_per_episode']:5.1f}  "
                  f"peak {r['peak_hour']:2d}:00 vs target "
                  f"{r['target_hour']:2d}:00  err {r['hour_error']:2d}h  "
                  f"conc {r['concentration']:.2f}  "
                  f"reward {r['reward_mean']:7.2f}  ctr {r['ctr']:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "personalisation.csv", index=False)

    print()
    print("=" * 84)
    print("CHOSEN SEND HOUR  (target in brackets)")
    for kind in df["agent"].unique():
        d = df[df["agent"] == kind]
        line = "  ".join(
            f"{r['archetype'][:12]}:{r['peak_hour']:02d}[{r['target_hour']:02d}]"
            for _, r in d.iterrows())
        print(f"  {kind:5} {line}")

    print()
    print("MEAN REWARD")
    print(df.pivot_table(index="archetype", columns="agent",
                         values="reward_mean").round(2).to_string())
    print()
    print("HOUR ERROR (0 = found the exact best hour)")
    print(df.pivot_table(index="archetype", columns="agent",
                         values="hour_error").round(1).to_string())

    print()
    good = df[df["hour_error"] <= 2]
    print(f"within 2 hours of the ideal : {len(good)} of {len(df)} cells")
    for kind in df["agent"].unique():
        d = df[df["agent"] == kind]
        distinct = d["peak_hour"].nunique()
        print(f"  {kind:5} picked {distinct} distinct hours across "
              f"{len(d)} people  "
              f"(mean hour error {d['hour_error'].mean():.1f}h)")
    print("\nwrote artifacts/personalisation.csv")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
