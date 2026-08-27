"""RQ3: does ONE policy adapt its pacing to each archetype, with no retraining?

The proposal is specific about this:

    "generalise across users with different daily routines without per-user
     retraining"
    RQ3. Does a single learned policy adapt its pacing to distinct user
         archetypes (e.g. daytime vs night-active routines) without per-user
         retraining?

Every per-archetype experiment in this repo answers a different, easier
question. This one trains a single agent on the mixed population -- where the
archetype is hidden, resampled each episode, and never appears in the state --
and then evaluates that one frozen policy separately on each archetype.

If it picks a different hour for the night-shift worker than for the housewife,
having never been told which is which, that is RQ3 answered yes: the policy is
inferring routine from context. If it picks the same hour for everyone, it has
learned one global schedule and RQ3 is answered no.

The minimum-contact wrapper is applied because an agent that sends nothing
cannot demonstrate adaptation either way -- but the wrapper only removes the
option of silence. Which hour, and which message type, remain entirely the
agent's choice.

    python tools/rq3_single_policy.py
    python tools/rq3_single_policy.py --agents dqn ddqn ppo --seeds 3
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
from tools.personalisation_test import (  # noqa: E402
    TARGET_HOUR, circular_hour_error,
)

OUT = ROOT / "artifacts"
MAKERS = {"dqn": DQNAgent, "ddqn": DDQNAgent, "ppo": PPOAgent}


def evaluate_on(agent, archetype: str, seeds) -> dict:
    """Run the frozen policy on one archetype and record where it sends."""
    counts = np.zeros(24, dtype=int)
    env = cane.CANEEnv(seed=7000, archetype=archetype)
    rewards, sends, clicks, optouts = [], 0, 0, 0

    for sd in seeds:
        if hasattr(agent, "reset"):
            agent.reset()
        s, _ = env.reset(seed=int(sd))
        done, total = False, 0.0
        while not done:
            a, _ = agent.act(s, greedy=True)
            if a != 0:
                counts[int(s[IDX_HOUR])] += 1
                sends += 1
            nxt, r, term, trunc, info = env.step(a)
            total += r
            clicks += int(bool(info.get("clicked", False)))
            s, done = nxt, term or trunc
        optouts += int(term)
        rewards.append(total)

    total_sends = int(counts.sum())
    peak = int(np.argmax(counts)) if total_sends else -1
    if total_sends:
        idx = [(peak - 1) % 24, peak, (peak + 1) % 24]
        conc = float(counts[idx].sum()) / total_sends
    else:
        conc = 0.0

    return {
        "archetype": archetype,
        "reward_mean": float(np.mean(rewards)),
        "ctr": clicks / sends if sends else 0.0,
        "sends_per_episode": sends / len(seeds),
        "optout_rate": optouts / len(seeds),
        "peak_hour": peak,
        "target_hour": TARGET_HOUR[archetype],
        "hour_error": circular_hour_error(peak, TARGET_HOUR[archetype])
                      if peak >= 0 else 99,
        "concentration": round(conc, 3),
        "hist": counts.tolist(),
    }


def run_seed(kind: str, seed: int, episodes: int, min_sends: int) -> list[dict]:
    """Train one agent on the MIXED population, then test it on each archetype."""
    t0 = time.time()
    inner = MAKERS[kind](seed=seed)
    agent = MinContactAgent(inner, min_sends=min_sends, window=24, horizon=168)

    # No `archetype=` argument: the environment resamples a hidden archetype
    # each episode and never exposes the label. This is the setting the proposal
    # describes.
    train_env = cane.CANEEnv(seed=1000 + seed)
    for _ in range(episodes):
        agent.reset()
        s, _ = train_env.reset()
        done = False
        while not done:
            a, aux = agent.act(s, greedy=False)
            nxt, r, term, trunc, _ = train_env.step(a)
            agent.update(s, a, r, nxt, term, aux)
            s, done = nxt, term or trunc

    eval_seeds = cane.EVAL_SEEDS[:60]
    rows = []
    for arch in cane.ARCHETYPES:
        r = evaluate_on(agent, arch, eval_seeds)
        r.update(agent=kind.upper(), seed=seed,
                 seconds=round(time.time() - t0, 1))
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", nargs="*", default=["dqn", "ddqn"],
                    choices=list(MAKERS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--min-sends", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    cells = [(k, s) for k in args.agents for s in range(args.seeds)]

    print("=" * 86)
    print("RQ3: ONE policy, trained on the mixed population, tested per person")
    print("=" * 86)
    print("archetype is hidden, resampled per episode, absent from the state")
    print(f"min contacts/day : {args.min_sends}  (agent chooses which hour)")
    print(f"runs {len(cells)} | workers {jobs} | episodes {args.episodes}\n",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_seed, k, s, args.episodes, args.min_sends):
                (k, s) for k, s in cells}
        for i, f in enumerate(as_completed(futs), 1):
            got = f.result()
            rows += got
            k, s = got[0]["agent"], got[0]["seed"]
            hours = "  ".join(f"{r['archetype'][:11]}:{r['peak_hour']:02d}"
                              f"[{r['target_hour']:02d}]" for r in got)
            print(f"  [{i}/{len(cells)}] {k} seed {s}: {hours}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.drop(columns=["hist"]).to_csv(OUT / "rq3_single_policy.csv", index=False)

    print()
    print("=" * 86)
    print("CHOSEN HOUR PER PERSON, by one policy that was never told who is who")
    print(df.pivot_table(index="archetype", columns=["agent", "seed"],
                         values="peak_hour").to_string())
    print()
    print("MEAN REWARD")
    print(df.pivot_table(index="archetype", columns="agent",
                         values="reward_mean").round(2).to_string())
    print()
    print("CTR")
    print(df.pivot_table(index="archetype", columns="agent",
                         values="ctr").round(3).to_string())

    print()
    print("=" * 86)
    print("RQ3 VERDICT")
    for (kind, seed), d in df.groupby(["agent", "seed"]):
        distinct = d["peak_hour"].nunique()
        err = d["hour_error"].mean()
        conc = d["concentration"].mean()
        verdict = ("ADAPTS - different hours for different people"
                   if distinct >= 3 else
                   "PARTIAL - some variation" if distinct == 2 else
                   "ONE GLOBAL SCHEDULE - same hour for everyone")
        print(f"  {kind} seed {seed}: {distinct} distinct hours across 5 "
              f"people | mean error {err:.1f}h | concentration {conc:.2f}")
        print(f"      -> {verdict}")
    print()
    print("A single policy that picks 3+ distinct hours, without ever seeing an")
    print("archetype label, is RQ3 answered yes.")
    print("wrote artifacts/rq3_single_policy.csv")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
