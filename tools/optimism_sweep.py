"""Does a stronger optimistic bias rescue more archetypes?

The optimistic-send initialisation rescues Housewife and no one else. If the
mechanism is signal strength -- how much of the day is profitable, and whether
those hours are contiguous -- then raising the bias should rescue archetypes in
descending order of profitable mass, and should keep failing on the weakest.

That is a falsifiable prediction, so this sweeps the bias and records where each
archetype flips from silent to sending.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cane  # noqa: E402
from cane.exploration_study import make_agent  # noqa: E402

OUT = ROOT / "artifacts"


def run_cell(archetype: str, bias: float, seed: int, episodes: int) -> dict:
    spec = {"overrides": {}, "optimistic": bias, "warm_start": 0,
            "note": f"optimistic bias {bias}"}
    t0 = time.time()
    agent = make_agent(seed, spec, archetype)
    env = cane.CANEEnv(seed=1000 + seed, archetype=archetype)
    cane.run_episodes(agent, env, n_episodes=episodes, learn=True, greedy=False)
    eval_env = cane.CANEEnv(seed=7000 + seed, archetype=archetype)
    m, _, _, _ = cane.run_episodes(agent, eval_env, seeds=cane.EVAL_SEEDS,
                                   learn=False, greedy=True)
    return {"archetype": archetype, "bias": bias, "seed": seed,
            "reward_mean": float(m["reward_mean"]), "ctr": float(m["ctr"]),
            "sends_per_episode": float(m["sends_per_episode"]),
            "optout_rate": float(m["optout_rate"]),
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--biases", type=float, nargs="*",
                    default=[5.0, 10.0, 20.0])
    ap.add_argument("--archetypes", nargs="*", default=list(cane.ARCHETYPES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    cells = [(a, b) for b in args.biases for a in args.archetypes]

    print(f"cells {len(cells)} | workers {jobs} | episodes {args.episodes}",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_cell, a, b, args.seed, args.episodes): (a, b)
                for a, b in cells}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i:2d}/{len(cells)}] bias {r['bias']:5.1f}  "
                  f"{r['archetype']:17} sends {r['sends_per_episode']:6.2f}  "
                  f"reward {r['reward_mean']:8.2f}  ctr {r['ctr']:.3f}",
                  flush=True)

    df = pd.DataFrame(rows).sort_values(["archetype", "bias"])
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "optimism_sweep.csv", index=False)
    print()
    print(df.pivot_table(index="archetype", columns="bias",
                         values="sends_per_episode").round(2).to_string())
    print(f"\nwrote artifacts/optimism_sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
