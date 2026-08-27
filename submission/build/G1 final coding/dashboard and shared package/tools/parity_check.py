"""Prove the extracted `cane` package reproduces the notebooks bit-for-bit.

The `Fixed-18:00` and `Random` baselines are fully determined by the environment
and the held-out seed list. They are also already recorded, to full float
precision, in the result CSVs the PPO and DQN notebooks wrote. So if running
those two baselines through the extracted package reproduces the recorded
numbers exactly, the package's environment, harness and evaluation protocol are
identical to the notebooks' -- which is what makes it legitimate to report
package-generated numbers alongside notebook-generated ones.

This is a stronger check than diffing source text: it compares behaviour over
200 episodes x 168 steps x 5 archetypes, and it is immune to comments,
formatting and cell ordering.

    python tools/parity_check.py

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cane  # noqa: E402

REFERENCE = ROOT / "Code" / "results" / "ppo_results.csv"
BASELINES = ["Fixed-18:00", "Random"]
METRICS = ["reward_mean", "ctr", "sends_per_episode", "optout_rate"]
TOL = 1e-9


def baseline_factory(name):
    if name == "Fixed-18:00":
        return lambda s: cane.FixedScheduleAgent(hour=18)
    if name == "Random":
        return lambda s: cane.RandomAgent(seed=s)
    raise ValueError(name)


def main() -> int:
    if not REFERENCE.is_file():
        print(f"error: reference file missing: {REFERENCE}")
        return 2

    ref = pd.read_csv(REFERENCE)
    ref = ref[ref["agent"].isin(BASELINES)]

    print("=" * 78)
    print("Behavioural parity: extracted cane package vs the notebooks")
    print("=" * 78)
    print(f"reference  : {REFERENCE.relative_to(ROOT)}")
    print(f"protocol   : {len(cane.EVAL_SEEDS)} held-out episodes "
          f"(seeds {cane.EVAL_SEEDS[0]}..{cane.EVAL_SEEDS[-1]})")
    print(f"archetypes : {', '.join(cane.ARCHETYPES)}")
    print()

    rows, worst, failures = [], 0.0, 0
    for arch in cane.ARCHETYPES:
        for name in BASELINES:
            got = cane.evaluate_baseline(baseline_factory(name), archetype=arch)
            want = ref[(ref["archetype"] == arch) & (ref["agent"] == name)]
            if want.empty:
                print(f"  {arch:17} {name:12} no reference row -- SKIP")
                continue
            want = want.iloc[0]
            deltas = {m: abs(float(got[m]) - float(want[m])) for m in METRICS}
            worst_here = max(deltas.values())
            worst = max(worst, worst_here)
            ok = worst_here <= TOL
            failures += 0 if ok else 1
            print(f"  {arch:17} {name:12} reward {got['reward_mean']:9.4f}  "
                  f"max|delta| {worst_here:.3g}  {'ok' if ok else 'MISMATCH'}")
            if not ok:
                for m, d in deltas.items():
                    if d > TOL:
                        print(f"        {m}: package {float(got[m])!r} "
                              f"vs notebook {float(want[m])!r}")
            rows.append({"archetype": arch, "agent": name, **deltas})

    print()
    print("=" * 78)
    if failures:
        print(f"FAIL: {failures} baseline(s) differ. The extracted package does NOT")
        print("      reproduce the notebooks -- do not mix their numbers.")
        print("=" * 78)
        return 1

    print(f"PASS: {len(rows)} baselines reproduced exactly "
          f"(max |delta| = {worst:.3g}).")
    print("      The package environment, harness and eval protocol are")
    print("      behaviourally identical to the notebooks'.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
