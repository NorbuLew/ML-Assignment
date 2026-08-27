"""Verify the four per-algorithm result CSVs describe the same experiment.

Each notebook writes its own results file. They are only comparable if every
notebook ran the identical environment and the identical held-out episode
seeds. The `Fixed-18:00` and `Random` baselines are deterministic given that
protocol, so if all four files agree on those rows to floating-point tolerance,
the learners in those files were evaluated on the same episodes and a
like-for-like comparison is sound.

    python tools/check_results.py

Exits non-zero if any file is missing, malformed, or disagrees on a baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "LinUCB": ROOT / "Code" / "results" / "linucb_results.csv",
    "DDQN":   ROOT / "Code" / "results" / "ddqn_results.csv",
    "PPO":    ROOT / "Code" / "results" / "ppo_results.csv",
    "DQN":    ROOT / "DQN" / "results" / "dqn_results.csv",
}

SCHEMA = ["archetype", "agent", "seed", "reward_mean",
          "ctr", "sends_per_episode", "optout_rate"]
BASELINES = ["Fixed-18:00", "Random"]
METRICS = ["reward_mean", "ctr", "sends_per_episode", "optout_rate"]
TOL = 1e-9


def main() -> int:
    frames, missing = {}, []
    for name, path in FILES.items():
        if not path.is_file():
            missing.append((name, path))
            continue
        frames[name] = pd.read_csv(path)

    print("=" * 72)
    print("CANE result files")
    print("=" * 72)
    for name, path in FILES.items():
        if name in frames:
            df = frames[name]
            agents = ", ".join(sorted(df["agent"].unique()))
            print(f"  {name:8} {len(df):3d} rows  {path.relative_to(ROOT)}")
            print(f"           agents: {agents}")
        else:
            print(f"  {name:8} MISSING  {path.relative_to(ROOT)}")

    if missing:
        print(f"\nFAIL: {len(missing)} of {len(FILES)} result files missing.")
        for name, path in missing:
            print(f"  - {name}: run the notebook that writes {path.name}")
        return 1

    # Schema check
    print("\n" + "-" * 72)
    print("Schema")
    print("-" * 72)
    bad_schema = False
    for name, df in frames.items():
        cols = list(df.columns)
        if cols == SCHEMA:
            print(f"  {name:8} ok")
        else:
            bad_schema = True
            print(f"  {name:8} MISMATCH")
            print(f"           expected {SCHEMA}")
            print(f"           got      {cols}")
    if bad_schema:
        print("\nFAIL: schema mismatch; the files cannot be concatenated.")
        return 1

    # Baseline agreement — the actual proof of a shared protocol
    print("\n" + "-" * 72)
    print("Baseline agreement (proof of a shared evaluation protocol)")
    print("-" * 72)
    ref_name = "PPO"
    ref = frames[ref_name]
    ref_base = (ref[ref["agent"].isin(BASELINES)]
                .set_index(["archetype", "agent"])[METRICS]
                .sort_index())

    worst = 0.0
    failed = []
    for name, df in frames.items():
        if name == ref_name:
            continue
        other = (df[df["agent"].isin(BASELINES)]
                 .set_index(["archetype", "agent"])[METRICS]
                 .sort_index())
        if not ref_base.index.equals(other.index):
            failed.append((name, "baseline rows differ in shape/keys"))
            print(f"  {name:8} FAIL  baseline index mismatch")
            continue
        delta = (ref_base - other).abs().to_numpy().max()
        worst = max(worst, float(delta))
        verdict = "ok" if delta <= TOL else "FAIL"
        if delta > TOL:
            failed.append((name, f"max baseline delta {delta:.3g}"))
        print(f"  {name:8} vs {ref_name}: max |delta| = {delta:.3g}  {verdict}")

    print("\n" + "=" * 72)
    if failed:
        print("FAIL: baselines disagree, so the learners did NOT share an")
        print("      evaluation protocol. Do not compare these files directly.")
        for name, why in failed:
            print(f"  - {name}: {why}")
        return 1

    combined = pd.concat(
        [df.assign(source=name) for name, df in frames.items()], ignore_index=True)
    learners = sorted(set(combined["agent"]) - set(BASELINES))
    print(f"PASS: all {len(frames)} files agree on baselines "
          f"(max |delta| = {worst:.3g}).")
    print(f"      {len(combined)} rows total; learners: {', '.join(learners)}")
    print("      A four-way comparison over these files is like-for-like.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
