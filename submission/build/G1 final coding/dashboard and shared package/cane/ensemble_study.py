"""Fit and evaluate the CANE ensembles, per archetype.

Why per archetype
-----------------
Measured on the mixed population, the reward-maximising policy is to send
nothing at all: averaged over five archetypes with different receptive windows,
no single hour clears the break-even click rate, so silence wins and every
combiner collapses onto it. The deep agents found that optimum; LinUCB did not.

Conditioned on an archetype the picture changes completely -- Fixed-18:00 earns
+2.01 on OfficeWorker and DQN reaches +44.5 on Housewife -- so this study fits
and reports one ensemble per archetype. Fitting on the mixed pool would average
away the very signal the ensemble is supposed to exploit.

Protocol
--------
Weights, gates and the hold bias are fitted on VAL_SEEDS (920_000-920_099).
Every reported number comes from EVAL_SEEDS (900_000-900_199). The two are
disjoint, so no ensemble is ever scored on an episode it was tuned on.

    python -m cane.ensemble_study --quick
    python -m cane.ensemble_study
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import cane
from cane.ensemble import (
    VAL_SEEDS, GatedEnsemble, MajorityVoteEnsemble,
    fit_gates, fit_weights, sweep_hold_bias,
)
from cane.persistence import find_checkpoints, load_agent

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts"

VAL_ENV_SEED = 8100      # disjoint from the training (1000+) and eval (7000+) envs
EVAL_ENV_SEED = 7000


def build_members(seed: int = 0, verbose: bool = True):
    """Load one trained agent per algorithm family, plus the fixed baseline.

    The baseline is deliberately a member. It is the only one of the five that
    reliably sends at a sensible hour, and with three never-senders in the pool
    it is often the only source of any send at all -- excluding it would make the
    ensemble's degeneracy a foregone conclusion rather than a finding.
    """
    checkpoints = find_checkpoints(ROOT)
    members, labels = [], []

    for family in ("LinUCB", "DQN", "DDQN", "PPO"):
        paths = [p for p in checkpoints.get(family, []) if p.stem.endswith(str(seed))]
        if not paths:
            if verbose:
                print(f"  {family:7} no seed-{seed} checkpoint -- skipped")
            continue
        members.append(load_agent(paths[0]))
        labels.append(family)
        if verbose:
            print(f"  {family:7} loaded {paths[0].name}")

    members.append(cane.FixedScheduleAgent(hour=18))
    labels.append("Fixed-18:00")
    if verbose:
        print(f"  {'Fixed':7} 18:00 daily schedule (baseline member)")

    return members, labels


def evaluate(agent, archetype, seeds, env_seed):
    env = cane.CANEEnv(seed=env_seed, archetype=archetype)
    m, _, _, _ = cane.run_episodes(agent, env, seeds=seeds, learn=False, greedy=True)
    return {"reward_mean": float(m["reward_mean"]),
            "ctr": float(m["ctr"]),
            "sends_per_episode": float(m["sends_per_episode"]),
            "optout_rate": float(m["optout_rate"])}


def study_archetype(archetype, members, labels, val_seeds, eval_seeds, biases):
    """Fit on the validation split, then report on the held-out test split."""
    rows = []

    # 1. Each member alone, on both splits. The validation rewards become the
    #    mixture weights; the test rewards are the bar the ensemble must clear.
    val_rewards = []
    for agent, name in zip(members, labels):
        v = evaluate(agent, archetype, val_seeds, VAL_ENV_SEED)
        t = evaluate(agent, archetype, eval_seeds, EVAL_ENV_SEED)
        val_rewards.append(v["reward_mean"])
        rows.append({"archetype": archetype, "scheme": "member", "member": name,
                     "split": "validation", "hold_bias": np.nan, **v})
        rows.append({"archetype": archetype, "scheme": "member", "member": name,
                     "split": "test", "hold_bias": np.nan, **t})

    weights = fit_weights(val_rewards)
    gates = fit_gates(members, archetype=archetype, seeds=val_seeds[:20],
                      env_seed=VAL_ENV_SEED)

    # 2. Majority vote -- the negative control.
    vote = MajorityVoteEnsemble(members, labels=labels)
    v = evaluate(vote, archetype, val_seeds, VAL_ENV_SEED)
    disagreement = vote.disagreement_rate
    t = evaluate(vote, archetype, eval_seeds, EVAL_ENV_SEED)
    rows.append({"archetype": archetype, "scheme": "vote", "member": "ENSEMBLE",
                 "split": "validation", "hold_bias": np.nan,
                 "disagreement_rate": disagreement, **v})
    rows.append({"archetype": archetype, "scheme": "vote", "member": "ENSEMBLE",
                 "split": "test", "hold_bias": np.nan,
                 "disagreement_rate": vote.disagreement_rate, **t})

    # 3. Gated ensemble: line-search the hold bias on validation only.
    sweep = sweep_hold_bias(members, weights, gates, biases=biases,
                            archetype=archetype, seeds=val_seeds,
                            env_seed=VAL_ENV_SEED, labels=labels)
    best = max(sweep, key=lambda r: r["reward_mean"])

    gated = GatedEnsemble(members, weights=weights, gates=gates,
                          hold_bias=best["hold_bias"], labels=labels)
    t = evaluate(gated, archetype, eval_seeds, EVAL_ENV_SEED)
    rows.append({"archetype": archetype, "scheme": "gated", "member": "ENSEMBLE",
                 "split": "validation", "hold_bias": best["hold_bias"],
                 **{k: best[k] for k in
                    ("reward_mean", "ctr", "sends_per_episode", "optout_rate")}})
    rows.append({"archetype": archetype, "scheme": "gated", "member": "ENSEMBLE",
                 "split": "test", "hold_bias": best["hold_bias"], **t})

    meta = {"archetype": archetype,
            "weights": {l: float(w) for l, w in zip(labels, weights)},
            "gates": {l: float(g) for l, g in zip(labels, gates)},
            "best_hold_bias": best["hold_bias"],
            "vote_disagreement_rate": float(vote.disagreement_rate)}
    sweep_rows = [{"archetype": archetype, **r} for r in sweep]
    return rows, sweep_rows, meta


def _study_one(archetype: str, seed: int, val_seeds, eval_seeds, biases):
    """Run one archetype in a worker process.

    The members are rebuilt here rather than passed in: torch modules do not
    survive pickling cleanly, and `build_members` is deterministic given the
    seed, so every worker reconstructs an identical pool from the checkpoints.
    """
    members, labels = build_members(seed=seed, verbose=False)
    return study_archetype(archetype, members, labels, val_seeds, eval_seeds,
                           biases)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="20 validation / 40 test episodes and 9 bias values")
    ap.add_argument("--seed", type=int, default=0,
                    help="which trained seed to take each member from")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = one per archetype (capped by "
                         "cores). 1 forces the serial path.")
    args = ap.parse_args()

    val_seeds = VAL_SEEDS[:20] if args.quick else VAL_SEEDS
    eval_seeds = cane.EVAL_SEEDS[:40] if args.quick else cane.EVAL_SEEDS
    biases = (np.linspace(-2, 2, 9) if args.quick else np.linspace(-3, 2, 21))

    print("=" * 78)
    print("CANE ensemble study" + ("  [QUICK]" if args.quick else ""))
    print("=" * 78)
    print(f"validation : {len(val_seeds)} episodes (seeds {val_seeds[0]}..{val_seeds[-1]})")
    print(f"test       : {len(eval_seeds)} episodes (seeds {eval_seeds[0]}..{eval_seeds[-1]})")
    print(f"hold bias  : {len(biases)} values in [{biases[0]:.1f}, {biases[-1]:.1f}]")
    assert not (set(val_seeds) & set(eval_seeds)), "validation and test overlap"
    print("\nmembers:")
    members, labels = build_members(seed=args.seed)
    if len(members) < 3:
        print("\nerror: need at least 3 members; run the notebooks to produce "
              "the missing checkpoints (see README.md)")
        return 1

    # The archetypes are independent studies -- separate fits, separate
    # sweeps, separate reports -- so they parallelise cleanly. Each is
    # single-threaded (cane pins torch to one thread), so this scales with
    # processes rather than threads.
    jobs = args.jobs or min(len(cane.ARCHETYPES),
                            max(1, (os.cpu_count() or 4) - 2))
    print(f"workers    : {jobs}")

    all_rows, all_sweeps, metas = [], [], []
    t0 = time.time()

    def report(arch, rows, meta):
        print(f"\n--- {arch} ---")
        test = [r for r in rows if r["split"] == "test"]
        for r in sorted(test, key=lambda r: -r["reward_mean"]):
            tag = (r["member"] if r["scheme"] == "member"
                   else f"ENSEMBLE({r['scheme']})")
            print(f"  {tag:22} reward {r['reward_mean']:9.3f}  "
                  f"sends {r['sends_per_episode']:6.2f}  ctr {r['ctr']:.3f}")
        print(f"  best hold_bias = {meta['best_hold_bias']:+.2f}  "
              f"| vote disagreement = {meta['vote_disagreement_rate']:.1%}",
              flush=True)

    if jobs == 1:
        for arch in cane.ARCHETYPES:
            rows, sweep, meta = study_archetype(
                arch, members, labels, val_seeds, eval_seeds, biases)
            all_rows += rows
            all_sweeps += sweep
            metas.append(meta)
            report(arch, rows, meta)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_study_one, arch, args.seed, val_seeds,
                                   eval_seeds, biases): arch
                       for arch in cane.ARCHETYPES}
            for fut in as_completed(futures):
                arch = futures[fut]
                rows, sweep, meta = fut.result()
                all_rows += rows
                all_sweeps += sweep
                metas.append(meta)
                report(arch, rows, meta)
        # Completion order is nondeterministic, so restore the canonical
        # archetype order -- the CSVs and the JSON should be stable between
        # runs and diffable.
        order = {a: i for i, a in enumerate(cane.ARCHETYPES)}
        metas.sort(key=lambda m: order.get(m["archetype"], 99))
        all_rows.sort(key=lambda r: order.get(r["archetype"], 99))
        all_sweeps.sort(key=lambda r: order.get(r["archetype"], 99))

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT / "ensemble.csv", index=False)
    pd.DataFrame(all_sweeps).to_csv(OUT / "ensemble_bias_sweep.csv", index=False)
    (OUT / "ensemble_meta.json").write_text(
        json.dumps(metas, indent=2), encoding="utf8")

    print(f"\n{'=' * 78}")
    print(f"wrote artifacts/ensemble.csv           ({len(all_rows)} rows)")
    print(f"wrote artifacts/ensemble_bias_sweep.csv ({len(all_sweeps)} rows)")
    print(f"wrote artifacts/ensemble_meta.json")
    print(f"elapsed {time.time() - t0:.0f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
