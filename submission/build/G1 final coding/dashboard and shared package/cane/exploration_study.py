"""Test the diagnosis: is the never-send collapse an exploration failure?

The finding this exists to check
-------------------------------
Three of the four learned agents converge to sending nothing at all, scoring
exactly 0.00. A closed-form break-even analysis says that cannot be optimal: a
send pays whenever the click probability clears

    p >= (W_send + W_fat * kappa / (1 - lam)) / R_click  =  0.34  for an Engage

and every archetype clears it for 3-8 hours of the 24. A fixed 18:00 schedule --
which learns nothing at all -- therefore beats every learned agent on
OfficeWorker (+2.01 against 0.00).

So the agents are not under-trained; five independent seeds agree, and epsilon
has already annealed to 0.05. The hypothesis is that they are *under-explored*:
every exploratory send raises fatigue immediately and risks a terminal opt-out,
while the payoff is delayed and uncertain. An agent that has not stumbled onto
the few good hours before epsilon decays will never find them.

The prediction
--------------
If the diagnosis is right, an agent that keeps exploring for longer -- or that
starts out believing sends are worth trying -- should find those hours and beat
both the never-send floor of 0.00 and the Fixed-18:00 baseline.

If it is wrong, the variants will also collapse to 0.00, and the honest
conclusion becomes that the reward weights, not the exploration schedule, make
the task unsolvable as specified.

Either outcome is a result. Run:

    python -m cane.exploration_study --quick
    python -m cane.exploration_study
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
import torch

import cane
from cane.agents_deep import DQNAgent
from cane.core import ENGAGE, INCENTIVE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts"

# The three archetypes on which every learner collapsed to silence. Testing on
# these is the point: the two where DQN already succeeds would not discriminate.
COLLAPSED = ["OfficeWorker", "NightShiftWorker", "NormalStudent"]

# Each variant changes exactly one thing against the shipped configuration, so
# any improvement is attributable.
VARIANTS = {
    "baseline": dict(
        overrides={}, optimistic=0.0, warm_start=0,
        note="shipped config: epsilon 1.0 -> 0.05 over 50k steps"),
    "slow_decay": dict(
        overrides=dict(epsilon_decay_steps=150_000), optimistic=0.0, warm_start=0,
        note="explore for 3x longer before committing"),
    "high_floor": dict(
        overrides=dict(epsilon_end=0.20), optimistic=0.0, warm_start=0,
        note="never stop exploring: epsilon floor raised 0.05 -> 0.20"),
    "optimistic_send": dict(
        overrides={}, optimistic=5.0, warm_start=0,
        note="Q-head bias +5 on the two send actions only, so ties break "
             "towards trying a send rather than towards silence"),
    "warm_start": dict(
        overrides={}, optimistic=0.0, warm_start=40,
        note="replay buffer seeded with 40 episodes of Fixed-18:00 behaviour "
             "before training, so the agent has seen a competent sending "
             "policy rather than only random ones"),
}


def seed_buffer_with_demonstrations(agent, archetype, n_episodes, seed=0):
    """Pre-fill the replay buffer with Fixed-18:00 trajectories.

    This targets a sharper version of the diagnosis. Undirected exploration in
    this environment is not merely slow, it is *actively misleading*: a random
    sender fires ~30 notifications a week, saturates fatigue, and opts the user
    out in 99.5% of episodes for a reward around -121. So the transitions an
    epsilon-greedy agent collects genuinely do show that sending is
    catastrophic, and it learns that lesson correctly. What it never sees is a
    *selective* sending policy.

    Seeding the buffer with a competent demonstrator supplies exactly those
    missing transitions. This is the same idea as learning from demonstrations
    (Hester et al., 2018), reduced to its simplest form.
    """
    demo = cane.FixedScheduleAgent(hour=18)
    env = cane.CANEEnv(seed=5000 + seed, archetype=archetype)
    for ep in range(n_episodes):
        state, _ = env.reset(seed=940_000 + seed * 1000 + ep)
        done = False
        while not done:
            action, _ = demo.act(state, greedy=True)
            nxt, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.push(state[:agent.d_in], action, reward,
                              nxt[:agent.d_in], term)
            state = nxt
    return agent


def make_agent(seed: int, spec: dict, archetype: str) -> DQNAgent:
    """Build a DQN configured for one variant.

    Optimistic initialisation is applied to the *send* actions only. A uniform
    optimistic bias does not work here: it lifts all three action values
    equally, argmax breaks the tie towards index 0, and index 0 is Hold -- so
    the agent still never tries a send. Biasing only the send actions is what
    makes the optimism actually change behaviour.
    """
    agent = DQNAgent(seed=seed, **spec["overrides"])
    if spec["optimistic"]:
        with torch.no_grad():
            for net in (agent.online_net, agent.target_net):
                head = [m for m in net.net if isinstance(m, torch.nn.Linear)][-1]
                head.bias.fill_(0.0)
                head.bias[ENGAGE] = float(spec["optimistic"])
                head.bias[INCENTIVE] = float(spec["optimistic"])
    if spec["warm_start"]:
        seed_buffer_with_demonstrations(agent, archetype,
                                        spec["warm_start"], seed=seed)
    return agent


def run_one(name: str, spec: dict, archetype: str, seed: int,
            train_episodes: int, eval_seeds: list[int]) -> dict:
    """Train and evaluate a single (variant, archetype, seed) cell.

    Module-level and taking only plain arguments so a ProcessPoolExecutor can
    pickle it. Every cell is independent -- separate agent, separate training
    env, separate evaluation env -- which is what makes the study parallelisable
    at all. The result dict is the row that lands in the CSV.
    """
    t0 = time.time()
    agent = make_agent(seed, spec, archetype)
    train_env = cane.CANEEnv(seed=1000 + seed, archetype=archetype)
    cane.run_episodes(agent, train_env, n_episodes=train_episodes,
                      learn=True, greedy=False)
    eval_env = cane.CANEEnv(seed=7000 + seed, archetype=archetype)
    m, _, _, _ = cane.run_episodes(agent, eval_env, seeds=eval_seeds,
                                   learn=False, greedy=True)
    return {
        "variant": name, "archetype": archetype, "seed": seed,
        "reward_mean": float(m["reward_mean"]),
        "ctr": float(m["ctr"]),
        "sends_per_episode": float(m["sends_per_episode"]),
        "optout_rate": float(m["optout_rate"]),
        "train_seconds": round(time.time() - t0, 1),
        "note": spec["note"],
    }


def run_variant(name: str, spec: dict, archetype: str, seeds: list[int],
                train_episodes: int, eval_seeds: list[int]) -> list[dict]:
    """Serial path, kept so the study still runs with --jobs 1."""
    rows = []
    for seed in seeds:
        row = run_one(name, spec, archetype, seed, train_episodes, eval_seeds)
        rows.append(row)
        print(f"    {name:11} {archetype:17} seed {seed}  "
              f"reward {row['reward_mean']:8.2f}  "
              f"sends {row['sends_per_episode']:6.2f}  "
              f"ctr {row['ctr']:.3f}  [{row['train_seconds']:.0f}s]", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, 150 train episodes, 50 eval episodes")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--archetypes", nargs="*", default=COLLAPSED)
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = one per core, minus two left "
                         "for the machine. 1 forces the serial path.")
    args = ap.parse_args()

    # Every cell is single-threaded by construction (cane pins torch to one
    # thread), so the study scales with processes rather than threads. Leaving
    # two cores free keeps the machine usable while this runs for an hour.
    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)

    seeds = [0] if args.quick else list(range(args.seeds))
    train_episodes = 150 if args.quick else cane.TRAIN_EPISODES
    eval_seeds = cane.EVAL_SEEDS[:50] if args.quick else cane.EVAL_SEEDS

    print("=" * 78)
    print("Exploration study - testing the never-send diagnosis"
          + ("  [QUICK]" if args.quick else ""))
    print("=" * 78)
    print(f"archetypes : {', '.join(args.archetypes)}")
    print(f"variants   : {', '.join(VARIANTS)}")
    print(f"seeds      : {seeds} | train {train_episodes} eps | "
          f"eval {len(eval_seeds)} eps")
    total = len(args.archetypes) * len(VARIANTS) * len(seeds)
    print(f"runs       : {total}")
    print(f"workers    : {jobs}")
    print()

    rows, t0 = [], time.time()

    # The reference every variant has to beat: a fixed 18:00 schedule, which
    # learns nothing yet outperforms every collapsed agent. Cheap, so it stays
    # on the main process.
    for archetype in args.archetypes:
        fixed = cane.evaluate_baseline(
            lambda s: cane.FixedScheduleAgent(hour=18), archetype=archetype)
        print(f"    {'Fixed-18:00':11} {archetype:17} (baseline) "
              f"reward {fixed['reward_mean']:8.2f}  "
              f"sends {fixed['sends_per_episode']:6.2f}  ctr {fixed['ctr']:.3f}",
              flush=True)
        rows.append({"variant": "Fixed-18:00", "archetype": archetype, "seed": 0,
                     "reward_mean": float(fixed["reward_mean"]),
                     "ctr": float(fixed["ctr"]),
                     "sends_per_episode": float(fixed["sends_per_episode"]),
                     "optout_rate": float(fixed["optout_rate"]),
                     "train_seconds": 0.0,
                     "note": "non-learning reference"})
    print(flush=True)

    cells = [(name, spec, archetype, seed)
             for archetype in args.archetypes
             for name, spec in VARIANTS.items()
             for seed in seeds]

    if jobs == 1:
        for archetype in args.archetypes:
            print(f"--- {archetype} ---", flush=True)
            for name, spec in VARIANTS.items():
                rows += run_variant(name, spec, archetype, seeds,
                                    train_episodes, eval_seeds)
            print(flush=True)
    else:
        # Results come back out of order, so each line says what it is rather
        # than relying on position. The CSV is sorted afterwards.
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(run_one, name, spec, archetype, seed,
                            train_episodes, eval_seeds): (name, archetype, seed)
                for name, spec, archetype, seed in cells
            }
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                rows.append(row)
                print(f"  [{i:2d}/{len(cells)}] {row['variant']:11} "
                      f"{row['archetype']:17} seed {row['seed']}  "
                      f"reward {row['reward_mean']:8.2f}  "
                      f"sends {row['sends_per_episode']:6.2f}  "
                      f"ctr {row['ctr']:.3f}  [{row['train_seconds']:.0f}s]",
                      flush=True)
        print(flush=True)

    df = pd.DataFrame(rows)
    # Parallel completion order is nondeterministic; sort so the CSV is stable
    # between runs and diffable.
    df = df.sort_values(["archetype", "variant", "seed"]).reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "exploration_study.csv", index=False)

    # Verdict
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    learners = df[df["variant"] != "Fixed-18:00"]
    pivot = (learners.groupby(["archetype", "variant"])["reward_mean"]
             .mean().unstack())
    print(pivot.round(2).to_string())
    print()

    verdict = []
    for archetype in args.archetypes:
        ref = float(df[(df["archetype"] == archetype)
                       & (df["variant"] == "Fixed-18:00")]["reward_mean"].iloc[0])
        sub = learners[learners["archetype"] == archetype]
        best_name = sub.groupby("variant")["reward_mean"].mean().idxmax()
        best_val = float(sub.groupby("variant")["reward_mean"].mean().max())
        beats_zero = best_val > 0.01
        beats_fixed = best_val > ref
        verdict.append({"archetype": archetype, "best_variant": best_name,
                        "best_reward": round(best_val, 3),
                        "fixed_reward": round(ref, 3),
                        "beats_never_send": bool(beats_zero),
                        "beats_fixed_schedule": bool(beats_fixed)})
        flag = ("BEATS both" if beats_fixed else
                "beats silence only" if beats_zero else "still collapsed")
        print(f"  {archetype:17} best={best_name:11} {best_val:8.2f}  "
              f"vs Fixed {ref:7.2f}   -> {flag}")

    (OUT / "exploration_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf8")

    n_fixed = sum(v["beats_never_send"] for v in verdict)
    print()
    if n_fixed == 0:
        print("  Diagnosis NOT supported: every variant still collapses to")
        print("  silence. The barrier is the reward weights, not exploration.")
    elif n_fixed == len(verdict):
        print("  Diagnosis SUPPORTED on every archetype tested: correcting the")
        print("  exploration schedule alone recovers a sending policy.")
    else:
        print(f"  Diagnosis PARTIALLY supported: {n_fixed} of {len(verdict)} "
              f"archetypes recovered.")
    print(f"\nwrote artifacts/exploration_study.csv ({len(df)} rows)")
    print(f"wrote artifacts/exploration_verdict.json")
    print(f"elapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
