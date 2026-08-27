"""Convergence speed: how many episodes until each algorithm is done learning?

The proposal names sample efficiency as a comparison metric and no results file
carries it. Final reward says which algorithm ends up best; it says nothing
about which gets there first, and for a notification system that has to be
deployed against real users those are different questions -- an agent that needs
three times the episodes spends three times as long sending badly timed
notifications to real people.

Every `--every` episodes each agent is frozen and evaluated greedily on a fixed
set of held-out episodes, so the curve measures the *learned policy* rather than
the noisy exploring one that generated it.

Convergence is reported as the first evaluation point that reaches 90% of the
agent's own final performance and stays there. Defining it relative to each
agent's own ceiling rather than a shared reward level is deliberate: the
alternative flatters whichever algorithm happens to score highest and says
nothing about speed.

    python tools/learning_curves.py
    python tools/learning_curves.py --agents dqn ddqn --archetypes Housewife
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
from cane.core import LinUCBAgent  # noqa: E402

OUT = ROOT / "artifacts"

MAKERS = {"dqn": DQNAgent, "ddqn": DDQNAgent, "ppo": PPOAgent,
          "linucb": LinUCBAgent}

# Evaluating on all 200 held-out episodes at every checkpoint would cost more
# than the training it is measuring. 40 is enough to rank the curves; the final
# numbers everywhere else in the project still use the full 200.
PROBE_EPISODES = 40
CONVERGENCE_FRACTION = 0.90


def evaluate(agent, env) -> dict:
    m, _, _, _ = cane.run_episodes(agent, env, seeds=cane.EVAL_SEEDS[:PROBE_EPISODES],
                                   learn=False, greedy=True)
    return {"reward": float(m["reward_mean"]), "ctr": float(m["ctr"]),
            "sends": float(m["sends_per_episode"])}


# Each algorithm's own winning configuration, from tools/tune_study.py and
# tools/ppo_fix.py. They differ because the algorithms differ: the deep
# Q-learners take a pre-seeded replay buffer, which PPO cannot use at all, and
# PPO needs a raised entropy coefficient, which the Q-learners have no analogue
# for. LinUCB has no tuned variant -- nothing in the study applies to a bandit.
TUNED_HP = {
    "dqn": dict(lr=3e-4, gamma=0.90, epsilon_end=0.08,
                epsilon_decay_steps=60000, buffer_size=20000),
    "ddqn": dict(lr=3e-4, gamma=0.90, epsilon_end=0.08,
                 epsilon_decay_steps=60000, buffer_size=20000),
    "ppo": dict(gamma=0.90, ent_coef=0.03),
    "linucb": {},
}


def run_cell(kind: str, archetype: str, seed: int, episodes: int,
             every: int, tuned: bool = False) -> list[dict]:
    t0 = time.time()
    agent = MAKERS[kind](seed=seed, **(TUNED_HP[kind] if tuned else {}))

    if tuned and hasattr(agent, "buffer"):
        # Only the replay-based agents can be pre-seeded. Doing it here rather
        # than inside the loop keeps the episode axis meaning the same thing
        # for every curve on the chart.
        from tools.preseed_study import preseed_buffer
        preseed_buffer(agent, cane.CANEEnv(seed=2000 + seed,
                                           archetype=archetype),
                       1000, np.random.default_rng(seed))
    if tuned and kind == "ppo":
        from tools.ppo_fix import HOLD_PRIOR, set_action_prior
        set_action_prior(agent, HOLD_PRIOR)

    train_env = cane.CANEEnv(seed=1000 + seed, archetype=archetype)
    eval_env = cane.CANEEnv(seed=7000 + seed, archetype=archetype)

    rows = []
    for ep in range(1, episodes + 1):
        if hasattr(agent, "reset"):
            agent.reset()
        s, _ = train_env.reset()
        done = False
        while not done:
            a, aux = agent.act(s, greedy=False)
            nxt, r, term, trunc, _ = train_env.step(a)
            agent.update(s, a, r, nxt, term, aux)
            s, done = nxt, term or trunc

        if ep % every == 0 or ep == episodes:
            m = evaluate(agent, eval_env)
            rows.append({"agent": kind.upper(), "archetype": archetype,
                         "seed": seed, "episode": ep, **m})

    for r in rows:
        r["seconds"] = round(time.time() - t0, 1)
    return rows


def convergence_episode(curve: pd.DataFrame) -> float:
    """First episode reaching `CONVERGENCE_FRACTION` of final reward, and holding.

    'And holding' matters. A noisy curve can touch the threshold once early and
    fall back; reporting that as convergence would make the least stable agent
    look like the fastest one.
    """
    curve = curve.sort_values("episode")
    final = float(curve["reward"].iloc[-1])
    if final <= 0:
        # A policy that ends at or below zero has no ceiling to converge to;
        # reporting a number here would invent one.
        return float("nan")
    target = CONVERGENCE_FRACTION * final
    reached = curve["reward"] >= target
    for i in range(len(curve)):
        if reached.iloc[i:].all():
            return float(curve["episode"].iloc[i])
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", nargs="*", default=["linucb", "dqn", "ddqn", "ppo"],
                    choices=list(MAKERS))
    ap.add_argument("--archetypes", nargs="*", default=list(cane.ARCHETYPES))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--every", type=int, default=25,
                    help="episodes between evaluation checkpoints")
    ap.add_argument("--tuned", action="store_true",
                    help="use each algorithm's winning configuration instead "
                         "of its shipped defaults")
    ap.add_argument("--out-suffix", default="",
                    help="appended to the output filenames, so a tuned run "
                         "does not overwrite the default-settings one")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    cells = [(k, a, s) for k in args.agents
             for a in args.archetypes for s in range(args.seeds)]

    print("=" * 88)
    print("Learning curves and convergence speed")
    print("=" * 88)
    print(f"checkpoint every {args.every} episodes, evaluated greedily on "
          f"{PROBE_EPISODES} held-out episodes")
    print(f"converged = first point at {CONVERGENCE_FRACTION:.0%} of final "
          f"reward that never drops back")
    print(f"cells {len(cells)} | workers {jobs} | episodes {args.episodes}\n",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_cell, k, a, s, args.episodes, args.every,
                            args.tuned): (k, a, s) for k, a, s in cells}
        for i, f in enumerate(as_completed(futs), 1):
            got = f.result()
            rows += got
            last = got[-1]
            print(f"  [{i:2d}/{len(cells)}] {last['agent']:6} "
                  f"{last['archetype']:17} final reward "
                  f"{last['reward']:8.2f}  [{last['seconds']:.0f}s]",
                  flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"learning_curves{args.out_suffix}.csv", index=False)

    conv = (df.groupby(["agent", "archetype", "seed"])
              .apply(convergence_episode, include_groups=False)
              .rename("converged_at").reset_index())
    conv.to_csv(OUT / f"convergence{args.out_suffix}.csv", index=False)

    print()
    print("=" * 88)
    print("EPISODES TO CONVERGE  (NaN = never rose above zero, so no ceiling)")
    print(conv.pivot_table(index="archetype", columns="agent",
                           values="converged_at").to_string())
    print()
    print("FINAL REWARD")
    final = df.sort_values("episode").groupby(
        ["agent", "archetype"]).tail(1)
    print(final.pivot_table(index="archetype", columns="agent",
                            values="reward").round(2).to_string())

    print()
    print("SAMPLE EFFICIENCY  (mean episodes to converge, where it converged)")
    eff = conv.groupby("agent")["converged_at"].agg(["mean", "count"])
    print(eff.round(1).to_string())
    print()
    print(f"wrote artifacts/learning_curves{args.out_suffix}.csv and "
          f"artifacts/convergence{args.out_suffix}.csv")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
