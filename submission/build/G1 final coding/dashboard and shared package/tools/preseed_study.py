"""Does a pre-seeded replay buffer break the silence? (teammate's intervention)

Three of the five archetypes receive nothing from every deep agent. Three
interventions were tested against that already -- an optimistic bias on the send
heads, a non-uniform exploration prior, and a fatigue ceiling on exploratory
sends -- and only the first moved the behaviour, and only on one archetype.

A fourth arrived from the DQN notebook (commit 6f3543f) and is tested here. It
attacks a different link in the chain. The diagnosis was that every Q-value
collapses because early training is dominated by churned episodes and losing
sends, so the network never sees evidence that a send can pay. Rather than
changing how the agent explores, this pre-fills the replay buffer with sends
placed in plausible peak-activity windows *before* learning starts, so the very
first gradient steps are computed against transitions that include profitable
clicks.

It comes with a hyperparameter change in the same commit, most importantly
`gamma` 0.99 -> 0.90. That matters on its own: the discounted fatigue debt of a
send is roughly `W_fat * kappa / (1 - gamma*lam)`, so lowering gamma makes a
send cheaper in the agent's own accounting, independent of what is in the
buffer. The two are therefore tested separately as well as together -- a
combined result that worked would not say which half did the work.

    python tools/preseed_study.py
    python tools/preseed_study.py --agents ddqn --archetypes OfficeWorker
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
from cane.core import ENGAGE, HOLD, IDX_HOUR, N_ACTIONS  # noqa: E402

OUT = ROOT / "artifacts"
MAKERS = {"dqn": DQNAgent, "ddqn": DDQNAgent, "ppo": PPOAgent}

# Generic peak-activity windows: morning commute, lunch, evening. Deliberately
# NOT each archetype's true best hour, which exhaustive search has already found
# and stored in artifacts/best_fixed_policy.csv. Seeding with the answer would
# guarantee a good peak_hour and make every personalisation number meaningless;
# the point is whether the agent can find its own person's hour from a generic
# hint.
PEAK_HOURS = (7, 8, 12, 13, 18, 19, 20)
PEAK_SEND_PROB = 0.4
HOLD_PROB = 0.7

# The hyperparameters shipped alongside the pre-seeding in the same commit.
TEAMMATE_HP = dict(lr=3e-4, gamma=0.90, epsilon_end=0.08,
                   epsilon_decay_steps=60000, buffer_size=20000)

VARIANTS = {
    "baseline":  dict(preseed=False, hp={}),
    "preseed":   dict(preseed=True,  hp={}),
    "gamma090":  dict(preseed=False, hp=dict(gamma=0.90)),
    "both":      dict(preseed=True,  hp=TEAMMATE_HP),
}


def preseed_buffer(agent, env, steps: int, rng: np.random.Generator) -> int:
    """Fill the replay buffer with peak-hour sends before learning starts.

    Returns the number of sends written, so the caller can report how much
    positive evidence the network actually began with rather than assuming it.
    """
    s, _ = env.reset()
    sends = 0
    for _ in range(steps):
        hour = int(s[IDX_HOUR])
        if hour in PEAK_HOURS and rng.random() < PEAK_SEND_PROB:
            a = ENGAGE
        else:
            a = HOLD if rng.random() < HOLD_PROB else int(rng.integers(N_ACTIONS))
        sends += int(a != HOLD)
        ns, r, term, trunc, _ = env.step(a)
        agent.buffer.push(agent._features(s), a, r, agent._features(ns),
                          float(term))
        s = env.reset()[0] if (term or trunc) else ns
    return sends


def run_cell(kind: str, variant: str, archetype: str, seed: int,
             episodes: int, steps: int) -> dict:
    spec = VARIANTS[variant]
    t0 = time.time()

    agent = MAKERS[kind](seed=seed, **spec["hp"])

    seeded = 0
    if spec["preseed"]:
        if not hasattr(agent, "buffer"):
            # PPO is on-policy and keeps no replay buffer, so there is nothing
            # to pre-seed. Its row still runs, carrying only the gamma change,
            # and is labelled below so the table does not imply otherwise.
            pass
        else:
            seed_env = cane.CANEEnv(seed=2000 + seed, archetype=archetype)
            seeded = preseed_buffer(agent, seed_env, steps,
                                    np.random.default_rng(seed))

    train_env = cane.CANEEnv(seed=1000 + seed, archetype=archetype)
    churn = 0
    for _ in range(episodes):
        s, _ = train_env.reset()
        done = False
        while not done:
            a, aux = agent.act(s, greedy=False)
            nxt, r, term, trunc, _ = train_env.step(a)
            agent.update(s, a, r, nxt, term, aux)
            s, done = nxt, term or trunc
        churn += int(term)

    eval_env = cane.CANEEnv(seed=7000 + seed, archetype=archetype)
    m, _, _, _ = cane.run_episodes(agent, eval_env, seeds=cane.EVAL_SEEDS,
                                   learn=False, greedy=True)

    return {
        "agent": kind.upper(), "variant": variant, "archetype": archetype,
        "seed": seed,
        "reward_mean": float(m["reward_mean"]), "ctr": float(m["ctr"]),
        "sends_per_episode": float(m["sends_per_episode"]),
        "optout_rate": float(m["optout_rate"]),
        "train_churn_rate": churn / episodes,
        "preseeded_sends": seeded,
        "applicable": bool(hasattr(agent, "buffer") or not spec["preseed"]),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", nargs="*", default=["dqn", "ddqn"],
                    choices=list(MAKERS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--archetypes", nargs="*", default=list(cane.ARCHETYPES))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--steps", type=int, default=1000,
                    help="pre-seeded transitions written before training")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    cells = [(k, v, a, s) for k in args.agents for v in args.variants
             for a in args.archetypes for s in range(args.seeds)]

    print("=" * 92)
    print("Pre-seeded replay buffer: does starting with profitable sends "
          "break the silence?")
    print("=" * 92)
    print(f"peak windows : {PEAK_HOURS}  (generic, NOT each person's true "
          f"best hour)")
    print(f"pre-seeded   : {args.steps} transitions before the first gradient "
          f"step")
    print(f"teammate hp  : {TEAMMATE_HP}")
    print(f"cells {len(cells)} | workers {jobs} | episodes {args.episodes}\n",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_cell, k, v, a, s, args.episodes, args.steps):
                (k, v, a, s) for k, v, a, s in cells}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i:2d}/{len(cells)}] {r['agent']:5} {r['variant']:9} "
                  f"{r['archetype']:17} sends {r['sends_per_episode']:6.2f}  "
                  f"reward {r['reward_mean']:8.2f}  ctr {r['ctr']:.3f}  "
                  f"churn {r['train_churn_rate']:4.0%}  "
                  f"[{r['seconds']:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "preseed_study.csv", index=False)

    print()
    print("=" * 92)
    print("SENDS PER WEEK  (0.00 = still silent)")
    print(df.pivot_table(index=["agent", "archetype"], columns="variant",
                         values="sends_per_episode").round(2).to_string())
    print()
    print("MEAN REWARD")
    print(df.pivot_table(index=["agent", "archetype"], columns="variant",
                         values="reward_mean").round(2).to_string())

    print()
    print("=" * 92)
    print("VERDICT  -- archetypes rescued from silence, by variant")
    base = df[df.variant == "baseline"].set_index(["agent", "archetype"])
    for variant in [v for v in args.variants if v != "baseline"]:
        d = df[df.variant == variant].set_index(["agent", "archetype"])
        rescued = [f"{a}/{ar}" for (a, ar), r in d.iterrows()
                   if r["sends_per_episode"] > 0.5
                   and base.loc[(a, ar), "sends_per_episode"] <= 0.5]
        gain = d["reward_mean"].mean() - base["reward_mean"].mean()
        print(f"  {variant:9} rescued {len(rescued):2d} cells  "
              f"mean reward {gain:+7.2f} vs baseline")
        if rescued:
            print(f"      {', '.join(rescued)}")
    print()
    print("A variant that rescues cells only in 'both' and not in 'preseed' or")
    print("'gamma090' alone means the two halves are only effective together.")
    print("\nwrote artifacts/preseed_study.csv")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
