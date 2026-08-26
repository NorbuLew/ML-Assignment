"""Fix the collapse: exploration that does not destroy the user it explores on.

The collapse trace shows the cause. Epsilon-greedy samples uniformly over three
actions, two of which are sends, so 68% of exploratory steps send. Fatigue obeys

    F(t+1) = lam*F(t) + kappa[a]

so a 68% send rate drives fatigue to a steady state of

    E[kappa] / (1 - lam) = 0.100 / 0.100 = 1.00

against a churn threshold of 0.70. The user quits in 70-100% of episodes for the
first half of training, every reward the agent sees is around -150, and the value
of *every* action -- including holding -- collapses. The agent then picks the
least-bad option, which is silence.

Nothing about the task requires this. It is an artifact of the exploration
distribution. Three fixes are tested here, all leaving the environment, the
reward function and the network untouched:

* `hold_prior`   -- explore from a non-uniform prior over actions, chosen so
                    the expected steady-state fatigue stays under the churn
                    threshold. This is the minimal, principled change.
* `fatigue_gate` -- explore uniformly, but suppress exploratory *sends* once
                    fatigue passes a ceiling. Safe exploration: the agent may
                    try anything that will not destroy the episode.
* `both`         -- both together.

    python tools/fix_exploration.py
    python tools/fix_exploration.py --archetypes OfficeWorker --jobs 1
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
from cane.agents_deep import DDQNAgent  # noqa: E402
from cane.core import CANE_CONFIG, IDX_FATIGUE, N_ACTIONS  # noqa: E402

OUT = ROOT / "artifacts"

# Chosen so E[kappa] under exploration keeps steady-state fatigue well clear of
# the churn threshold. With kappa = {0, 0.12, 0.18} and lam = 0.9:
#   E[kappa] = 0.13*0.12 + 0.07*0.18 = 0.0282  ->  F* = 0.282  (threshold 0.70)
# It is also roughly the action mix any sane notification policy would use:
# mostly silence, occasional contact.
HOLD_PRIOR = np.array([0.80, 0.13, 0.07])

FATIGUE_CEILING = 0.45


def steady_state_fatigue(prior: np.ndarray) -> float:
    C = CANE_CONFIG
    e_kappa = sum(prior[a] * C["kappa"][a] for a in range(N_ACTIONS))
    return e_kappa / (1.0 - C["lam"])


class SafeExplorationDDQN(DDQNAgent):
    """DDQN whose *exploration* is shaped; its learning rule is unchanged.

    Only `act` differs from the parent. The replay buffer, the target network,
    the Double-DQN target and every hyperparameter are inherited, so any change
    in the result is attributable to the exploration distribution alone.
    """

    def __init__(self, *a, prior=None, fatigue_ceiling=None, **kw):
        super().__init__(*a, **kw)
        self.prior = None if prior is None else np.asarray(prior, dtype=float)
        self.fatigue_ceiling = fatigue_ceiling

    def act(self, state, greedy=False):
        # Greedy behaviour is untouched: evaluation must measure the learned
        # policy, not the exploration scheme wrapped around it.
        if greedy:
            return super().act(state, greedy=True)

        eps = self._epsilon()
        if self.rng.random() >= eps:
            return super().act(state, greedy=True)

        p = (self.prior.copy() if self.prior is not None
             else np.full(N_ACTIONS, 1.0 / N_ACTIONS))

        if self.fatigue_ceiling is not None:
            if float(state[IDX_FATIGUE]) >= self.fatigue_ceiling:
                # Too fatigued to risk an exploratory send: the information a
                # send would buy is not worth the episode it would end.
                p = np.zeros(N_ACTIONS)
                p[0] = 1.0

        p = p / p.sum()
        action = int(self.rng.choice(N_ACTIONS, p=p))
        _, aux = super().act(state, greedy=True)
        return action, aux


VARIANTS = {
    "baseline":     dict(prior=None,       gate=None),
    "hold_prior":   dict(prior=HOLD_PRIOR, gate=None),
    "fatigue_gate": dict(prior=None,       gate=FATIGUE_CEILING),
    "both":         dict(prior=HOLD_PRIOR, gate=FATIGUE_CEILING),
}


def run_cell(variant: str, archetype: str, seed: int, episodes: int) -> dict:
    spec = VARIANTS[variant]
    t0 = time.time()

    if variant == "baseline":
        agent = DDQNAgent(seed=seed)
    else:
        agent = SafeExplorationDDQN(seed=seed, prior=spec["prior"],
                                    fatigue_ceiling=spec["gate"])

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
        "variant": variant, "archetype": archetype, "seed": seed,
        "reward_mean": float(m["reward_mean"]), "ctr": float(m["ctr"]),
        "sends_per_episode": float(m["sends_per_episode"]),
        "optout_rate": float(m["optout_rate"]),
        "train_churn_rate": churn / episodes,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archetypes", nargs="*",
                    default=["OfficeWorker", "NightShiftWorker",
                             "NormalStudent"])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    seeds = list(range(args.seeds))

    print("=" * 78)
    print("Exploration fix: does safe exploration recover the silent archetypes?")
    print("=" * 78)
    print(f"uniform exploration  -> steady-state fatigue "
          f"{steady_state_fatigue(np.full(3, 1 / 3)):.3f}  "
          f"(churn threshold {CANE_CONFIG['churn_threshold']})")
    print(f"hold-prior           -> steady-state fatigue "
          f"{steady_state_fatigue(HOLD_PRIOR):.3f}")
    print(f"fatigue ceiling      -> {FATIGUE_CEILING}")
    print()

    cells = [(v, a, s) for v in args.variants
             for a in args.archetypes for s in seeds]
    print(f"cells {len(cells)} | workers {jobs} | episodes {args.episodes}\n",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_cell, v, a, s, args.episodes): (v, a, s)
                for v, a, s in cells}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i:2d}/{len(cells)}] {r['variant']:13} "
                  f"{r['archetype']:17} s{r['seed']}  "
                  f"sends {r['sends_per_episode']:6.2f}  "
                  f"reward {r['reward_mean']:8.2f}  ctr {r['ctr']:.3f}  "
                  f"train-churn {r['train_churn_rate']:5.0%}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "exploration_fix.csv", index=False)

    print()
    print("=" * 78)
    print("MEAN REWARD")
    print(df.pivot_table(index="archetype", columns="variant",
                         values="reward_mean").round(2).to_string())
    print()
    print("SENDS PER WEEK")
    print(df.pivot_table(index="archetype", columns="variant",
                         values="sends_per_episode").round(2).to_string())
    print()
    print("TRAINING CHURN RATE")
    print(df.pivot_table(index="archetype", columns="variant",
                         values="train_churn_rate").round(3).to_string())
    print()
    base = df[df.variant == "baseline"]["sends_per_episode"].mean()
    best = (df[df.variant != "baseline"]
            .groupby("variant")["reward_mean"].mean().sort_values())
    print(f"baseline sends/week (mean): {base:.2f}")
    print("mean reward by variant:")
    print(best.round(2).to_string())
    print("\nwrote artifacts/exploration_fix.csv")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
