"""The same rescue attempt, adapted for PPO.

PPO stays almost silent on every archetype (0.5-1.1 sends a week, reward -1.98
to +0.28). The intervention that arrived from the DQN notebook -- pre-filling
the replay buffer with profitable sends -- cannot be applied to it directly:
PPO is on-policy and keeps no replay buffer, only a rollout it discards after
each update. So the mechanism has to be translated rather than copied.

Two things are being translated, matching the two halves of that commit:

* `gamma` 0.99 -> 0.90 transfers unchanged. It is not a replay trick at all; it
  shrinks the discounted fatigue debt of a send, roughly `W_fat*kappa/(1-gamma*lam)`,
  which makes sending cheaper in the agent's own accounting whatever algorithm
  is doing the accounting. In the DQN/DDQN study this is the half that carried
  the result.

* Pre-seeding translates into two on-policy analogues, tested separately:

  `hold_init`  -- the *structural* analogue. PPO's actor head starts near zero,
                  so the initial policy is close to uniform over three actions,
                  two of which send. That gives a 2/3 send rate, a steady-state
                  fatigue of 1.00 against a churn threshold of 0.70, and an
                  early training signal dominated by users who quit. Setting the
                  head bias to log(0.80, 0.13, 0.07) starts the policy at a
                  survivable send rate instead. Nothing else changes.

  `warmstart`  -- the *behavioural* analogue, and the closest thing to real
                  pre-seeding. Before PPO runs, the actor is briefly trained by
                  cross-entropy to imitate a peak-hour sending policy, so the
                  first rollout is collected by a policy that already sends at
                  plausible hours and can be improved from, rather than one that
                  must discover sending from scratch under a clip that makes
                  large policy changes expensive.

    python tools/ppo_fix.py
    python tools/ppo_fix.py --variants baseline gamma090 --jobs 5
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cane  # noqa: E402
from cane.agents_deep import DEVICE, PPOAgent  # noqa: E402
from cane.core import ENGAGE, HOLD, IDX_HOUR, N_ACTIONS  # noqa: E402
from tools.preseed_study import PEAK_HOURS  # noqa: E402

OUT = ROOT / "artifacts"

# Same prior the exploration study derived: E[kappa]/(1-lam) = 0.282, comfortably
# under the 0.70 churn threshold, and roughly the action mix any real
# notification product would use.
HOLD_PRIOR = np.array([0.80, 0.13, 0.07])

VARIANTS = {
    "baseline":  dict(hp={},                  init=None,       warm=0),
    "gamma090":  dict(hp=dict(gamma=0.90),    init=None,       warm=0),
    "hold_init": dict(hp={},                  init=HOLD_PRIOR, warm=0),
    "warmstart": dict(hp={},                  init=None,       warm=2000),
    "both":      dict(hp=dict(gamma=0.90, ent_coef=0.03),
                      init=HOLD_PRIOR, warm=2000),
}


def set_action_prior(agent: PPOAgent, prior: np.ndarray) -> None:
    """Start the policy at `prior` instead of near-uniform.

    Only the output bias is touched. The weights keep their orthogonal
    initialisation, so the policy still depends on the state from the first
    step -- this shifts where it starts, it does not freeze what it can learn.
    """
    logits = np.log(np.clip(prior, 1e-6, None))
    logits = logits - logits.mean()
    with torch.no_grad():
        head = agent.net.actor[-1]
        head.bias.copy_(torch.as_tensor(logits, dtype=torch.float32,
                                        device=DEVICE))


def warm_start(agent: PPOAgent, env, steps: int,
               rng: np.random.Generator) -> int:
    """Behaviour-clone the actor onto a peak-hour sending policy.

    This is the on-policy counterpart of pre-filling a replay buffer: instead of
    handing the critic profitable transitions, it hands the actor a starting
    policy that produces them. The critic is left untouched and learns normally
    from the first rollout.

    Returns the number of send-labelled examples used, so the caller can report
    how much the actor was actually shown rather than assuming.
    """
    opt = torch.optim.Adam(agent.net.actor.parameters(), lr=1e-3)
    s, _ = env.reset()
    feats, labels = [], []
    for _ in range(steps):
        hour = int(s[IDX_HOUR])
        a = ENGAGE if (hour in PEAK_HOURS and rng.random() < 0.5) else HOLD
        feats.append(agent._features(s))
        labels.append(a)
        ns, _, term, trunc, _ = env.step(a)
        s = env.reset()[0] if (term or trunc) else ns

    x = torch.as_tensor(np.asarray(feats), dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(agent.net.actor(x), y)
        loss.backward()
        opt.step()
    return int(sum(1 for a in labels if a != HOLD))


def run_cell(variant: str, archetype: str, seed: int, episodes: int) -> dict:
    spec = VARIANTS[variant]
    t0 = time.time()

    agent = PPOAgent(seed=seed, **spec["hp"])
    if spec["init"] is not None:
        set_action_prior(agent, spec["init"])

    cloned = 0
    if spec["warm"]:
        warm_env = cane.CANEEnv(seed=2000 + seed, archetype=archetype)
        cloned = warm_start(agent, warm_env, spec["warm"],
                            np.random.default_rng(seed))

    train_env = cane.CANEEnv(seed=1000 + seed, archetype=archetype)
    churn = 0
    for _ in range(episodes):
        agent.reset()
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
        "agent": "PPO", "variant": variant, "archetype": archetype,
        "seed": seed,
        "reward_mean": float(m["reward_mean"]), "ctr": float(m["ctr"]),
        "sends_per_episode": float(m["sends_per_episode"]),
        "optout_rate": float(m["optout_rate"]),
        "train_churn_rate": churn / episodes,
        "cloned_sends": cloned,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--archetypes", nargs="*", default=list(cane.ARCHETYPES))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    cells = [(v, a, s) for v in args.variants
             for a in args.archetypes for s in range(args.seeds)]

    print("=" * 92)
    print("PPO rescue: the DQN buffer trick, translated for an on-policy agent")
    print("=" * 92)
    print(f"hold_init prior : {HOLD_PRIOR.tolist()}  "
          f"(steady-state fatigue 0.282 vs churn threshold 0.70)")
    print(f"warmstart       : behaviour cloning on peak hours {PEAK_HOURS}")
    print(f"cells {len(cells)} | workers {jobs} | episodes {args.episodes}\n",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_cell, v, a, s, args.episodes): (v, a, s)
                for v, a, s in cells}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i:2d}/{len(cells)}] {r['variant']:9} "
                  f"{r['archetype']:17} sends {r['sends_per_episode']:6.2f}  "
                  f"reward {r['reward_mean']:8.2f}  ctr {r['ctr']:.3f}  "
                  f"churn {r['train_churn_rate']:4.0%}  "
                  f"[{r['seconds']:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "ppo_fix.csv", index=False)

    print()
    print("=" * 92)
    print("SENDS PER WEEK")
    print(df.pivot_table(index="archetype", columns="variant",
                         values="sends_per_episode").round(2).to_string())
    print()
    print("MEAN REWARD")
    print(df.pivot_table(index="archetype", columns="variant",
                         values="reward_mean").round(2).to_string())
    print()
    print("CTR")
    print(df.pivot_table(index="archetype", columns="variant",
                         values="ctr").round(3).to_string())

    print()
    print("=" * 92)
    print("VERDICT")
    base = df[df.variant == "baseline"].set_index("archetype")
    for variant in [v for v in args.variants if v != "baseline"]:
        d = df[df.variant == variant].set_index("archetype")
        talking = int((d["sends_per_episode"] > 0.5).sum())
        gain = d["reward_mean"].mean() - base["reward_mean"].mean()
        print(f"  {variant:9} contacts {talking}/{len(d)} people  "
              f"mean reward {gain:+7.2f} vs baseline")
    print("\nwrote artifacts/ppo_fix.csv")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
