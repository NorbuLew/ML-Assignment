"""Instrument a training run to find out *why* the deep agents stop sending.

Asserting "value pessimism" from one Q-value probe on a finished checkpoint is
not evidence -- it describes the end state, not the cause. This records the
whole trajectory: what the agent experienced while exploring, what its Q-values
did in response, and at which episode the policy flipped to silence.

Logged every `--every` episodes:

* epsilon, and the fraction of actions that were sends
* mean training reward and the share of episodes ending in churn
* Q(Hold), Q(Engage), Q(Incentive) at a fixed probe state -- the archetype's
  most profitable hour, with a fresh unfatigued user
* what a greedy policy would do at that moment

    python tools/collapse_trace.py --archetype OfficeWorker
    python tools/collapse_trace.py --archetype Housewife --agent dqn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cane  # noqa: E402
from cane.agents_deep import DDQNAgent, DQNAgent  # noqa: E402
from cane.core import ARCHETYPE_TABLE, ARCHETYPE_W, CANE_CONFIG, IDX_HOUR  # noqa: E402

OUT = ROOT / "artifacts"
ACTIONS = ["Hold", "Engage", "Incentive"]


def break_even(action: int = 1) -> float:
    C = CANE_CONFIG
    debt = C["W_fat"] * C["kappa"][action] / (1.0 - C["lam"])
    return (C["W_send"][action] + debt) / C["R_click"]


def probe_state(archetype: str):
    """The state that most favours sending: best hour, zero fatigue."""
    curve = np.asarray(ARCHETYPE_TABLE[archetype], dtype=float)
    w = ARCHETYPE_W[archetype][1] if isinstance(ARCHETYPE_W[archetype], dict) else 1.0
    best_h = int(np.argmax(np.clip(curve * w, 0, 1)))

    env = cane.CANEEnv(seed=7000, archetype=archetype)
    s, _ = env.reset(seed=900_000)
    for _ in range(300):
        if int(s[IDX_HOUR]) == best_h:
            break
        s, _, term, trunc, _ = env.step(0)  # hold, so fatigue stays at zero
        if term or trunc:
            s, _ = env.reset(seed=900_000)
    return s, best_h, float(np.clip(curve * w, 0, 1)[best_h])


def q_of(agent, state) -> np.ndarray:
    with torch.no_grad():
        feat = torch.as_tensor(agent._features(state),
                               dtype=torch.float32).unsqueeze(0)
        return agent.online_net(feat).squeeze(0).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archetype", default="OfficeWorker")
    ap.add_argument("--agent", default="ddqn", choices=["dqn", "ddqn"])
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    make = DQNAgent if args.agent == "dqn" else DDQNAgent
    agent = make(seed=args.seed)
    env = cane.CANEEnv(seed=1000 + args.seed, archetype=args.archetype)

    s_probe, best_h, p_best = probe_state(args.archetype)
    be = break_even(1)

    print(f"archetype   : {args.archetype}")
    print(f"agent       : {args.agent.upper()}  seed {args.seed}")
    print(f"probe state : hour {best_h:02d}:00, fatigue 0.00, "
          f"p(click) {p_best:.3f}")
    print(f"break-even  : {be:.3f}  -> sending here is "
          f"{'PROFITABLE' if p_best > be else 'unprofitable'}")
    print()
    print(f"{'ep':>4} {'eps':>5} {'send%':>6} {'reward':>9} {'churn%':>7} "
          f"{'Q(Hold)':>9} {'Q(Eng)':>8} {'Q(Inc)':>8}  greedy")
    print("-" * 78)

    rows = []
    window_rewards, window_sends, window_steps, window_churn, window_eps = \
        [], 0, 0, 0, 0

    for ep in range(1, args.episodes + 1):
        s, _ = env.reset()
        done, total, steps, sends = False, 0.0, 0, 0
        while not done:
            a, aux = agent.act(s, greedy=False)
            nxt, r, term, trunc, _ = env.step(a)
            # The package's Agent protocol folds "store the transition" and
            # "take a gradient step" into one call; there is no separate
            # observe()/learn() pair.
            agent.update(s, a, r, nxt, term, aux)
            total += r
            sends += int(a != 0)
            steps += 1
            s, done = nxt, term or trunc
        window_rewards.append(total)
        window_sends += sends
        window_steps += steps
        window_churn += int(term)
        window_eps += 1

        if ep % args.every == 0:
            q = q_of(agent, s_probe)
            eps = float(agent._epsilon())
            row = {
                "episode": ep,
                "epsilon": eps,
                "send_share": window_sends / max(window_steps, 1),
                "train_reward": float(np.mean(window_rewards)),
                "churn_rate": window_churn / max(window_eps, 1),
                "q_hold": float(q[0]), "q_engage": float(q[1]),
                "q_incentive": float(q[2]),
                "greedy": ACTIONS[int(np.argmax(q))],
            }
            rows.append(row)
            print(f"{ep:>4d} {eps:>5.2f} {row['send_share']:>6.1%} "
                  f"{row['train_reward']:>9.2f} {row['churn_rate']:>7.0%} "
                  f"{q[0]:>9.3f} {q[1]:>8.3f} {q[2]:>8.3f}  {row['greedy']}",
                  flush=True)
            window_rewards, window_sends, window_steps, window_churn, \
                window_eps = [], 0, 0, 0, 0

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"collapse_trace_{args.agent}_{args.archetype}.csv"
    df.to_csv(path, index=False)

    # Where did greedy last prefer a send?
    sending = df[df["greedy"] != "Hold"]
    print()
    print("=" * 78)
    if sending.empty:
        print("The greedy policy never preferred a send at the probe state, "
              "not even at episode 1.")
    else:
        last = int(sending["episode"].max())
        print(f"Greedy last preferred a send at episode {last} "
              f"(of {args.episodes}).")
        first_hold = df[(df["episode"] > last) & (df["greedy"] == "Hold")]
        if not first_hold.empty:
            e = int(first_hold['episode'].min())
            r = df[df.episode == e].iloc[0]
            print(f"It flipped to Hold at episode {e}, with epsilon "
                  f"{r['epsilon']:.2f} and churn rate {r['churn_rate']:.0%}.")
    print(f"peak churn rate observed : {df['churn_rate'].max():.0%}")
    print(f"final Q(Hold) - Q(Engage): "
          f"{df.iloc[-1]['q_hold'] - df.iloc[-1]['q_engage']:+.3f}")
    print(f"wrote {path.relative_to(ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
