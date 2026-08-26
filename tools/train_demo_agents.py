"""Train one Double DQN per archetype, for the dashboard's live simulation.

The notebooks' checkpointing cells train on the *mixed* population -- one agent
that sees all five archetypes interleaved -- and on that mixture the reward
structure makes holding the best simple policy, so every saved checkpoint is
silent on every lane. That is a real result and the dashboard says so, but it
makes a poor animation: five lanes of nothing.

The per-archetype numbers in the results CSVs come from a different code path.
`run_all()` in each notebook trains a fresh agent per archetype and reports it
without ever saving it. This script does that same training and *keeps* the
weights, so the simulation page can show five agents that genuinely learned
five different people.

These checkpoints are for the demo only. They are written to their own folder
and are never picked up by the ensemble study or the results CSVs.

    python tools/train_demo_agents.py            # all five, in parallel
    python tools/train_demo_agents.py --jobs 1   # serial, for debugging
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cane  # noqa: E402
from cane.agents_deep import DDQNAgent, DQNAgent, PPOAgent  # noqa: E402
from cane.core import LinUCBAgent  # noqa: E402

# The live simulation should be able to show any of the four algorithms,
# not just the one that happens to win. Each is trained per archetype and
# kept, which the notebooks never do.
MAKERS = {"ddqn": DDQNAgent, "dqn": DQNAgent, "ppo": PPOAgent,
          "linucb": LinUCBAgent}

OUT = ROOT / "Code" / "models" / "demo"


def train_one(archetype: str, seed: int, episodes: int,
              variant: str = "plain", algo: str = "ddqn") -> dict:
    """Train and save a single archetype-specialised agent.

    `variant="optimistic"` applies the one intervention the exploration study
    found that makes a deep agent send at all: a positive initial bias on the
    two send heads only. A uniform lift changes nothing, because argmax breaks
    ties toward index 0, which is Hold.
    """
    t0 = time.time()
    if variant == "optimistic":
        from cane.exploration_study import VARIANTS, make_agent
        agent = make_agent(seed, VARIANTS["optimistic_send"], archetype)
    elif algo != "ddqn":
        # Only Double DQN has a tuned configuration; the other three are
        # trained at their shipped defaults so the comparison on the
        # simulation page is between algorithms, not between tuning efforts.
        agent = MAKERS[algo](seed=seed)
    elif variant == "tuned":
        # The winning configuration from tools/tune_study.py ("both_long"):
        # the DQN notebook's hyperparameters, a pre-seeded replay buffer, and
        # 1500 training episodes instead of 600. It is the only variant that
        # recovered both coverage and profit -- and the largest single gain
        # came from the episode count, not from any of the clever parts.
        from tools.preseed_study import TEAMMATE_HP, preseed_buffer
        agent = DDQNAgent(seed=seed, **TEAMMATE_HP)
        preseed_buffer(agent, cane.CANEEnv(seed=2000 + seed,
                                           archetype=archetype),
                       1000, np.random.default_rng(seed))
        episodes = max(episodes, 1500)
    else:
        agent = DDQNAgent(seed=seed)

    if variant == "mincontact":
        # Wrapped *before* training, not after. That is the whole point: the
        # forced transitions go into the replay buffer, so the agent learns
        # what a send at 09:00 was worth for this person and can then choose
        # to send at 09:00 on its own. Wrapping only at evaluation time gives
        # the deadline every send and teaches the agent nothing.
        from cane.min_contact import MinContactAgent
        agent = MinContactAgent(agent, min_sends=1, window=24, horizon=168)

    env = cane.CANEEnv(seed=1000 + seed, archetype=archetype)
    cane.run_episodes(agent, env, n_episodes=episodes, learn=True, greedy=False)

    eval_env = cane.CANEEnv(seed=7000 + seed, archetype=archetype)
    m, _, _, _ = cane.run_episodes(agent, eval_env, seeds=cane.EVAL_SEEDS,
                                   learn=False, greedy=True)

    OUT.mkdir(parents=True, exist_ok=True)
    # The filename must start with "ddqn" for find_checkpoints/_infer_family to
    # place it in the right family; the archetype rides after the prefix.
    tag = "demo" if variant == "plain" else f"demo{variant}"
    # LinUCB's "model" is plain numpy (A_inv and theta per arm), so it is saved
    # as an .npz rather than a torch checkpoint. persistence.load_agent keys off
    # the suffix.
    if algo == "linucb":
        # LinUCB has no state_dict_for_save: its "model" is the ridge statistics
        # themselves. The keys below are exactly what persistence.load_linucb
        # reads back, so the two stay in step.
        path = OUT / f"linucb_{tag}_{archetype}.npz"
        np.savez(path, A=agent.A, A_inv=agent.A_inv, b=agent.b,
                 n_pulls=agent.n_pulls, alpha=agent.alpha, lam=agent.lam,
                 n_actions=agent.n_actions,
                 encoder_name=agent.encoder_name)
    else:
        path = OUT / f"{algo}_{tag}_{archetype}.pt"
        torch.save(agent.state_dict_for_save(), path)

    return {
        "archetype": archetype,
        "algo": algo,
        "variant": variant,
        "path": str(path),
        "reward_mean": float(m["reward_mean"]),
        "ctr": float(m["ctr"]),
        "sends_per_episode": float(m["sends_per_episode"]),
        "optout_rate": float(m["optout_rate"]),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", default="ddqn", choices=list(MAKERS),
                    help="which algorithm to specialise per archetype")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=cane.TRAIN_EPISODES)
    ap.add_argument("--archetypes", nargs="*", default=list(cane.ARCHETYPES))
    ap.add_argument("--variant", default="plain",
                    choices=["plain", "optimistic", "mincontact", "tuned"],
                    help="'optimistic' biases the send heads at init -- the "
                         "only setting that makes the agent send at all")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = one per archetype")
    args = ap.parse_args()

    jobs = args.jobs or min(len(args.archetypes),
                            max(1, (os.cpu_count() or 4) - 2))

    print("=" * 78)
    print("Training per-archetype demo agents (Double DQN)")
    print("=" * 78)
    print(f"archetypes : {', '.join(args.archetypes)}")
    print(f"episodes   : {args.episodes} | seed {args.seed} | "
          f"variant {args.variant} | workers {jobs}")
    print(f"output     : {OUT.relative_to(ROOT)}")
    print(flush=True)

    t0 = time.time()
    results = []
    if jobs == 1:
        for arch in args.archetypes:
            r = train_one(arch, args.seed, args.episodes, args.variant,
                          args.algo)
            results.append(r)
            print(f"  {r['archetype']:17} reward {r['reward_mean']:8.2f}  "
                  f"sends {r['sends_per_episode']:6.2f}  ctr {r['ctr']:.3f}  "
                  f"[{r['seconds']:.0f}s]", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(train_one, arch, args.seed, args.episodes,
                                   args.variant, args.algo)
                       for arch in args.archetypes]
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                print(f"  {r['archetype']:17} reward {r['reward_mean']:8.2f}  "
                      f"sends {r['sends_per_episode']:6.2f}  "
                      f"ctr {r['ctr']:.3f}  [{r['seconds']:.0f}s]", flush=True)

    order = {a: i for i, a in enumerate(cane.ARCHETYPES)}
    results.sort(key=lambda r: order.get(r["archetype"], 99))

    print()
    print("=" * 78)
    talkers = [r for r in results if r["sends_per_episode"] > 0.5]
    print(f"{len(talkers)} of {len(results)} archetype agents learned to send.")
    if not talkers:
        print("All five stayed silent even when trained on a single archetype.")
        print("That is a stronger version of the same finding, not a failure.")
    print(f"elapsed {time.time() - t0:.0f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
