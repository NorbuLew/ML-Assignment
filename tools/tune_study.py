"""Break the silence *without* flooding: combining the two half-fixes.

Two interventions have each solved half of this problem and broken the other
half:

* Safe exploration (`tools/fix_exploration.py`) drove training churn from 52%
  to 0% by keeping exploratory sends off an already-fatigued user. The training
  signal became clean -- and the agent still sent nothing, because at gamma=0.99
  a send's discounted fatigue debt makes contact unaffordable in the agent's own
  accounting.

* The DQN notebook's configuration (`tools/preseed_study.py`) made contact
  affordable by lowering gamma to 0.90, and it worked: Double DQN went from
  contacting nobody to contacting all five. But it floods -- 26 sends a week
  into NormalStudent at a CTR of 0.242, against a break-even CTR of 0.340, so
  every one of those sends loses money -- and training churn rose from 52% to
  71%. It broke the silence by breaking the user.

Each fixes exactly what the other breaks, which is the reason to test them
together rather than to keep tuning either one. `both_safe` is that
combination. Two controls sit beside it so a win cannot be misread: `both_long`
tests whether the flooding is simply undertraining, and `gamma095` tests whether
a gamma between the two extremes is all that was ever needed.

    python tools/tune_study.py
    python tools/tune_study.py --variants both both_safe --archetypes Housewife
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
from cane.core import CANE_CONFIG, IDX_HOUR  # noqa: E402
from tools.fix_exploration import (  # noqa: E402
    FATIGUE_CEILING, HOLD_PRIOR, SafeExplorationDDQN,
)
from tools.personalisation_test import TARGET_HOUR, circular_hour_error  # noqa: E402
from tools.preseed_study import TEAMMATE_HP, preseed_buffer  # noqa: E402

OUT = ROOT / "artifacts"

# The click probability a send must clear to pay for itself:
#   (W_send + W_fat*kappa/(1-lam)) / R_click
# Sending below this rate is a loss no amount of training can turn into a gain,
# so it is the line every variant below is judged against.
BREAK_EVEN_CTR = 0.340

VARIANTS = {
    # The control: teammate's configuration exactly as it arrived.
    "both": dict(hp=TEAMMATE_HP, safe=False, preseed=True, episodes=None),

    # The combination this study exists to test.
    "both_safe": dict(hp=TEAMMATE_HP, safe=True, preseed=True, episodes=None),

    # Control 1 -- is the flooding just undertraining?
    "both_long": dict(hp=TEAMMATE_HP, safe=False, preseed=True, episodes=1500),

    # Control 2 -- is a middle gamma all that was ever needed?
    "gamma095": dict(hp=dict(gamma=0.95), safe=False, preseed=False,
                     episodes=None),
}


def build(variant: str, seed: int):
    spec = VARIANTS[variant]
    if spec["safe"]:
        return SafeExplorationDDQN(seed=seed, prior=HOLD_PRIOR,
                                   fatigue_ceiling=FATIGUE_CEILING,
                                   **spec["hp"])
    return DDQNAgent(seed=seed, **spec["hp"])


def send_hours(agent, archetype: str, seeds) -> np.ndarray:
    counts = np.zeros(24, dtype=int)
    env = cane.CANEEnv(seed=7000, archetype=archetype)
    for sd in seeds:
        agent.reset()
        s, _ = env.reset(seed=int(sd))
        done = False
        while not done:
            a, _ = agent.act(s, greedy=True)
            if a != 0:
                counts[int(s[IDX_HOUR])] += 1
            s, _, term, trunc, _ = env.step(a)
            done = term or trunc
    return counts


def run_cell(variant: str, archetype: str, seed: int,
             default_episodes: int) -> dict:
    spec = VARIANTS[variant]
    episodes = spec["episodes"] or default_episodes
    t0 = time.time()

    agent = build(variant, seed)

    if spec["preseed"]:
        seed_env = cane.CANEEnv(seed=2000 + seed, archetype=archetype)
        preseed_buffer(agent, seed_env, 1000, np.random.default_rng(seed))

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

    counts = send_hours(agent, archetype, cane.EVAL_SEEDS[:40])
    total = int(counts.sum())
    peak = int(np.argmax(counts)) if total else -1
    if total:
        idx = [(peak - 1) % 24, peak, (peak + 1) % 24]
        conc = float(counts[idx].sum()) / total
    else:
        conc = 0.0

    ctr = float(m["ctr"])
    return {
        "agent": "DDQN", "variant": variant, "archetype": archetype,
        "seed": seed, "episodes": episodes,
        "reward_mean": float(m["reward_mean"]), "ctr": ctr,
        "sends_per_episode": float(m["sends_per_episode"]),
        "optout_rate": float(m["optout_rate"]),
        "train_churn_rate": churn / episodes,
        # The single most diagnostic column: a policy sending below break-even
        # is losing money on every contact regardless of how much reward the
        # good archetypes bring in.
        "above_break_even": bool(ctr >= BREAK_EVEN_CTR),
        "peak_hour": peak, "target_hour": TARGET_HOUR[archetype],
        "hour_error": (circular_hour_error(peak, TARGET_HOUR[archetype])
                       if peak >= 0 else 99),
        "concentration": round(conc, 3),
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

    print("=" * 100)
    print("Contact without flooding: combining safe exploration with an "
          "affordable discount rate")
    print("=" * 100)
    print(f"break-even CTR : {BREAK_EVEN_CTR}  "
          f"(a send below this loses money by construction)")
    print(f"safe explore   : prior {HOLD_PRIOR.tolist()}, "
          f"fatigue ceiling {FATIGUE_CEILING}")
    print(f"churn threshold: {CANE_CONFIG['churn_threshold']}")
    print(f"cells {len(cells)} | workers {jobs}\n", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(run_cell, v, a, s, args.episodes): (v, a, s)
                for v, a, s in cells}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            flag = "OK " if r["above_break_even"] else "LOSS"
            print(f"  [{i:2d}/{len(cells)}] {r['variant']:10} "
                  f"{r['archetype']:17} sends {r['sends_per_episode']:6.2f}  "
                  f"reward {r['reward_mean']:8.2f}  ctr {r['ctr']:.3f} {flag} "
                  f"peak {r['peak_hour']:2d}[{r['target_hour']:2d}] "
                  f"churn {r['train_churn_rate']:4.0%}  "
                  f"[{r['seconds']:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "tune_study.csv", index=False)

    for col, title in [("sends_per_episode", "SENDS PER WEEK"),
                       ("reward_mean", "MEAN REWARD"),
                       ("ctr", "CTR  (break-even 0.340)"),
                       ("train_churn_rate", "TRAINING CHURN"),
                       ("hour_error", "HOUR ERROR vs that person's best hour")]:
        print()
        print(title)
        print(df.pivot_table(index="archetype", columns="variant",
                             values=col).round(3).to_string())

    print()
    print("=" * 100)
    print("VERDICT")
    for variant in args.variants:
        d = df[df.variant == variant]
        contacted = int((d["sends_per_episode"] > 0.5).sum())
        profitable = int((d["reward_mean"] > 0).sum())
        above = int(d["above_break_even"].sum())
        print(f"  {variant:10} contacts {contacted}/{len(d)}  "
              f"profitable {profitable}/{len(d)}  "
              f"above break-even {above}/{len(d)}  "
              f"total reward {d['reward_mean'].sum():8.2f}  "
              f"mean churn {d['train_churn_rate'].mean():4.0%}")
    print()
    print("The target is a variant that contacts 5/5 AND is profitable on 5/5.")
    print("Contacting everyone at a loss is not an improvement over silence.")
    print("\nwrote artifacts/tune_study.csv")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
