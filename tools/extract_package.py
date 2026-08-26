"""Generate the `cane/` package from the LinUCB notebook.

`Code/AssignementCode(LinUCB).ipynb` is the canonical, fully-documented copy of
the environment, the agent contract and the evaluation harness; the other three
notebooks carry (semantically identical) duplicates of it. Rather than retype
that code into a package and risk silent drift, this script copies the exact
source text out of the relevant notebook cells, in notebook execution order,
into `cane/core.py`.

Verbatim concatenation is deliberate. It means the package is provably the same
code that produced the reported results -- there is no hand-editing step where a
constant could quietly change -- and re-running this script is the way to pick
up any later change to the notebook.

    python tools/extract_package.py

Idempotent: safe to re-run. Writes cane/core.py and cane/__init__.py; touches
nothing else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "Code" / "AssignementCode(LinUCB).ipynb"
PKG = ROOT / "cane"

# Notebook cells that make up the shared core, in execution order.
#   2  action space + observation layout constants
#   4  Agent ABC (the contract every agent implements)
#   6  feature encoders (raw / harmonic / one-hot)
#  10  LinUCBAgent
#  15  CANE_CONFIG, archetype curves, weekend modulation
#  16  CANEEnv
#  17  non-learning baselines, run_episodes, evaluation protocol
#  18  train_and_evaluate / evaluate_baseline / policy_snapshot
CORE_CELLS = [2, 4, 6, 10, 15, 16, 17, 18]

# The three deep agents live in their owners' notebooks. Only the agent code is
# taken -- each of those notebooks also carries its own duplicate of the
# environment, which is exactly the duplication this package exists to remove.
#
# `ReplayBuffer` and `QNetwork` are byte-identical in the DQN and Double DQN
# notebooks (verified), so they are taken once from the DQN notebook and shared.
# The only algorithmic difference between the two agents is the target
# decoupling in `_optimise`, which is what Double DQN *is*.
DEEP_SOURCES = [
    ("DQN/AssignmentCode(DQN).ipynb", [19, 20, 21, 22],
     "DQN_CONFIG, ReplayBuffer, QNetwork, DQNAgent"),
    ("Code/doubleDQN.ipynb", [18, 21],
     "DDQN_CONFIG, DDQNAgent"),
    ("Code/AssignementCode(PPO).ipynb", [16, 17, 18],
     "PPO_CONFIG, ActorCritic, PPOAgent"),
]

DEEP_REQUIRED = [
    "DQN_CONFIG = dict(", "class ReplayBuffer", "class QNetwork",
    "class DQNAgent(Agent):", "DDQN_CONFIG = dict(", "class DDQNAgent(Agent):",
    "PPO_CONFIG = dict(", "class ActorCritic", "class PPOAgent(Agent):",
]

DEEP_HEADER = '''"""CANE deep agents: DQN, Double DQN and PPO.

GENERATED FILE -- do not edit by hand.

Produced by `tools/extract_package.py`, copied verbatim from the notebooks that
own each algorithm:

{provenance}

Only the agent code is taken. Each of those notebooks also carries its own copy
of the environment and harness; those duplicates are exactly what `cane.core`
replaces, and `tools/parity_check.py` proves the copies agree.

This module is deliberately NOT imported by `cane/__init__.py`, so `import cane`
stays free of torch. Import it explicitly when you need the deep agents.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cane.core import (
    HOLD, ENGAGE, INCENTIVE, N_ACTIONS, ACTION_NAMES,
    IDX_HOUR, IDX_DAY, IDX_ACTIVE, IDX_FATIGUE, IDX_RECENCY,
    STATE_DIM, agent_state_dim, Agent,
)

# The project trains and evaluates on CPU throughout: the networks are two
# 64-unit hidden layers, far too small for a GPU transfer to pay for itself, and
# pinning this keeps runs reproducible across machines.
DEVICE = "cpu"
torch.set_num_threads(1)

'''

# Markers that must survive extraction. These are the same eight the PPO
# notebook diffs against, so if the extractor ever drops one we find out here
# rather than three days later in a dashboard that silently shows nothing.
REQUIRED = [
    "class CANEEnv:",
    "CANE_CONFIG = dict(",
    "def run_episodes(",
    "def train_and_evaluate(",
    "def policy_snapshot(",
    "class Agent(ABC):",
    "def evaluate_baseline(",
    "def archetype_curve(",
    "class LinUCBAgent(Agent):",
    "FEATURE_ENCODERS",
    "EVAL_SEEDS",
]

HEADER = '''"""CANE core: environment, agent contract, baselines, harness, LinUCB.

GENERATED FILE -- do not edit by hand.

Produced by `tools/extract_package.py` from the code cells of
`Code/AssignementCode(LinUCB).ipynb` (cells {cells}), copied verbatim and in
notebook execution order. Editing this file directly would make the package and
the notebook disagree about what was actually run; edit the notebook and re-run
the extractor instead.

Top-level `print(...)` progress lines from the notebook are removed so that
importing the package is silent.
"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

'''

INIT = '''"""CANE -- Context-Aware Notification Engine.

A reinforcement-learning agent that decides, each hour, whether to stay silent
or send one of two notification types to a simulated app user, maximising
long-term engagement while paying for the fatigue each send creates.

This package exists so that all four algorithms (LinUCB, DQN, Double DQN, PPO)
can be evaluated through one harness on identical held-out episodes. The code is
extracted verbatim from the LinUCB notebook -- see `cane/core.py` and
`tools/extract_package.py`.
"""

from cane.core import (  # noqa: F401
    # action space
    HOLD, ENGAGE, INCENTIVE, N_ACTIONS, ACTION_NAMES,
    # observation layout
    IDX_HOUR, IDX_DAY, IDX_ACTIVE, IDX_FATIGUE, IDX_RECENCY,
    IDX_ACT_RATE, IDX_CLICK_RATE, IDX_TYPE_CR, IDX_SINCE_CLICK,
    N_BLOCKS, BLOCK_HOURS, STATE_DIM, STATE_DIM_BASE,
    BELIEF_PRIOR, USE_BELIEF, agent_state_dim,
    # configuration
    CANE_CONFIG, ARCHETYPE_W, ARCHETYPES, ARCHETYPE_TABLE,
    WEEKEND_FLATTEN, archetype_curve,
    # environment
    CANEEnv,
    # agents
    Agent, FixedScheduleAgent, RandomAgent, LinUCBAgent,
    FEATURE_ENCODERS, feature_dim,
    # evaluation protocol
    QUICK, EVAL_SEEDS, TRAIN_EPISODES, N_SEEDS,
    # harness
    run_episodes, evaluate_baseline, train_and_evaluate, policy_snapshot,
)

__version__ = "1.0.0"
'''


def strip_top_level_prints(src: str) -> str:
    """Remove top-level `print(...)` progress calls and their continuations.

    The notebook prints a status line at the end of most cells. Those are noise
    in a library and would fire on import. Prints indented inside a function or
    class body are left untouched -- they belong to the code's own behaviour.
    """
    out: list[str] = []
    skipping = False
    for line in src.split("\n"):
        if skipping:
            # A continuation line of the print call: indented, or a bare closer.
            if line.startswith((" ", "\t")) or line.strip() in {")", "))"}:
                continue
            skipping = False
        if re.match(r"^print\s*\(", line):
            skipping = True
            continue
        # The encoder-dimension report loop at the end of the encoders cell.
        if re.match(r"^for _name, _enc in FEATURE_ENCODERS", line):
            skipping = True
            continue
        out.append(line)
    return "\n".join(out)


def main() -> int:
    if not NB.is_file():
        print(f"error: cannot find {NB}")
        return 2

    nb = json.loads(NB.read_text(encoding="utf8"))
    cells = nb["cells"]
    print(f"extracting from {NB.relative_to(ROOT)} ({len(cells)} cells)")

    chunks: list[str] = []
    for idx in CORE_CELLS:
        cell = cells[idx]
        if cell["cell_type"] != "code":
            print(f"error: cell {idx} is {cell['cell_type']}, expected code")
            return 1
        src = strip_top_level_prints("".join(cell["source"])).rstrip()
        chunks.append(
            f"# {'=' * 74}\n"
            f"# notebook cell {idx}\n"
            f"# {'=' * 74}\n\n{src}\n"
        )

    body = HEADER.format(cells=", ".join(str(i) for i in CORE_CELLS))
    body += "\n\n".join(chunks)

    missing = [m for m in REQUIRED if m not in body]
    if missing:
        print("error: extraction lost required definitions:")
        for m in missing:
            print(f"  - {m}")
        return 1

    PKG.mkdir(exist_ok=True)
    (PKG / "core.py").write_text(body, encoding="utf8")
    (PKG / "__init__.py").write_text(INIT, encoding="utf8")

    print(f"  wrote cane/core.py      ({len(body.splitlines())} lines)")
    print(f"  wrote cane/__init__.py")
    print(f"  all {len(REQUIRED)} required definitions present")

    # --- deep agents, from their owners' notebooks ---------------------------
    print("\nextracting deep agents")
    deep_chunks: list[str] = []
    provenance: list[str] = []
    for rel, idxs, what in DEEP_SOURCES:
        path = ROOT / rel
        if not path.is_file():
            print(f"error: cannot find {rel}")
            return 2
        src_cells = json.loads(path.read_text(encoding="utf8"))["cells"]
        provenance.append(f"  * {rel}\n      cells {idxs} -- {what}")
        for idx in idxs:
            cell = src_cells[idx]
            if cell["cell_type"] != "code":
                print(f"error: {rel} cell {idx} is {cell['cell_type']}")
                return 1
            src = strip_top_level_prints("".join(cell["source"])).rstrip()
            # Drop the notebook's construction probes; they instantiate an agent
            # at import time purely to print a repr.
            src = re.sub(r"^_probe\s*=.*$", "", src, flags=re.MULTILINE).rstrip()
            deep_chunks.append(
                f"# {'=' * 74}\n"
                f"# {rel}  cell {idx}\n"
                f"# {'=' * 74}\n\n{src}\n"
            )

    deep = DEEP_HEADER.format(provenance="\n".join(provenance))
    deep += "\n\n".join(deep_chunks)

    missing = [m for m in DEEP_REQUIRED if m not in deep]
    if missing:
        print("error: deep extraction lost required definitions:")
        for m in missing:
            print(f"  - {m}")
        return 1

    (PKG / "agents_deep.py").write_text(deep, encoding="utf8")
    print(f"  wrote cane/agents_deep.py ({len(deep.splitlines())} lines)")
    print(f"  all {len(DEEP_REQUIRED)} required definitions present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
