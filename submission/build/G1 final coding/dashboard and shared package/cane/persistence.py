"""Load trained CANE agents back from disk.

The notebooks each save checkpoints but none of them ever reloads one, so there
was no way to use a trained policy without retraining it. That blocks two things
this project needs: an ensemble over the already-trained members, and a
dashboard that inspects a learned policy interactively.

Every agent family stores enough to rebuild itself -- the deep agents record
`cfg` and `d_in` alongside the weights, and LinUCB's entire learned state is its
per-arm ridge statistics -- so loading needs no outside knowledge beyond which
family a file belongs to.

    from cane.persistence import load_agent
    agent = load_agent("Code/models/ppo_seed0.pt")

Unlike the rest of the package this module is hand-written rather than extracted
from a notebook: it is new behaviour, not a copy of existing code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Which checkpoint keys identify which family. The deep agents are told apart by
# structure (DQN keeps a target network, PPO does not); DQN and Double DQN share
# a structure, so their family comes from the filename.
_TORCH_SUFFIXES = {".pt", ".pth"}


def _infer_family(path: Path, blob: dict) -> str:
    stem = path.stem.lower()
    if "ddqn" in stem or "double" in stem:
        return "DDQN"
    if "ppo" in stem or "net" in blob and "online_net" not in blob:
        return "PPO"
    if "online_net" in blob:
        return "DQN"
    raise ValueError(
        f"cannot tell which agent family {path.name} belongs to; "
        f"keys were {sorted(blob)}"
    )


def load_linucb(path: str | Path, alpha: float | None = None):
    """Rebuild a LinUCB agent from a .npz of its ridge statistics.

    Args:
        path: a file written by the LinUCB notebook's checkpointing cell.
        alpha: override the exploration coefficient. Evaluation is greedy, so
            alpha only matters if you intend to keep training.
    """
    from cane.core import LinUCBAgent

    path = Path(path)
    z = np.load(path, allow_pickle=False)
    agent = LinUCBAgent(
        encoder_name=str(z["encoder_name"]),
        alpha=float(z["alpha"]) if alpha is None else float(alpha),
        lam=float(z["lam"]),
        n_actions=int(z["n_actions"]),
        seed=0,
    )
    agent.A = z["A"]
    agent.A_inv = z["A_inv"]
    agent.b = z["b"]
    agent.n_pulls = z["n_pulls"]
    return agent


def load_deep(path: str | Path, family: str | None = None):
    """Rebuild a DQN, Double DQN or PPO agent from a .pt checkpoint.

    The checkpoint carries the hyperparameter dict it was trained under, so the
    reconstructed agent matches the trained one rather than the current default
    config -- which matters, because the hyperparameter search writes agents
    whose settings differ from `*_CONFIG`.
    """
    import torch

    from cane.agents_deep import DQNAgent, DDQNAgent, PPOAgent

    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    family = family or _infer_family(path, blob)

    cfg = dict(blob.get("cfg", {}))
    seed = int(blob.get("seed", 0))

    if family == "PPO":
        agent = PPOAgent(seed=seed, **cfg)
        agent.net.load_state_dict(blob["net"])
        agent.net.eval()
    elif family in {"DQN", "DDQN"}:
        cls = DQNAgent if family == "DQN" else DDQNAgent
        agent = cls(seed=seed, **cfg)
        agent.online_net.load_state_dict(blob["online_net"])
        agent.target_net.load_state_dict(blob["target_net"])
        agent.online_net.eval()
        agent.target_net.eval()
    else:
        raise ValueError(f"unknown agent family: {family!r}")

    expected = int(blob.get("d_in", agent.d_in))
    if expected != agent.d_in:
        raise ValueError(
            f"{path.name} was trained on a {expected}-dimensional state but this "
            f"build of cane produces {agent.d_in}. The observation layout "
            f"changed since the checkpoint was written; retrain or check out the "
            f"matching commit."
        )

    agent.n_updates = int(blob.get("n_updates", 0))
    agent.total_steps = int(blob.get("total_steps", 0))
    return agent


def load_agent(path: str | Path, family: str | None = None):
    """Load any saved CANE agent, dispatching on file type."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix in _TORCH_SUFFIXES:
        return load_deep(path, family=family)
    if path.suffix == ".npz":
        return load_linucb(path)
    raise ValueError(f"unrecognised checkpoint type: {path.suffix}")


def find_checkpoints(root: str | Path = ".") -> dict[str, list[Path]]:
    """Index every saved checkpoint under `root`, grouped by agent family.

    The four algorithms were developed in separate folders, so their models are
    split across `Code/models/` and `DQN/models/`. This gathers them into one
    view for the ensemble and the dashboard.
    """
    root = Path(root)
    found: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix not in _TORCH_SUFFIXES | {".npz"}:
            continue
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        stem = path.stem.lower()
        if stem.startswith("linucb"):
            family = "LinUCB"
        elif stem.startswith("ddqn"):
            family = "DDQN"
        elif stem.startswith("dqn"):
            family = "DQN"
        elif stem.startswith("ppo"):
            family = "PPO"
        else:
            continue
        found.setdefault(family, []).append(path)
    return found
