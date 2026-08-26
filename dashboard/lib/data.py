"""Cached loaders for everything the dashboard displays.

Two design rules, both of which exist because this dashboard has to survive a
live graded demo:

1. **Never raise on a missing artifact.** The four algorithms are produced by
   four separate notebooks, so at any moment some result files may not exist
   yet. Each loader returns an empty frame with the right columns and records
   what was missing, and the pages render what they have with an honest note
   about the rest. A dashboard that shows four of five archetypes is useful; one
   that shows a stack trace is not.

2. **Nothing here imports torch.** The CSV-driven pages must work on a machine
   where the deep-learning stack is unavailable or broken. Only the pages that
   genuinely need live inference import it, and they import it themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

# Where each algorithm's notebook writes its results. The split between Code/
# and DQN/ is historical -- the four algorithms were developed in separate
# folders -- so the loader gathers them rather than assuming one location.
RESULT_FILES = {
    "LinUCB": ROOT / "Code" / "results" / "linucb_results.csv",
    "DDQN":   ROOT / "Code" / "results" / "ddqn_results.csv",
    "PPO":    ROOT / "Code" / "results" / "ppo_results.csv",
    "DQN":    ROOT / "DQN" / "results" / "dqn_results.csv",
}

ARTIFACTS = ROOT / "artifacts"

SUMMARY_COLS = ["archetype", "agent", "seed", "reward_mean",
                "ctr", "sends_per_episode", "optout_rate"]

BASELINES = ["Fixed-18:00", "Random"]


@st.cache_data(show_spinner=False)
def load_summary() -> tuple[pd.DataFrame, list[str]]:
    """Every algorithm's per-seed results, concatenated.

    The baselines appear once per source file (each notebook re-evaluates them),
    so they are de-duplicated here -- keeping every copy would make Fixed-18:00
    look four times as certain as it is. They are verified identical across
    files by `tools/check_results.py`; that identity is the evidence the four
    algorithms shared one evaluation protocol.

    Returns (frame, missing) where `missing` names the algorithms whose result
    file does not exist yet.
    """
    frames, missing = [], []
    for name, path in RESULT_FILES.items():
        if not path.is_file():
            missing.append(name)
            continue
        df = pd.read_csv(path)
        df["source"] = name
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLS + ["source"]), list(RESULT_FILES)

    out = pd.concat(frames, ignore_index=True)
    learners = out[~out["agent"].isin(BASELINES)]
    baselines = (out[out["agent"].isin(BASELINES)]
                 .drop_duplicates(subset=["archetype", "agent", "seed"]))
    out = pd.concat([learners, baselines], ignore_index=True)
    return out, missing


@st.cache_data(show_spinner=False)
def load_ensemble() -> pd.DataFrame:
    path = ARTIFACTS / "ensemble.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["archetype", "scheme", "member", "split",
                                     "hold_bias", "reward_mean", "ctr",
                                     "sends_per_episode", "optout_rate"])
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_bias_sweep() -> pd.DataFrame:
    path = ARTIFACTS / "ensemble_bias_sweep.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["archetype", "hold_bias", "reward_mean",
                                     "ctr", "sends_per_episode", "optout_rate"])
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_ensemble_meta() -> dict[str, dict]:
    """Per-archetype mixture weights, gate temperatures and chosen hold bias.

    Keyed by archetype so a page can look up one without scanning. Returns {}
    when the study has not been run, which every caller must tolerate: the
    ensemble artifacts are produced by a separate script, not by the notebooks.
    """
    path = ARTIFACTS / "ensemble_meta.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf8"))
    return {entry["archetype"]: entry for entry in raw}


@st.cache_data(show_spinner=False)
def load_figures() -> dict[str, list[Path]]:
    """Index the PNGs each notebook produced, keyed by owning folder.

    Keyed by folder rather than filename on purpose: `Code/figures/` and
    `DQN/figures/` both contain a `D1_ddqn_learning_curve.png`, for *different*
    algorithms. A flat filename index would silently show one in place of the
    other.
    """
    out: dict[str, list[Path]] = {}
    for folder, label in ((ROOT / "Code" / "figures", "Code"),
                          (ROOT / "DQN" / "figures", "DQN")):
        if folder.is_dir():
            out[label] = sorted(folder.glob("*.png"))
    return out


def leaderboard(df: pd.DataFrame, archetype: str | None = None) -> pd.DataFrame:
    """Mean and 95% CI per agent, aggregated over seeds.

    The interval uses a normal approximation over the seed means. It is
    deliberately shown even when it is wide: with five seeds a wide interval is
    the honest reading, and collapsing it to a bare mean would overstate how
    separated the algorithms are.
    """
    if df.empty:
        return df
    d = df if archetype in (None, "All") else df[df["archetype"] == archetype]
    if d.empty:
        return pd.DataFrame()

    g = (d.groupby("agent")
           .agg(reward=("reward_mean", "mean"),
                sd=("reward_mean", "std"),
                n=("reward_mean", "size"),
                ctr=("ctr", "mean"),
                sends=("sends_per_episode", "mean"),
                optout=("optout_rate", "mean"))
           .reset_index())
    g["sd"] = g["sd"].fillna(0.0)
    sem = g["sd"] / g["n"].clip(lower=1) ** 0.5
    g["ci_low"] = g["reward"] - 1.96 * sem
    g["ci_high"] = g["reward"] + 1.96 * sem
    return g.sort_values("reward", ascending=False).reset_index(drop=True)


def break_even_ctr(config: dict, action: int = 1) -> float:
    """Click rate at which one send exactly pays for itself.

    A send costs its immediate send weight plus the discounted fatigue it
    creates. Fatigue decays geometrically at `lam`, so a single send of size
    `kappa` adds a discounted debt of W_fat * kappa / (1 - lam) spread over all
    future steps. The send is worth making only when

        R_click * p  >=  W_send + W_fat * kappa / (1 - lam)

    This one line explains the project's central result: with the shipped
    weights the threshold sits well above most archetypes' peak click
    propensity, so a correctly-reasoning agent declines to send at all.
    """
    w_send = config["W_send"][action]
    kappa = config["kappa"][action]
    debt = config["W_fat"] * kappa / (1.0 - config["lam"])
    return (w_send + debt) / config["R_click"]
