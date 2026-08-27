"""Check every agent config in cane/ against the notebook that owns it.

The notebooks are the source of truth: each is self-contained and each team
member tunes their own. `cane/` is the extracted copy the dashboard studies and
`run_all.py` drive, and it can silently fall behind -- which it did, for both
value-based agents at once, so every study kept reproducing a never-send
collapse the notebooks had already fixed.

`parity_check.py` could not catch this: it compares the deterministic BASELINE
rows, which do not depend on any learner's hyperparameters. This does.

    .venv/Scripts/python.exe tools/config_parity.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# config name -> the notebook that owns it
OWNERS = {
    "DQN_CONFIG":    ROOT / "DQN" / "AssignmentCode(DQN).ipynb",
    "DDQN_CONFIG":   ROOT / "Code" / "doubleDQN.ipynb",
    "PPO_CONFIG":    ROOT / "Code" / "AssignementCode(PPO).ipynb",
    "LINUCB_CONFIG": ROOT / "Code" / "AssignementCode(LinUCB).ipynb",
}
PKG = ROOT / "cane" / "agents_deep.py"


def literal_config(text: str, name: str) -> dict | None:
    """Evaluate `NAME = dict(...)` literally, or return None if absent."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == name for t in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        try:
            return {kw.arg: ast.literal_eval(kw.value)
                    for kw in node.value.keywords}
        except ValueError:
            return None
    return None


def from_notebook(path: Path, name: str) -> dict | None:
    if not path.is_file():
        return None
    nb = json.loads(path.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if f"{name} = dict(" in src:
            got = literal_config(src, name)
            if got is not None:
                return got
    return None


def main() -> int:
    pkg_text = PKG.read_text(encoding="utf-8")
    bad = 0

    for name, nb_path in OWNERS.items():
        nb_cfg = from_notebook(nb_path, name)
        pkg_cfg = literal_config(pkg_text, name)

        if nb_cfg is None and pkg_cfg is None:
            print(f"--   {name:14} not defined in either place")
            continue
        if nb_cfg is None:
            print(f"--   {name:14} not found in {nb_path.name}; "
                  f"package-only, nothing to compare")
            continue
        if pkg_cfg is None:
            print(f"--   {name:14} not in cane/agents_deep.py; "
                  f"notebook-only, nothing to compare")
            continue

        diffs = [k for k in set(nb_cfg) | set(pkg_cfg)
                 if nb_cfg.get(k) != pkg_cfg.get(k)]
        if diffs:
            bad += 1
            print(f"DIFF {name:14} {len(diffs)} key(s) differ from "
                  f"{nb_path.name}")
            for k in sorted(diffs):
                print(f"       {k:22} notebook={nb_cfg.get(k)!r:>12}  "
                      f"cane={pkg_cfg.get(k)!r}")
        else:
            print(f"ok   {name:14} matches {nb_path.name} "
                  f"({len(nb_cfg)} keys)")

    print()
    if bad:
        print(f"FAIL: {bad} config(s) have drifted. Any study driven through "
              f"cane/ is\n      training a different agent than the notebook "
              f"of the same name.")
    else:
        print("PASS: cane/ and the notebooks describe the same agents.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
