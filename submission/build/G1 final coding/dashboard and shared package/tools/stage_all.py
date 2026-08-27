"""Stage all four final coding notebooks into the submission build.

    .venv/Scripts/python.exe tools/stage_all.py

Copies each executed notebook under its submission name and re-renders its PDF.
Refuses to stage a notebook that did not finish running, because a partial save
renders to a PDF that simply stops halfway and nothing about the file says so.

`tools/stage_ddqn.py` remains the Double DQN-specific entry point: it also
checks that the agent did not collapse to never-send. This script is the
whole-item-1 pass for when every notebook needs restaging at once.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "submission" / "build" / "G1 final coding"

NOTEBOOKS = [
    (ROOT / "Code" / "AssignementCode(LinUCB).ipynb",
     "G1 final coding 1 - LinUCB and shared environment"),
    (ROOT / "DQN" / "AssignmentCode(DQN).ipynb",
     "G1 final coding 2 - DQN"),
    (ROOT / "Code" / "doubleDQN.ipynb",
     "G1 final coding 3 - DDQN"),
    (ROOT / "Code" / "AssignementCode(PPO).ipynb",
     "G1 final coding 4 - PPO"),
]


def unrun(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"]
            if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    # execution_count, not outputs: a definition cell legitimately prints nothing
    missing = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    errored = [i for i, c in enumerate(code)
               if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    out = []
    if missing:
        out.append(f"{len(missing)}/{len(code)} code cells never ran "
                   f"(first index {missing[0]})")
    if errored:
        out.append(f"{len(errored)} code cell(s) recorded an error {errored}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    failed = 0

    for src, stem in NOTEBOOKS:
        if not src.is_file():
            print(f"MISSING  {src}")
            failed += 1
            continue

        problems = unrun(src)
        if problems and not args.force:
            print(f"SKIP     {stem}")
            for p in problems:
                print(f"           {p}")
            failed += 1
            continue

        shutil.copy2(src, DEST / f"{stem}.ipynb")
        note = "  (forced)" if problems else ""
        print(f"staged   {stem}.ipynb{note}")

        if args.skip_pdf:
            continue
        cmd = [sys.executable, str(ROOT / "tools" / "nb_to_pdf.py"), str(src),
               "--out-dir", str(DEST), "--name", stem]
        if args.force:
            cmd.append("--allow-empty")
        if subprocess.run(cmd, cwd=ROOT,
                          stdout=subprocess.DEVNULL).returncode:
            print(f"         PDF FAILED for {stem}")
            failed += 1
        else:
            kb = (DEST / f"{stem}.pdf").stat().st_size // 1024
            print(f"         PDF {kb} KB")

    print()
    print(f"{len(NOTEBOOKS) - failed}/{len(NOTEBOOKS)} notebooks staged")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
