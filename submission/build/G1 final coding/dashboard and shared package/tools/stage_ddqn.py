"""Stage the executed Double DQN notebook into the submission build.

Run this once Xin Yi's notebook run has finished and been saved:

    .venv/Scripts/python.exe tools/stage_ddqn.py

It refuses to stage a run that is not fit to submit, because the two ways this
notebook has failed before are both silent in the staged artifact:

* a partial save -- the run was interrupted and only some cells carry outputs,
  which renders to a PDF that simply stops halfway;
* a collapse -- the run completes cleanly and the agent contacted 2 of the 5
  archetypes, scoring exactly 0.00 on the rest. Nothing about that is an error;
  it is a valid run of a policy that decided silence was optimal. Only the
  results table shows it.

Both are caught here rather than discovered in the submitted PDF. Use --force to
stage anyway (for instance to submit a known-collapsed run deliberately).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Code" / "doubleDQN.ipynb"
RESULTS = ROOT / "Code" / "results" / "ddqn_results.csv"
DEST_DIR = ROOT / "submission" / "build" / "G1 final coding"
STEM = "G1 final coding 3 - DDQN"

# A send rate below this is silence, not a quiet policy: the wrapper alone
# guarantees 7 sends a week, so anything near zero means it was not applied.
SILENT_SENDS = 0.05
N_ARCHETYPES = 5


def check_executed(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"]
            if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    # execution_count, not outputs: a pure definition cell legitimately runs
    # and prints nothing. An unrun cell is the one with no execution_count.
    missing = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    errored = [i for i, c in enumerate(code)
               if any(o.get("output_type") == "error" for o in c.get("outputs", []))]

    problems = []
    if missing:
        problems.append(
            f"{len(missing)} of {len(code)} code cells never ran "
            f"(first: index {missing[0]}) -- the run did not finish, or the "
            f"notebook was not saved afterwards")
    if errored:
        problems.append(
            f"{len(errored)} code cell(s) recorded an error "
            f"(indices {errored})")
    if not problems:
        counts = [c["execution_count"] for c in code]
        order = "in order" if counts == sorted(counts) else "OUT OF ORDER"
        print(f"  executed : {len(code)}/{len(code)} code cells {order}, no errors")
    return problems


def check_not_collapsed(results: Path) -> list[str]:
    if not results.is_file():
        return [f"{results.relative_to(ROOT)} not found -- did section 8 run?"]

    import pandas as pd

    df = pd.read_csv(results)
    d = df[df["agent"] == "DDQN"]
    if d.empty:
        return [f"no DDQN rows in {results.name}"]

    per = d.groupby("archetype")[["sends_per_episode", "reward_mean"]].mean()
    silent = per[per["sends_per_episode"] < SILENT_SENDS]
    reached = len(per) - len(silent)
    total = per["reward_mean"].sum()

    print(f"  coverage : {reached}/{len(per)} archetypes contacted, "
          f"total reward {total:.2f}")
    for arch, row in per.iterrows():
        mark = "SILENT" if row["sends_per_episode"] < SILENT_SENDS else "      "
        print(f"    {mark} {arch:17} sends {row['sends_per_episode']:6.2f}  "
              f"reward {row['reward_mean']:8.2f}")

    problems = []
    if len(per) < N_ARCHETYPES:
        problems.append(f"only {len(per)} archetypes in the results, "
                        f"expected {N_ARCHETYPES}")
    if len(silent):
        problems.append(
            "the agent still collapses on "
            + ", ".join(silent.index)
            + " -- check that section 8.1 ran and MIN_SENDS is 1")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="stage even if the checks fail")
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()

    if not NB.is_file():
        sys.exit(f"{NB} not found")

    print(f"checking {NB.relative_to(ROOT)}")
    problems = check_executed(NB) + check_not_collapsed(RESULTS)

    if problems:
        print("\nNOT READY TO STAGE:")
        for p in problems:
            print(f"  - {p}")
        if not args.force:
            print("\nFix the run and re-save, or pass --force to stage anyway.")
            return 1
        print("\n--force given; staging regardless.")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = DEST_DIR / f"{STEM}.ipynb"
    shutil.copy2(NB, dest)
    print(f"\nstaged {dest.relative_to(ROOT)}")

    if not args.skip_pdf:
        cmd = [sys.executable, str(ROOT / "tools" / "nb_to_pdf.py"), str(NB),
               "--out-dir", str(DEST_DIR), "--name", STEM]
        if args.force:
            cmd.append("--allow-empty")
        print("rendering PDF ...")
        if subprocess.run(cmd, cwd=ROOT).returncode:
            print("PDF rendering FAILED -- the .ipynb is staged, the PDF is not")
            return 1
        print(f"staged {(DEST_DIR / (STEM + '.pdf')).relative_to(ROOT)}")

    print("\nverification:")
    for script in ("check_results.py", "parity_check.py"):
        print(f"  --- {script} ---")
        subprocess.run([sys.executable, str(ROOT / "tools" / script)], cwd=ROOT)

    print("\nAll four final coding files are now staged. Remaining for item 1: "
          "nothing. See submission/CHECKLIST.md for items 2-8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
