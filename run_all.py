"""One command that runs the whole CANE evaluation, end to end.

The four algorithms were developed as four hand-run notebooks. That is fine for
developing them and hopeless for reproducing them: the results only agree if
every notebook is run against the same config, in the right order, with the
hyperparameter search kept out of the path that writes the results CSV. This
script encodes that order so it does not have to be remembered.

    python run_all.py                     # everything
    python run_all.py --quick             # fast smoke pass, minutes not hours
    python run_all.py --list              # show the stages and stop
    python run_all.py --only protocol parity
    python run_all.py --skip notebooks    # reuse CSVs that already exist

Every stage writes its own log under `logs/`, and a failing stage does not stop
the ones after it -- a broken ensemble study should not hide a passing parity
check. The exit code is non-zero if any stage failed, so CI or a marker can
treat it as a single pass/fail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():  # non-Windows checkout, or a bare interpreter
    PY = sys.executable

# Globbed rather than listed. A hard-coded list silently stops checking a page
# the moment one is added, and silently fails on one that is removed -- both of
# which happened before this was changed.
DASHBOARD_PAGES = ["dashboard/app.py"] + [
    f"dashboard/pages/{p.name}"
    for p in sorted((ROOT / "dashboard" / "pages").glob("*.py"))
]


@dataclass
class Stage:
    """One step of the pipeline.

    `commands` is a list because a stage can be several processes that must all
    succeed (the dashboard check is one per page). `parallel` marks the stages
    whose commands are independent -- the two notebook runs are the only ones,
    and running them together roughly halves the wall clock.
    """

    name: str
    summary: str
    commands: list[list[str]]
    parallel: bool = False
    quick_commands: list[list[str]] | None = None
    log: str = field(default="")

    def __post_init__(self):
        self.log = self.log or f"{self.name}.log"

    def resolved(self, quick: bool) -> list[list[str]]:
        if quick and self.quick_commands is not None:
            return self.quick_commands
        return self.commands


def build_stages() -> list[Stage]:
    return [
        Stage(
            name="notebooks",
            summary="train LinUCB and Double DQN, write their result CSVs",
            parallel=True,
            commands=[
                [PY, "-u", "tools/run_notebook.py",
                 "Code/AssignementCode(LinUCB).ipynb"],
                # Cell 41 is the 20-trial hyperparameter search. It runs in its
                # own stage below: leaving it here would put the search in front
                # of the CSV that every later stage waits on. Indices are
                # *whole-notebook* indices, markdown included -- run_notebook.py
                # keeps them that way so they match what you see in Jupyter.
                [PY, "-u", "tools/run_notebook.py", "Code/doubleDQN.ipynb",
                 "--skip", "41"],
            ],
            quick_commands=[
                [PY, "-u", "tools/run_notebook.py",
                 "Code/AssignementCode(LinUCB).ipynb", "--quick"],
                [PY, "-u", "tools/run_notebook.py", "Code/doubleDQN.ipynb",
                 "--quick", "--skip", "41"],
            ],
        ),
        Stage(
            name="protocol",
            summary="check all four CSVs describe the same experiment",
            commands=[[PY, "tools/check_results.py"]],
        ),
        Stage(
            name="parity",
            summary="check cane/ reproduces the notebooks exactly",
            commands=[[PY, "tools/parity_check.py"]],
        ),
        Stage(
            name="search",
            summary="Double DQN hyperparameter search (notebook cell 41)",
            commands=[[
                PY, "-u", "tools/run_notebook.py", "Code/doubleDQN.ipynb",
                "--until", "41",
                # Skip the training and evaluation cells: this stage only needs
                # the search, and re-running the sweep here would overwrite the
                # results CSV the protocol check just validated.
                "--skip", "12", "13", "14", "15", "16", "31", "32", "34", "35",
                "36", "38", "39",
            ]],
        ),
        Stage(
            name="exploration",
            summary="five-variant diagnosis of the never-send collapse (RQ1)",
            commands=[[PY, "-u", "-m", "cane.exploration_study"]],
            quick_commands=[[PY, "-u", "-m", "cane.exploration_study", "--quick"]],
        ),
        Stage(
            name="rescue",
            summary="seven interventions against the never-send collapse",
            commands=[
                [PY, "-u", "tools/preseed_study.py"],
                [PY, "-u", "tools/ppo_fix.py"],
                [PY, "-u", "tools/tune_study.py"],
            ],
            quick_commands=[
                [PY, "-u", "tools/preseed_study.py", "--episodes", "80",
                 "--archetypes", "OfficeWorker", "Housewife"],
                [PY, "-u", "tools/ppo_fix.py", "--episodes", "80",
                 "--archetypes", "OfficeWorker", "Housewife"],
                [PY, "-u", "tools/tune_study.py", "--episodes", "80",
                 "--archetypes", "OfficeWorker", "Housewife"],
            ],
        ),
        Stage(
            name="personalisation",
            summary="which hour each agent picks per person (RQ3)",
            commands=[
                [PY, "-u", "tools/personalisation_test.py"],
                [PY, "-u", "tools/rq3_single_policy.py"],
            ],
            quick_commands=[
                [PY, "-u", "tools/personalisation_test.py", "--episodes", "80"],
                [PY, "-u", "tools/rq3_single_policy.py", "--episodes", "80",
                 "--seeds", "1"],
            ],
        ),
        Stage(
            name="curves",
            summary="learning curves and convergence speed",
            commands=[
                [PY, "-u", "tools/learning_curves.py"],
                # A second pass under each algorithm's winning configuration.
                # The default-settings curves mostly sit at zero, which says
                # what the shipped hyperparameters do but nothing about what
                # the algorithms are capable of.
                [PY, "-u", "tools/learning_curves.py", "--tuned",
                 "--out-suffix", "_tuned", "--episodes", "1500",
                 "--every", "50"],
            ],
            quick_commands=[[PY, "-u", "tools/learning_curves.py",
                             "--episodes", "100", "--every", "25",
                             "--archetypes", "OfficeWorker", "Housewife"]],
        ),
        Stage(
            name="demo",
            summary="per-archetype checkpoints for the live simulation",
            commands=[[PY, "-u", "tools/train_demo_agents.py",
                       "--variant", "tuned"]],
            quick_commands=[[PY, "-u", "tools/train_demo_agents.py",
                             "--variant", "plain", "--episodes", "80"]],
        ),
        Stage(
            name="ensemble",
            summary="fit and evaluate both ensemble schemes, per archetype",
            commands=[[PY, "-u", "-m", "cane.ensemble_study"]],
            quick_commands=[[PY, "-u", "-m", "cane.ensemble_study", "--quick"]],
        ),
        Stage(
            name="dashboard",
            summary="render every Streamlit page headlessly",
            commands=[[PY, "tools/render_check.py", page]
                      for page in DASHBOARD_PAGES],
        ),
    ]


def run_stage(stage: Stage, quick: bool) -> tuple[bool, float]:
    """Run one stage, streaming nothing and logging everything.

    Output goes to a file rather than the terminal on purpose: the full run is
    hours long and the interesting part is the per-stage verdict, not the
    training chatter. The log path is printed on failure.
    """
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / stage.log
    commands = stage.resolved(quick)
    started = time.time()

    with log_path.open("w", encoding="utf8") as fh:
        fh.write(f"$ {' '.join(commands[0])}\n" if len(commands) == 1 else "")
        if stage.parallel:
            procs = []
            for cmd in commands:
                fh.write(f"$ {' '.join(cmd)}\n")
                fh.flush()
                procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=fh,
                                              stderr=subprocess.STDOUT))
            codes = [p.wait() for p in procs]
        else:
            codes = []
            for cmd in commands:
                if len(commands) > 1:
                    fh.write(f"\n$ {' '.join(cmd)}\n")
                fh.flush()
                codes.append(subprocess.call(cmd, cwd=ROOT, stdout=fh,
                                             stderr=subprocess.STDOUT))

    return all(c == 0 for c in codes), time.time() - started


def fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def main() -> int:
    stages = build_stages()
    names = [s.name for s in stages]

    ap = argparse.ArgumentParser(
        description="Run the full CANE evaluation pipeline.")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke pass; the numbers are NOT reportable")
    ap.add_argument("--only", nargs="+", metavar="STAGE", choices=names,
                    help="run only these stages")
    ap.add_argument("--skip", nargs="+", metavar="STAGE", choices=names,
                    default=[], help="run everything except these stages")
    ap.add_argument("--list", action="store_true",
                    help="print the stages and exit")
    args = ap.parse_args()

    if args.list:
        print("stages, in order:\n")
        for s in stages:
            print(f"  {s.name:<12} {s.summary}")
        return 0

    selected = [s for s in stages
                if (args.only is None or s.name in args.only)
                and s.name not in args.skip]
    if not selected:
        print("nothing to run")
        return 0

    if args.quick:
        print("QUICK MODE - smoke pass only. Do not report these numbers.\n")

    width = 78
    results: list[tuple[str, bool, float]] = []
    t0 = time.time()

    for i, stage in enumerate(selected, 1):
        head = f"[{i}/{len(selected)}] {stage.name}"
        print(f"{head}  {stage.summary}")
        print(f"{'.' * width}")
        ok, elapsed = run_stage(stage, args.quick)
        results.append((stage.name, ok, elapsed))
        verdict = "PASS" if ok else "FAIL"
        print(f"  {verdict}  ({fmt(elapsed)})  -> logs/{stage.log}\n")

    print("=" * width)
    print(f"{'stage':<14}{'result':<10}{'time':>8}")
    print("-" * width)
    for name, ok, elapsed in results:
        print(f"{name:<14}{'PASS' if ok else 'FAIL':<10}{fmt(elapsed):>8}")
    print("-" * width)
    failed = [n for n, ok, _ in results if not ok]
    print(f"{'total':<14}{'':<10}{fmt(time.time() - t0):>8}")
    print("=" * width)

    if failed:
        print(f"\n{len(failed)} stage(s) failed: {', '.join(failed)}")
        print("Read the log named beside each FAIL above.")
        return 1

    print("\nAll stages passed.")
    print("Open the dashboard with:")
    print(f"  {Path(PY).name} -m streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
