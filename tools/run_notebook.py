"""Execute a Jupyter notebook headlessly, without a Jupyter install.

The four CANE notebooks are plain top-to-bottom scripts once the markdown is
stripped, so we can concatenate their code cells and exec them in one namespace.
Running them this way lets us smoke-test with QUICK=True and then commit to the
full run, and it keeps the evaluation sweep scriptable from `run_all.py`.

    python tools/run_notebook.py "Code/doubleDQN.ipynb" --quick
    python tools/run_notebook.py "Code/doubleDQN.ipynb" --until 32

Notes
-----
* cwd is switched to the notebook's own directory, because every notebook writes
  to relative `figures/`, `models/` and `results/` paths.
* matplotlib is forced to the Agg backend so `plt.show()` is a no-op.
* `--quick` rewrites the `QUICK = False` line in place (in memory only; the
  notebook file on disk is never modified).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

PREAMBLE = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as _plt\n"
    "_plt.show = lambda *a, **k: None\n"
)


def load_code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf8"))
    return [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "".join(c["source"]).strip()
    ]


def apply_quick(src: str) -> str:
    return re.sub(r"^QUICK\s*=\s*False", "QUICK = True", src, flags=re.MULTILINE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook")
    ap.add_argument("--quick", action="store_true", help="force QUICK = True")
    ap.add_argument("--until", type=int, default=None,
                    help="stop after this notebook cell index (inclusive)")
    ap.add_argument("--skip", type=int, nargs="*", default=[],
                    help="notebook cell indices to skip")
    args = ap.parse_args()

    nb_path = Path(args.notebook).resolve()
    if not nb_path.is_file():
        print(f"error: no such notebook: {nb_path}", file=sys.stderr)
        return 2

    raw = json.loads(nb_path.read_text(encoding="utf8"))
    # Keep the original notebook indices so --until/--skip match what the
    # user sees in Jupyter, not a code-cell-only renumbering.
    cells = [
        (i, "".join(c["source"]))
        for i, c in enumerate(raw["cells"])
        if c["cell_type"] == "code" and "".join(c["source"]).strip()
    ]

    os.chdir(nb_path.parent)
    ns: dict = {"__name__": "__main__", "__file__": str(nb_path)}
    exec(compile(PREAMBLE, "<preamble>", "exec"), ns)

    t0 = time.time()
    for idx, src in cells:
        if idx in args.skip:
            print(f"[cell {idx:3d}] skipped", flush=True)
            continue
        if args.until is not None and idx > args.until:
            break
        if args.quick:
            src = apply_quick(src)
        t1 = time.time()
        print(f"[cell {idx:3d}] running...", flush=True)
        try:
            exec(compile(src, f"<cell {idx}>", "exec"), ns)
        except Exception:
            print(f"[cell {idx:3d}] FAILED after {time.time() - t1:.1f}s", flush=True)
            traceback.print_exc()
            return 1
        print(f"[cell {idx:3d}] ok ({time.time() - t1:.1f}s)", flush=True)

    print(f"\nNotebook completed in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
