"""Render an executed notebook to PDF, without LaTeX.

`nbconvert --to pdf` needs a full TeX install and `--to webpdf` needs a
playwright chromium download; both are large and neither is present here. The
machine already has Chrome, and Chrome's headless `--print-to-pdf` renders the
same HTML nbconvert produces, so this goes notebook -> HTML -> PDF through it.

The submission guidelines ask for the coding files in both `.ipynb` and PDF, so
this runs once per notebook at the end of the pipeline.

    python tools/nb_to_pdf.py "Code/doubleDQN.ipynb" --out-dir "submission/build"

A notebook with no saved outputs produces a PDF of bare source, which is not
what the guidelines ask for -- so that is refused rather than silently shipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHROMES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def find_browser() -> Path:
    for p in CHROMES:
        if p.is_file():
            return p
    raise SystemExit("no Chrome or Edge found for PDF rendering")


def output_count(nb_path: Path) -> tuple[int, int]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"
            and "".join(c["source"]).strip()]
    return sum(1 for c in code if c.get("outputs")), len(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--name", default=None,
                    help="basename for the PDF; defaults to the notebook's")
    ap.add_argument("--allow-empty", action="store_true",
                    help="render even when the notebook has no saved outputs")
    args = ap.parse_args()

    nb = Path(args.notebook).resolve()
    if not nb.is_file():
        raise SystemExit(f"not found: {nb}")

    have, total = output_count(nb)
    print(f"{nb.name}: {have}/{total} code cells carry outputs")
    if have == 0 and not args.allow_empty:
        raise SystemExit(
            "refusing: no saved outputs. Execute it first with\n"
            "  python -m jupyter nbconvert --to notebook --execute --inplace "
            "--ExecutePreprocessor.kernel_name=cane "
            f'"{args.notebook}"\n'
            "or pass --allow-empty if a source-only PDF really is what you want.")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / ((args.name or nb.stem) + ".pdf")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        html = tmp / "nb.html"
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
             "--output", html.stem, "--output-dir", str(tmp), str(nb)],
            check=True, capture_output=True)

        browser = find_browser()
        subprocess.run(
            [str(browser), "--headless", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
             "--virtual-time-budget=20000",
             f"--print-to-pdf={pdf}", html.as_uri()],
            check=True, capture_output=True, timeout=300)

    if not pdf.is_file() or pdf.stat().st_size < 2000:
        raise SystemExit(f"PDF looks empty: {pdf}")
    print(f"wrote {pdf}  ({pdf.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
