"""Run the full FinRisk-OPE experiment, export evidence, and verify it."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/results_rerun"))
    parser.add_argument("--assets", type=Path, default=Path("artifacts/reproduced"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    assets = args.assets if args.assets.is_absolute() else ROOT / args.assets
    if output.exists() or assets.exists():
        raise SystemExit("Output paths must not already exist: " + str(output) + ", " + str(assets))
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "directional_validation" / "run_directional.py"),
         "--out", str(output)], check=True, cwd=ROOT)
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "export_paper_evidence.py"),
         "--run", str(output), "--paper", str(assets)], check=True, cwd=ROOT)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_release.py"),
         "--run", str(output), "--assets", str(assets)], check=True, cwd=ROOT)
    print("Reproduction verified:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
