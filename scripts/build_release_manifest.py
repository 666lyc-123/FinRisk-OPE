"""Write SHA-256 hashes for the reorganized public repository files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ("src", "data", "paper_source", "scripts", "tests")
INCLUDE_FILES = ("README.md", "LICENSE", ".gitignore")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    files = [ROOT / name for name in INCLUDE_FILES if (ROOT / name).is_file()]
    for directory in INCLUDE_DIRS:
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix != ".pyc")
    payload = {
        "schema": 1,
        "description": "Hashes for the reorganized FinRisk-OPE public repository",
        "sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted(files)
        },
    }
    output = ROOT / "artifacts" / "release_manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Wrote", output, "with", len(files), "files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
