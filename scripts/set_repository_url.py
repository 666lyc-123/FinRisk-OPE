"""Replace the camera-ready repository URL placeholder once the GitHub repo exists."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    if not args.url.startswith("https://github.com/"):
        raise SystemExit("Expected an HTTPS GitHub repository URL")
    path = ROOT / "paper_source" / "main.tex"
    text = path.read_text(encoding="utf-8")
    if "PUBLIC_REPOSITORY_URL" not in text:
        raise SystemExit("PUBLIC_REPOSITORY_URL placeholder not found")
    path.write_text(text.replace("PUBLIC_REPOSITORY_URL", args.url), encoding="utf-8")
    print("Updated", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
