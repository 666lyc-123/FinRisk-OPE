"""Fast structural checks; these do not rerun the full benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_required_root_layout(self):
        for name in ("README.md", "LICENSE", "src", "data", "outputs",
                     "artifacts", "paper_source", "scripts", "tests"):
            self.assertTrue((ROOT / name).exists(), name)

    def test_reference_counts(self):
        manifest = json.loads((ROOT / "outputs" / "reference_run" / "manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(manifest["case_rows"], 50784)
        self.assertEqual(manifest["tasks"], 15)
        self.assertEqual(manifest["cells"], 2068330)

    def test_camera_ready_url_is_real(self):
        paper = (ROOT / "paper_source" / "main.tex").read_text(encoding="utf-8")
        self.assertIn("https://github.com/666lyc-123/FinRisk-OPE", paper)
        self.assertNotIn("PUBLIC_REPOSITORY_URL", paper)
        self.assertIn("149 of its 249", paper)
        self.assertIn("transaction costs and market impact", paper)


if __name__ == "__main__":
    unittest.main()
