import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("submission_review", ROOT / "tools/review.py")
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)
PACKAGE_SPEC = importlib.util.spec_from_file_location("submission_package", ROOT / "tools/package_submission.py")
PACKAGE = importlib.util.module_from_spec(PACKAGE_SPEC)
PACKAGE_SPEC.loader.exec_module(PACKAGE)


class IntegrityTests(unittest.TestCase):
    def test_manifest_rejects_corruption_and_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "record.json"; path.write_text('{"status":"original"}')
            index = {"record.json": REVIEW.sha256(path)}
            self.assertEqual(REVIEW.verify_manifest(root, index), 1)
            path.write_text('{"status":"changed"}')
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                REVIEW.verify_manifest(root, index)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                REVIEW.verify_manifest(root, {"../outside": "unused"})

    def test_numeric_comparison_rejects_changed_result(self):
        REVIEW.same({"accuracy": .6}, {"accuracy": .6}, "fixture")
        with self.assertRaisesRegex(ValueError, "Different measurement"):
            REVIEW.same({"accuracy": .61}, {"accuracy": .6}, "fixture")

    def test_notebooks_retain_execution_and_match_recorded_runner(self):
        training = json.loads((ROOT / "notebooks/02_train_and_verify_soup_t4.executed.ipynb").read_text())
        source = (ROOT / "run/verify_artifacts.py").read_text()
        embedded = None
        executed = 0
        for cell in training["cells"]:
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            tree = ast.parse(text)
            executed += cell.get("execution_count") is not None
            self.assertFalse(any(o["output_type"] == "error" for o in cell.get("outputs", [])))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RUNTIME_SOURCE" for t in node.targets):
                    embedded = ast.literal_eval(node.value)
        self.assertEqual(embedded, source)
        self.assertEqual(executed, 15)

    def test_report_is_two_pages_and_contains_the_failed_decision(self):
        pages = PdfReader(ROOT / "report.pdf").pages
        self.assertEqual(len(pages), 2)
        text = "\n".join(page.extract_text() for page in pages)
        for required in ("DON'T SHIP", "0.03867", "49.33%", "8.993", "Reflection and AI use"):
            self.assertIn(required, text)
        self.assertNotIn("PENDING", text)

    def test_package_round_trip_preserves_bytes_and_excludes_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"; root.mkdir()
            (root / "raw.log").write_bytes(b"original\r\nbytes\n")
            (root / ".git").mkdir(); (root / ".git/config").write_text("exclude")
            result = PACKAGE.build(Path(directory) / "submission.zip", root)
            self.assertEqual(result["files"], 2)
            self.assertEqual(len(result["sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "outside"):
                PACKAGE.build(root / "recursive.zip", root)

    def test_original_evidence_replays_the_recorded_negative_verdict(self):
        result = REVIEW.review(ROOT / "run", ROOT / "data-preparation")
        self.assertEqual(result["review_status"], "PASS")
        self.assertEqual(result["release_verdict"], "DON'T SHIP")
        self.assertEqual(result["original_run_files_hash_verified"], 215)
        self.assertEqual(result["preparation_files_hash_verified"], 53)
        self.assertEqual(result["test_predictions_rescored"], 450)
        self.assertEqual(result["transitions"]["M_to_Q"]["lost_correct"], 22)
        self.assertEqual(result["transitions"]["M_to_Q"]["gained_correct"], 6)
        self.assertEqual(result["bootstrap_comparisons_reproduced"], 3)


if __name__ == "__main__":
    unittest.main()
