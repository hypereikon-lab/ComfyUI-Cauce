import json
from pathlib import Path
import tempfile
import unittest

from cauce.artifacts import read_json, write_json_atomic
from cauce.profiles import get_profile, model_manifest, preflight
from cauce.runner import materialize, run_project


class RunnerAndProfileTests(unittest.TestCase):
    def test_materialize_preserves_native_values_for_exact_placeholder(self):
        template = {
            "1": {
                "inputs": {
                    "length": "{{window.frames}}",
                    "label": "clip-{{window.id}}",
                }
            }
        }
        result = materialize(template, {"window": {"frames": 124, "id": "001"}})
        self.assertEqual(result["1"]["inputs"]["length"], 124)
        self.assertEqual(result["1"]["inputs"]["label"], "clip-001")

    def test_dry_run_is_restartable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(
                root / "workflow.json",
                {"1": {"class_type": "Example", "inputs": {"seed": "{{window.seed}}"}}},
            )
            write_json_atomic(
                root / "project.json",
                {
                    "schema": "cauce.project/1",
                    "workflow_template": "workflow.json",
                    "windows": [
                        {"id": "w1", "seed": 1},
                        {"id": "w2", "seed": 2},
                    ],
                },
            )
            first = run_project(root / "project.json", dry_run=True, once=True)
            self.assertEqual(first["windows"]["w1"]["status"], "materialized")
            second = run_project(root / "project.json", dry_run=True)
            self.assertEqual(second["windows"]["w2"]["status"], "materialized")
            self.assertTrue((root / ".cauce/runs/w1.workflow.json").exists())

    def test_preflight_is_non_mutating_and_reports_missing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = get_profile("h3-5090-fl2va-640")
            report = preflight(profile, directory, inspect_torch=False)
            self.assertFalse(report["ready"])
            self.assertEqual(len(report["missing"]), 4)
            self.assertEqual(len(model_manifest(profile)), 4)

    def test_preflight_does_not_create_a_missing_models_root(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-created" / "models"
            profile = get_profile("h3-5090-fl2va-640")
            report = preflight(profile, missing, inspect_torch=False)
            self.assertFalse(missing.exists())
            self.assertFalse(report["models_root_exists"])
            self.assertFalse(report["ready"])


if __name__ == "__main__":
    unittest.main()
