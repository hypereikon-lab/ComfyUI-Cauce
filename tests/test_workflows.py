import json
from pathlib import Path
import tempfile
import unittest

from cauce.runner import PLACEHOLDER, materialize, run_project


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "examples" / "workflows"
API_DIR = ROOT / "examples" / "api"


class VisualWorkflowTests(unittest.TestCase):
    def test_shipped_visual_workflow_suite_is_consistent(self):
        expected = {
            "00_plate_sketch_handoff.json",
            "10_h3_fl2va_first_last.json",
            "20_h3_ref2va_motion_reference.json",
            "30_h3_timed_guide.json",
            "40_h3_two_window_continuation.json",
            "50_h3_latent_bridge.json",
        }
        paths = sorted(WORKFLOW_DIR.glob("*.json"))
        self.assertEqual({path.name for path in paths}, expected)
        for path in paths:
            with self.subTest(path=path.name):
                workflow = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(workflow["version"], 0.4)
                self.assertTrue(workflow["groups"])
                nodes = {node["id"]: node for node in workflow["nodes"]}
                self.assertEqual(len(nodes), len(workflow["nodes"]))
                self.assertNotIn("MiniMaxH3MotionContext", {n["type"] for n in nodes.values()})
                self.assertNotIn("HypereikonH3Production", {n["type"] for n in nodes.values()})

                link_ids = set()
                for link in workflow["links"]:
                    self.assertEqual(len(link), 6)
                    link_id, source_id, source_slot, target_id, target_slot, socket_type = link
                    self.assertNotIn(link_id, link_ids)
                    link_ids.add(link_id)
                    self.assertIn(source_id, nodes)
                    self.assertIn(target_id, nodes)
                    source = nodes[source_id]["outputs"][source_slot]
                    target = nodes[target_id]["inputs"][target_slot]
                    self.assertEqual(source["type"], socket_type)
                    self.assertEqual(target["type"], socket_type)
                    self.assertIn(link_id, source["links"])
                    self.assertEqual(target["link"], link_id)
                self.assertEqual(workflow["last_link_id"], max(link_ids, default=0))
                self.assertEqual(workflow["last_node_id"], max(nodes, default=0))

    def test_continuation_uses_decoded_endpoint_and_constant_render_envelope(self):
        workflow = json.loads(
            (WORKFLOW_DIR / "40_h3_two_window_continuation.json").read_text()
        )
        nodes = workflow["nodes"]
        self.assertIn("CauceSelectImageFrame", {node["type"] for node in nodes})
        windows = [
            node for node in nodes if node["type"] == "CauceGenerationWindow"
        ]
        second = next(
            node for node in windows if node["widgets_values"][0] == "continuation_window_b"
        )
        self.assertAlmostEqual(second["widgets_values"][2], 3.541666667)

    def test_ref2va_demo_uses_landscape_profile(self):
        workflow = json.loads(
            (WORKFLOW_DIR / "20_h3_ref2va_motion_reference.json").read_text()
        )
        profile = next(
            node for node in workflow["nodes"] if node["type"] == "CauceExecutionProfile"
        )
        self.assertEqual(profile["widgets_values"], ["h3-5090-ref2va-576x320"])

    def test_demo_assets_exist_and_are_bounded(self):
        assets = ROOT / "examples" / "assets"
        expected = {
            "cauce_forest_a.jpg",
            "cauce_forest_b.jpg",
            "cauce_forest_c.jpg",
            "cauce_motion_reference.mp4",
        }
        self.assertEqual({path.name for path in assets.iterdir()}, expected)
        self.assertLess(sum(path.stat().st_size for path in assets.iterdir()), 5 * 1024**2)


class APIWorkflowTests(unittest.TestCase):
    def test_templates_materialize_without_placeholders(self):
        project = json.loads((ROOT / "examples" / "project.example.json").read_text())
        for window in project["windows"]:
            selected = window.get("workflow_template", project["workflow_template"])
            template = json.loads((ROOT / "examples" / selected).read_text())
            workflow = materialize(template, {"project": project, "window": window})
            encoded = json.dumps(workflow)
            self.assertIsNone(PLACEHOLDER.search(encoded))
            node_ids = set(workflow)
            for node in workflow.values():
                self.assertIn("class_type", node)
                for value in node.get("inputs", {}).values():
                    if (
                        isinstance(value, list)
                        and len(value) == 2
                        and isinstance(value[0], str)
                        and isinstance(value[1], int)
                    ):
                        self.assertIn(value[0], node_ids)

    def test_runner_selects_template_per_window(self):
        project = json.loads((ROOT / "examples" / "project.example.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "api").mkdir()
            for template in API_DIR.glob("*.json"):
                (root / "api" / template.name).write_text(template.read_text())
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project))
            state = run_project(project_path, dry_run=True)
            self.assertEqual(
                state["windows"]["forest_window_001"]["workflow_template"],
                "api/h3_fl2va_window.template.json",
            )
            self.assertEqual(
                state["windows"]["forest_window_002"]["workflow_template"],
                "api/h3_continuation_window.template.json",
            )

    def test_continuation_template_derives_first_frame_from_parent(self):
        template = json.loads(
            (API_DIR / "h3_continuation_window.template.json").read_text()
        )
        self.assertEqual(template["8"]["inputs"]["first_frame"], ["26", 0])
        self.assertEqual(template["25"]["inputs"]["samples"], ["9", 0])
        self.assertEqual(template["26"]["class_type"], "CauceSelectImageFrame")


if __name__ == "__main__":
    unittest.main()
