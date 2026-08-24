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
            "50_h3_temporal_inpainting.json",
            "60_h3_native_latent_loop.json",
            "70_motion_map_composition.json",
            "71_h3_warped_noise.json",
            "72_h3_sequential_latent_pass.json",
            "73_depth_advection_preview.json",
            "90_storage_maintenance.json",
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
                self.assertNotIn("VAEDecodeAudio", {n["type"] for n in nodes.values()})

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

    def test_temporal_inpaint_demo_owns_its_exact_h3_window(self):
        workflow = json.loads(
            (WORKFLOW_DIR / "50_h3_temporal_inpainting.json").read_text()
        )
        types = {node["type"] for node in workflow["nodes"]}
        self.assertTrue(
            {
                "CauceBuildSeamWindow",
                "CauceTemporalInpaintFields",
                "CaucePrepareH3TemporalInpaint",
                "CauceApplySeamPatch",
                "CauceH3TemporalInpaintGuides",
                "SamplerCustomAdvanced",
            }.issubset(types)
        )
        self.assertNotIn("CauceGenerationWindow", types)
        build = next(
            node for node in workflow["nodes"] if node["type"] == "CauceBuildSeamWindow"
        )
        self.assertEqual(
            build["widgets_values"],
            [24.0, 24.0, 2.5, 3.0, 22, 362],
        )
        scales = [node for node in workflow["nodes"] if node["type"] == "ImageScale"]
        self.assertEqual(len(scales), 2)
        self.assertTrue(
            all(node["widgets_values"][1:3] == [640, 640] for node in scales)
        )
        fields = next(
            node for node in workflow["nodes"] if node["type"] == "CauceTemporalInpaintFields"
        )
        self.assertEqual(fields["widgets_values"], [4, "cosine"])
        prepare = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "CaucePrepareH3TemporalInpaint"
        )
        self.assertEqual(prepare["widgets_values"], ["cover", 0.5])

    def test_native_loop_demo_preserves_source_latents_and_repairs_both_seams(self):
        workflow = json.loads(
            (WORKFLOW_DIR / "60_h3_native_latent_loop.json").read_text()
        )
        types = [node["type"] for node in workflow["nodes"]]
        self.assertEqual(types.count("CauceBuildNativeLatentSeam"), 2)
        self.assertEqual(types.count("CaucePrepareH3NativeLatentInpaint"), 2)
        self.assertEqual(types.count("CauceH3TemporalInpaintGuides"), 2)
        self.assertEqual(types.count("CauceAssembleNativeTwoClipLoop"), 1)
        self.assertEqual(types.count("PrimitiveInt"), 1)
        self.assertEqual(types.count("CauceSaveAVLatent"), 2)
        profile = next(
            node for node in workflow["nodes"] if node["type"] == "CauceExecutionProfile"
        )
        self.assertEqual(profile["widgets_values"], ["h3-5090-fl2va-768x512"])
        seams = [
            node for node in workflow["nodes"] if node["type"] == "CauceBuildNativeLatentSeam"
        ]
        self.assertTrue(
            all(
                node["widgets_values"] == [24.0, 24.0, "22", "124"]
                for node in seams
            )
        )
        self.assertTrue(
            all(
                next(
                    item
                    for item in node["inputs"]
                    if item["name"] == "accepted_repair_frames"
                )["link"]
                is not None
                for node in seams
            )
        )
        accepted = next(node for node in workflow["nodes"] if node["type"] == "PrimitiveInt")
        self.assertEqual(accepted["widgets_values"], [72])
        conditions = [node for node in workflow["nodes"] if node["type"] == "CauceH3FL2VA"]
        self.assertEqual(
            [node["widgets_values"][0] for node in conditions[:2]],
            ["infinite fast zoom transition into"] * 2,
        )
        seam_prompts = [node["widgets_values"][0] for node in conditions[2:]]
        self.assertEqual(len(seam_prompts), 2)
        self.assertTrue(
            all(prompt.startswith("Regenerate only the masked temporal interval.") for prompt in seam_prompts)
        )
        assembly = next(
            node for node in workflow["nodes"] if node["type"] == "CauceAssembleNativeTwoClipLoop"
        )
        self.assertEqual(assembly["widgets_values"], [4, "cosine"])

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

    def test_motion_map_examples_distinguish_composition_and_sequential_passes(self):
        workflow = json.loads(
            (WORKFLOW_DIR / "70_motion_map_composition.json").read_text()
        )
        types = [node["type"] for node in workflow["nodes"]]
        self.assertEqual(types.count("CauceComposeMotionMaps"), 1)
        self.assertEqual(types.count("CauceWarpImage"), 3)
        self.assertEqual(types.count("SaveVideo"), 2)

        warped_noise = json.loads(
            (WORKFLOW_DIR / "71_h3_warped_noise.json").read_text()
        )
        noise_types = [node["type"] for node in warped_noise["nodes"]]
        self.assertIn("CauceWarpedH3Noise", noise_types)
        self.assertNotIn("RandomNoise", noise_types)
        self.assertEqual(noise_types.count("SamplerCustomAdvanced"), 1)
        warped_noise_node = next(
            node
            for node in warped_noise["nodes"]
            if node["type"] == "CauceWarpedH3Noise"
        )
        self.assertEqual(
            warped_noise_node["widgets_values"],
            [2026082401, "fixed", "reflection", 0.05],
        )
        modulate_node = next(
            node
            for node in warped_noise["nodes"]
            if node["type"] == "CauceModulateMotionMap"
        )
        self.assertEqual(modulate_node["widgets_values"], [0.0, 0.15, "sine_loop"])

        sequential = json.loads(
            (WORKFLOW_DIR / "72_h3_sequential_latent_pass.json").read_text()
        )
        sequential_types = [node["type"] for node in sequential["nodes"]]
        self.assertIn("CauceWarpH3Latent", sequential_types)
        self.assertEqual(sequential_types.count("SamplerCustomAdvanced"), 2)
        second_scheduler = [
            node for node in sequential["nodes"] if node["type"] == "BasicScheduler"
        ][-1]
        self.assertEqual(second_scheduler["widgets_values"], ["simple", 12, 0.35])

        depth = json.loads(
            (WORKFLOW_DIR / "73_depth_advection_preview.json").read_text()
        )
        depth_types = {node["type"] for node in depth["nodes"]}
        self.assertTrue(
            {
                "CauceDepthCameraMotionMap",
                "CauceVectorField",
                "CauceIntegrateAdvection",
                "CauceComposeMotionMaps",
            }.issubset(depth_types)
        )


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
