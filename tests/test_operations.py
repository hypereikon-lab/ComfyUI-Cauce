import copy
import json
from pathlib import Path
import unittest

from cauce.operations import (
    load_json,
    load_operation_catalog,
    operation_contract_hash,
    validate_operation_spec,
)


ROOT = Path(__file__).resolve().parents[1]


class OperationContractTests(unittest.TestCase):
    def test_catalog_is_complete_and_semantic(self):
        operations = load_operation_catalog(ROOT)
        self.assertEqual(
            set(operations),
            {
                "connect.two_sided_guides",
                "continue.native_av",
                "frames.assemble",
                "generate.from_references",
                "generate.keyframed",
                "generate.with_guides",
                "reference.transform",
            },
        )
        self.assertTrue(all(not operation_id.startswith("W") for operation_id in operations))

    def test_official_operations_do_not_claim_cauce_nodes(self):
        operations = load_operation_catalog(ROOT)
        for operation_id in (
            "generate.keyframed",
            "generate.from_references",
            "generate.with_guides",
        ):
            spec = operations[operation_id]
            self.assertEqual(spec["implementation_class"], "official-h3")
            self.assertNotIn("cauce", {stage["owner"] for stage in spec["graph_contract"]})

    def test_composition_classes_match_real_graph_owners(self):
        operations = load_operation_catalog(ROOT)
        for operation_id in ("continue.native_av", "connect.two_sided_guides"):
            owners = {stage["owner"] for stage in operations[operation_id]["graph_contract"]}
            self.assertTrue({"official-comfy", "cauce"} <= owners)
        for operation_id in ("reference.transform", "frames.assemble"):
            spec = operations[operation_id]
            owners = {stage["owner"] for stage in spec["graph_contract"]}
            self.assertEqual(spec["kind"], "decoded-media-transform")
            self.assertNotIn("official-comfy", owners)

    def test_contract_only_operations_ship_no_graph_pair(self):
        operations = load_operation_catalog(ROOT)
        for spec in operations.values():
            self.assertEqual(spec["artifacts"]["state"], "contract-only")
            self.assertIsNone(spec["artifacts"]["ui_graph"])
            self.assertIsNone(spec["artifacts"]["api_template"])
        graph_products = [
            path
            for path in (ROOT / "operations").rglob("*.json")
            if path.name.endswith(".ui.json") or ".api." in path.name
        ]
        self.assertEqual(graph_products, [])

    def test_visual_verdict_cannot_outrun_evidence(self):
        spec = copy.deepcopy(load_operation_catalog(ROOT)["generate.keyframed"])
        spec["evidence"]["visual_verdict"] = "accepted"
        self.assertIn(
            "visual verdict requires visually-characterized evidence",
            validate_operation_spec(spec),
        )

    def test_operation_hash_is_canonical(self):
        spec = load_operation_catalog(ROOT)["generate.keyframed"]
        reordered = json.loads(json.dumps(spec, sort_keys=True))
        self.assertEqual(operation_contract_hash(spec), operation_contract_hash(reordered))
        self.assertRegex(operation_contract_hash(spec), r"^[0-9a-f]{64}$")

    def test_native_continuation_smoke_is_evidence_not_template(self):
        evidence = load_json(ROOT / "operations" / "evidence" / "continue.native_av-smoke.json")
        self.assertEqual(evidence["operation"], "continue.native_av")
        self.assertEqual(evidence["evidence_level"], "executes")
        self.assertFalse(evidence["graph_recovery"]["executable_api_template"])
        nodes = evidence["graph_recovery"]["nodes"]
        self.assertEqual(len(nodes), 18)
        self.assertEqual(nodes[-1]["class_type"], "CauceSaveAVLatent")
        self.assertEqual(evidence["verification"]["frame_count"], 73)
        self.assertEqual(evidence["verification"]["structural_audio_tokens"], 122)


if __name__ == "__main__":
    unittest.main()
