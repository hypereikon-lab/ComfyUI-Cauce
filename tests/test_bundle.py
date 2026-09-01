import json
import unittest
from pathlib import Path

from cauce.bundle import build_contract_bundle, validate_contract_bundle
from cauce.operations import load_operation_history

ROOT = Path(__file__).resolve().parents[1]


class ContractBundleTests(unittest.TestCase):
    def test_historical_contract_is_exact_and_content_addressed(self):
        history = load_operation_history(ROOT)
        self.assertEqual(set(history), {("regenerate.spatial", 2)})
        record = history[("regenerate.spatial", 2)]
        self.assertEqual(
            record["contract_hash"],
            "40a801bf58ab20e2757725f961298ba48078d241517f596c7215de4efe7e267d",
        )
        self.assertEqual(record["spec"]["version"], 2)

    def test_serialized_bundle_is_a_fresh_canonical_compilation(self):
        path = ROOT / "operations" / "contract-bundle.json"
        serialized = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(serialized, build_contract_bundle(ROOT))
        self.assertEqual(validate_contract_bundle(ROOT, serialized), [])
        self.assertEqual(len(serialized["nodes"]), 28)
        self.assertEqual(len(serialized["operations"]["current"]), 13)
        self.assertEqual(len(serialized["topologies"]["entries"]), 35)

    def test_bundle_hash_fails_closed_on_tampering(self):
        bundle = build_contract_bundle(ROOT)
        bundle["operations"]["historical"][0]["contract_hash"] = "0" * 64
        self.assertEqual(
            validate_contract_bundle(ROOT, bundle),
            ["contract bundle differs from canonical CAUCE sources"],
        )


if __name__ == "__main__":
    unittest.main()
