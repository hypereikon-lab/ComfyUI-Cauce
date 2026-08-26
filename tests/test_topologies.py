import importlib.util
from pathlib import Path
import sys
import unittest

from cauce.topologies import load_topology_catalog


ROOT = Path(__file__).resolve().parents[1]


class TopologyTests(unittest.TestCase):
    def test_offline_topologies_cover_every_operation(self):
        topologies = load_topology_catalog(ROOT)
        self.assertEqual(
            set(topologies),
            {
                "complete.native_av@backward-prefix",
                "complete.native_av@local-replacement",
                "complete.native_av@two-sided-infill",
                "complete.native_av@two-source-connection",
                "continue.native_av@keyframe-overlap",
                "continue.native_av@masked-overlap",
                "continue.native_av@masked-overlap-future-guide",
                "frames.assemble@ordered-concatenation",
                "generate.from_references@image-reference-match",
                "generate.from_references@image-reference-max",
                "generate.from_references@video-reference",
                "generate.from_references@video-reference-with-guide",
                "generate.keyframed@first-frame",
                "generate.keyframed@first-last",
                "generate.keyframed@last-frame",
                "generate.keyframed@text-only",
                "generate.with_guides@first-last-interior",
                "generate.with_guides@guide-clip",
                "generate.with_guides@multi-anchor",
                "generate.with_guides@single-anchor",
                "reference.transform@affine",
                "rollback.native_av@branch-suffix",
            },
        )
        self.assertEqual(
            {value["operation"] for value in topologies.values()},
            {
                "complete.native_av",
                "continue.native_av",
                "frames.assemble",
                "generate.from_references",
                "generate.keyframed",
                "generate.with_guides",
                "reference.transform",
                "rollback.native_av",
            },
        )
        self.assertTrue(all(value["state"] == "offline-draft" for value in topologies.values()))

    def test_topologies_are_designs_not_executable_graphs(self):
        for topology in load_topology_catalog(ROOT).values():
            self.assertNotIn("class_type", topology)
            self.assertNotIn("links", topology)
            self.assertTrue(topology["live_gates"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
    def test_cauce_topology_ports_match_installed_node_contracts(self):
        package_name = "comfyui_cauce_topology_test_plugin"
        spec = importlib.util.spec_from_file_location(
            package_name,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        try:
            assert spec.loader is not None
            spec.loader.exec_module(module)
            mappings = module.NODE_CLASS_MAPPINGS
            self._assert_cauce_ports(mappings)
        finally:
            for key in list(sys.modules):
                if key == package_name or key.startswith(package_name + "."):
                    sys.modules.pop(key, None)

    def _assert_cauce_ports(self, mappings):
        for topology in load_topology_catalog(ROOT).values():
            nodes = {node["key"]: node for node in topology["nodes"]}

            def node_contract(node_key):
                node = nodes[node_key]
                if node["owner"] != "cauce":
                    return None
                self.assertIn(
                    node["class_type"],
                    mappings,
                    f"{topology['operation']} references unknown CAUCE node "
                    f"{node['class_type']!r}",
                )
                return mappings[node["class_type"]]

            def input_names(node_class):
                declared = node_class.INPUT_TYPES()
                names = set()
                for group in ("required", "optional", "hidden"):
                    values = declared.get(group, {})
                    if isinstance(values, dict):
                        names.update(values)
                return names

            for binding in topology["bindings"]:
                node_class = node_contract(binding["target"]["node"])
                if node_class is not None:
                    self.assertIn(
                        binding["target"]["input"],
                        input_names(node_class),
                        f"{topology['operation']} binding {binding['name']!r} targets "
                        "an unknown CAUCE input",
                    )

            for edge in topology["edges"]:
                source_class = node_contract(edge["from"]["node"])
                if source_class is not None:
                    self.assertIn(
                        edge["from"]["port"],
                        source_class.RETURN_NAMES,
                        f"{topology['operation']} edge reads an unknown CAUCE output",
                    )
                target_class = node_contract(edge["to"]["node"])
                if target_class is not None:
                    self.assertIn(
                        edge["to"]["port"],
                        input_names(target_class),
                        f"{topology['operation']} edge targets an unknown CAUCE input",
                    )

            for output in topology["outputs"]:
                source_class = node_contract(output["source"]["node"])
                if source_class is not None:
                    self.assertIn(
                        output["source"]["port"],
                        source_class.RETURN_NAMES,
                        f"{topology['operation']} operation output reads an unknown "
                        "CAUCE output",
                    )


if __name__ == "__main__":
    unittest.main()
