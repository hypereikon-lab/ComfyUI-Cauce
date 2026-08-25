import importlib.util
from pathlib import Path
import sys
import unittest


class RegistrationTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "NumPy is supplied by ComfyUI, not CAUCE",
    )
    def test_root_registers_every_node_without_importing_comfy(self):
        root = Path(__file__).resolve().parents[1]
        name = "comfyui_cauce_test_plugin"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertEqual(len(module.NODE_CLASS_MAPPINGS), 24)
            self.assertEqual(
                set(module.NODE_CLASS_MAPPINGS),
                set(module.NODE_DISPLAY_NAME_MAPPINGS),
            )
            self.assertTrue(
                all(name.startswith("Cauce") for name in module.NODE_CLASS_MAPPINGS)
            )
            self.assertEqual(
                {node.CATEGORY for node in module.NODE_CLASS_MAPPINGS.values()},
                {
                    "CAUCE/Assembly",
                    "CAUCE/H3 AV Latent",
                    "CAUCE/H3 Planning",
                    "CAUCE/Motion Maps",
                    "CAUCE/Persistence",
                },
            )
            for name in (
                "CauceH3InspectAVLatent",
                "CauceH3PlanAVWindow",
                "CauceH3AllocateAVWindow",
                "CauceH3ExtractAVSpan",
                "CauceH3AddAVSpanGuide",
                "CauceH3AppendAVSpan",
                "CauceH3SplitAVLatent",
                "CauceH3ResolveTargetShape",
                "CauceH3PrepareGuideClip",
                "CauceH3PrepareReferenceClip",
                "CauceH3InspectConditioning",
            ):
                self.assertIn(name, module.NODE_CLASS_MAPPINGS)
        finally:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    sys.modules.pop(key, None)


if __name__ == "__main__":
    unittest.main()
