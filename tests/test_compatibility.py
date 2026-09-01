import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from cauce.av.backend import TensorBackend, backend_for
from cauce.comfy_compat.capabilities import probe_comfy_capabilities


class CompatibilityTests(unittest.TestCase):
    def test_numpy_conforms_to_the_tensor_backend_protocol(self):
        import numpy as np

        backend = backend_for(np.zeros((1,), dtype=np.float32))
        self.assertIsInstance(backend, TensorBackend)
        self.assertEqual(backend.name, "numpy")

    def test_capability_probe_reports_supported_current_h3_shape(self):
        modules = {
            "comfy": types.ModuleType("comfy"),
            "comfy.nested_tensor": types.ModuleType("comfy.nested_tensor"),
            "comfy.ldm": types.ModuleType("comfy.ldm"),
            "comfy.ldm.minimax": types.ModuleType("comfy.ldm.minimax"),
            "comfy.ldm.minimax.model": types.ModuleType("comfy.ldm.minimax.model"),
            "node_helpers": types.ModuleType("node_helpers"),
            "folder_paths": types.ModuleType("folder_paths"),
        }

        class NestedTensor:
            pass

        class PackedLayout:
            def __init__(
                self,
                text_len,
                latent_t,
                latent_h,
                latent_w,
                audio_t,
                keyframes=None,
                refs=None,
            ):
                pass

        modules["comfy.nested_tensor"].NestedTensor = NestedTensor
        modules["comfy.ldm.minimax.model"].PackedLayout = PackedLayout
        modules["node_helpers"].conditioning_set_values = lambda value, updates: value
        modules["folder_paths"].get_output_directory = lambda: str(Path.cwd())
        with patch.dict(sys.modules, modules, clear=False):
            report = probe_comfy_capabilities()
        self.assertTrue(report.nested_tensor)
        self.assertTrue(report.packed_layout)
        self.assertTrue(report.arbitrary_h3_guides)
        self.assertTrue(report.conditioning_set_values)
        self.assertTrue(report.output_directory)
        self.assertIsNone(report.error)

    def test_capability_probe_marks_legacy_packed_layout_unsupported(self):
        module = types.ModuleType("comfy.ldm.minimax.model")

        class LegacyPackedLayout:
            def __init__(self, text_len, frame_count):
                pass

        module.PackedLayout = LegacyPackedLayout
        with patch.dict(sys.modules, {"comfy.ldm.minimax.model": module}, clear=False):
            report = probe_comfy_capabilities()
        self.assertFalse(report.arbitrary_h3_guides)


if __name__ == "__main__":
    unittest.main()
