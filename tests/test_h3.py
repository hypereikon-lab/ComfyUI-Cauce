from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cauce.h3 import h3_temporal_mask_capabilities, require_h3_temporal_mask_runtime


class H3Tests(unittest.TestCase):
    def test_temporal_mask_runtime_fails_closed_without_official_hooks(self):
        class IncompleteMiniMaxH3:
            pass

        with patch(
            "cauce.h3.importlib.import_module",
            return_value=SimpleNamespace(MiniMaxH3=IncompleteMiniMaxH3),
        ):
            capabilities = h3_temporal_mask_capabilities()
            self.assertFalse(capabilities["ready"])
            self.assertFalse(capabilities["per_token_denoise_mask"])
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                require_h3_temporal_mask_runtime()

    def test_temporal_mask_runtime_accepts_complete_official_hooks(self):
        class CurrentMiniMaxH3:
            def _token_grid_masks(self):
                pass

            def _denoise_mask_conds(self):
                pass

            def scale_latent_inpaint(self):
                pass

        class CurrentMiniMaxH3Model:
            def forward(self, denoise_mask=None, audio_denoise_mask=None):
                pass

            def _forward(self, denoise_mask=None, audio_denoise_mask=None):
                pass

        def import_current(name):
            if name == "comfy.model_base":
                return SimpleNamespace(MiniMaxH3=CurrentMiniMaxH3)
            if name == "comfy.ldm.minimax.model":
                return SimpleNamespace(
                    MiniMaxH3Model=CurrentMiniMaxH3Model,
                    mask_row_values=lambda: None,
                    _mod_row=lambda: None,
                )
            raise ImportError(name)

        with patch("cauce.h3.importlib.import_module", side_effect=import_current):
            capabilities = require_h3_temporal_mask_runtime()
        self.assertTrue(capabilities["ready"])
        self.assertTrue(capabilities["per_token_denoise_mask"])


if __name__ == "__main__":
    unittest.main()
