from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cauce.contracts import append_field_span, make_window
from cauce.h3 import (
    append_reference,
    empty_reference_set,
    execute_add_guide,
    frame_index_in_window,
    h3_temporal_edit_capabilities,
    official_h3_nodes,
    require_h3_temporal_edit_runtime,
)
from cauce.masks import compile_audio_field, compile_video_field


class H3AndMaskTests(unittest.TestCase):
    def test_temporal_edit_runtime_requires_native_guides_and_mask_hooks(self):
        class OldMiniMaxH3:
            pass

        with patch(
            "cauce.h3.official_h3_nodes", return_value=(object(), object(), None)
        ), patch(
            "cauce.h3.importlib.import_module",
            return_value=SimpleNamespace(MiniMaxH3=OldMiniMaxH3),
        ):
            capabilities = h3_temporal_edit_capabilities()
            self.assertFalse(capabilities["ready"])
            self.assertFalse(capabilities["add_guide"])
            self.assertFalse(capabilities["per_token_denoise_mask"])
            with self.assertRaisesRegex(RuntimeError, "unsafe for temporal editing"):
                require_h3_temporal_edit_runtime()

    def test_temporal_edit_runtime_accepts_feature_complete_official_core(self):
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

        current_engine = SimpleNamespace(
            MiniMaxH3Model=CurrentMiniMaxH3Model,
            mask_row_values=lambda: None,
            _mod_row=lambda: None,
        )

        def import_current(name):
            if name == "comfy.model_base":
                return SimpleNamespace(MiniMaxH3=CurrentMiniMaxH3)
            if name == "comfy.ldm.minimax.model":
                return current_engine
            raise ImportError(name)

        with patch(
            "cauce.h3.official_h3_nodes",
            return_value=(object(), object(), object()),
        ), patch(
            "cauce.h3.importlib.import_module",
            side_effect=import_current,
        ):
            capabilities = require_h3_temporal_edit_runtime()
        self.assertTrue(capabilities["ready"])
        self.assertTrue(capabilities["add_guide"])
        self.assertTrue(capabilities["per_token_denoise_mask"])

    def test_official_fl2va_and_ref2va_do_not_require_add_guide(self):
        module = SimpleNamespace(
            MiniMaxH3ImageToVideo=object(),
            MiniMaxH3ReferenceToVideo=object(),
        )
        with patch("cauce.h3.importlib.import_module", return_value=module):
            image_to_video, reference_to_video, add_guide = official_h3_nodes()
        self.assertIs(image_to_video, module.MiniMaxH3ImageToVideo)
        self.assertIs(reference_to_video, module.MiniMaxH3ReferenceToVideo)
        self.assertIsNone(add_guide)

    def test_timed_guide_reports_optional_runtime_gap(self):
        with patch(
            "cauce.h3.official_h3_nodes", return_value=(object(), object(), None)
        ):
            with self.assertRaisesRegex(
                RuntimeError, "does not provide MiniMaxH3AddGuide"
            ):
                execute_add_guide()

    def test_reference_limits_are_fail_closed(self):
        refs = empty_reference_set()
        for index in range(9):
            refs = append_reference(refs, kind="image", media=f"image-{index}")
        with self.assertRaises(ValueError):
            append_reference(refs, kind="image", media="one-too-many")

    def test_reference_duration_bounds(self):
        with self.assertRaises(ValueError):
            append_reference(
                None, kind="video", media="video", duration_seconds=1.9
            )

    def test_guide_uses_absolute_master_time(self):
        window = make_window("w", 10, 5)
        self.assertEqual(frame_index_in_window(window, 10), 0)
        self.assertEqual(frame_index_in_window(window, Fraction(241, 24)), 1)
        with self.assertRaises(ValueError):
            frame_index_in_window(window, 9)

    def test_fields_compile_on_distinct_video_and_audio_grids(self):
        window = make_window("w", 10, 5)
        field = append_field_span(
            None,
            channel="both",
            start=10,
            end=Fraction(121, 12),  # first two visible frames
            strength=0.0,
        )
        video = compile_video_field(window, field)
        audio = compile_audio_field(window, field)
        self.assertEqual(len(video), window["shape"]["video_latent_frames"])
        self.assertEqual(len(audio), window["shape"]["audio_latent_frames"])
        self.assertEqual(video[0], 0.0)
        self.assertEqual(audio[0], 0.0)
        self.assertEqual(video[-1], 1.0)
        self.assertEqual(audio[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
