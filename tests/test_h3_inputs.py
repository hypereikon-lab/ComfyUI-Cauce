import unittest

try:
    import numpy as np
except ImportError:
    np = None

from cauce.h3_inputs import (
    plan_h3_control_clip,
    plan_h3_guide_clip,
    plan_h3_reference_clip,
    prepare_h3_guide_clip,
    prepare_h3_reference_clip,
    resolve_h3_target_shape,
)
from cauce.timebase import h3_audio_token_boundary, h3_visual_latent_frames


@unittest.skipIf(np is None, "NumPy is supplied by ComfyUI, not CAUCE")
class H3InputPlanningTests(unittest.TestCase):
    @staticmethod
    def latent(frames, *, origin=0):
        video = np.zeros((1, 24, h3_visual_latent_frames(frames), 2, 3), dtype=np.float32)
        audio_tokens = h3_audio_token_boundary(origin + frames) - h3_audio_token_boundary(origin)
        audio = np.zeros((1, 32, 2, audio_tokens), dtype=np.float32)
        return {"samples": (video, audio)}

    def test_target_shape_exposes_official_ceil_and_token_geometry(self):
        plan = resolve_h3_target_shape(120, 1344, 768)
        self.assertEqual(plan["resolved_frames"], 124)
        self.assertEqual(plan["added_frames"], 4)
        self.assertEqual(plan["video_tokens"], 37)
        self.assertEqual(plan["audio_tokens"], 207)
        self.assertEqual(plan["duration"], {"numerator": 31, "denominator": 6})
        self.assertTrue(plan["inside_trained_range"])
        self.assertEqual(len(plan["plan_hash"]), 64)
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            resolve_h3_target_shape(124, 1300, 768)

    def test_guide_clip_makes_single_frame_and_floor_rules_explicit(self):
        target = self.latent(124)
        short = np.zeros((4, 8, 8, 3), dtype=np.float32)
        accepted, short_plan = prepare_h3_guide_clip(short, target, 0)
        self.assertEqual(accepted.shape[0], 1)
        self.assertEqual(short_plan["accepted_frames"], 1)
        self.assertEqual(short_plan["discarded_tail_frames"], 3)

        clip = np.zeros((60, 8, 8, 3), dtype=np.float32)
        plan = plan_h3_guide_clip(clip, target, -56)
        self.assertEqual(plan["accepted_frames"], 56)
        self.assertEqual(plan["resolved_frame_idx"], 68)
        self.assertEqual(plan["guide_range"], [68, 124])
        with self.assertRaisesRegex(ValueError, "does not fit"):
            plan_h3_guide_clip(clip, target, 69)

        aligned_window = self.latent(56, origin=17)
        aligned = plan_h3_guide_clip(
            clip[:5],
            aligned_window,
            0,
            timeline_origin_frame=17,
        )
        self.assertEqual(aligned["timeline_origin_frame"], 17)
        self.assertEqual(aligned["accepted_frames"], 5)

    def test_reference_clip_exposes_target_clamp_floor_and_qwen_samples(self):
        clip = np.zeros((60, 8, 8, 3), dtype=np.float32)
        accepted, plan = prepare_h3_reference_clip(clip, 120)
        self.assertEqual(accepted.shape[0], 56)
        self.assertEqual(plan["resolved_target_frames"], 124)
        self.assertEqual(plan["accepted_frames"], 56)
        self.assertEqual(plan["discarded_tail_frames"], 4)
        self.assertTrue(plan["inside_documented_duration_range"])
        self.assertEqual(plan["documented_duration_range_frames"], [48, 360])
        self.assertEqual(plan["qwen_sample_indices"], [0, 12, 24, 36, 48])
        self.assertEqual(plan["qwen_timestamps_seconds"], [0.0, 0.5, 1.0, 1.5, 2.0])

        long_clip = np.zeros((200, 8, 8, 3), dtype=np.float32)
        limited = plan_h3_reference_clip(long_clip, 120)
        self.assertEqual(limited["accepted_frames"], 124)
        self.assertEqual(limited["discarded_tail_frames"], 76)
        out_of_spec = plan_h3_reference_clip(clip[:22], 124)
        self.assertEqual(out_of_spec["accepted_frames"], 22)
        self.assertFalse(out_of_spec["inside_documented_duration_range"])
        with self.assertRaisesRegex(ValueError, "at least 5"):
            plan_h3_reference_clip(clip[:4], 124)
        with self.assertRaisesRegex(ValueError, "positive"):
            plan_h3_reference_clip(clip, 0)

    def test_control_clip_exposes_repeat_truncate_and_spatial_fit(self):
        target = self.latent(124)
        short = np.zeros((100, 720, 1280, 3), dtype=np.float32)
        repeated = plan_h3_control_clip(short, target)
        self.assertEqual(repeated["temporal_policy"], "repeat-last-frame")
        self.assertEqual(repeated["repeated_tail_frames"], 24)
        self.assertEqual(repeated["discarded_tail_frames"], 0)
        self.assertEqual(repeated["source_geometry"], {"width": 1280, "height": 720})
        self.assertEqual(repeated["target_geometry"], {"width": 48, "height": 32})
        self.assertEqual(repeated["spatial_policy"], "bilinear-resize-and-center-crop")
        self.assertFalse(repeated["mutates_images"])

        long = np.zeros((140, 32, 48, 3), dtype=np.float32)
        truncated = plan_h3_control_clip(long, target)
        self.assertEqual(truncated["temporal_policy"], "truncate-tail")
        self.assertEqual(truncated["accepted_source_frames"], 124)
        self.assertEqual(truncated["discarded_tail_frames"], 16)


if __name__ == "__main__":
    unittest.main()
