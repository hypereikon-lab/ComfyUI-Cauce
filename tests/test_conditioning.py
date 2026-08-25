import unittest

try:
    import numpy as np
except ImportError:
    np = None

from cauce.conditioning import inspect_h3_conditioning
from cauce.timebase import h3_audio_token_boundary, h3_visual_latent_frames


@unittest.skipIf(np is None, "NumPy is supplied by ComfyUI, not CAUCE")
class ConditioningInspectionTests(unittest.TestCase):
    @staticmethod
    def target(frames=124, *, origin=0):
        video = np.zeros((1, 24, h3_visual_latent_frames(frames), 2, 3), dtype=np.float32)
        audio_tokens = h3_audio_token_boundary(origin + frames) - h3_audio_token_boundary(origin)
        audio = np.zeros((1, 32, 2, audio_tokens), dtype=np.float32)
        return {"samples": (video, audio)}

    def test_reports_guides_references_and_overlap_without_mutation(self):
        guide_22 = np.zeros((1, 24, 7, 2, 3), dtype=np.float32)
        guide_5 = np.zeros((1, 24, 2, 2, 3), dtype=np.float32)
        audio_22 = np.zeros((1, 32, 2, 37), dtype=np.float32)
        metadata = {
            "minimax_keyframes": [
                {
                    "resolved_frame_index": 0,
                    "latent": guide_22,
                    "audio_latent": audio_22,
                },
                {"resolved_frame_index": 17, "latent": guide_5},
            ],
            "minimax_refs": [
                {"kind": "image", "latent": guide_5, "latent_t": 2},
            ],
        }
        positive = [[np.zeros((1, 2), dtype=np.float32), metadata]]
        report = inspect_h3_conditioning(positive, self.target())
        self.assertEqual(report["target_frames"], 124)
        self.assertEqual(report["keyframe_count"], 2)
        self.assertEqual(report["reference_count"], 1)
        self.assertEqual(report["overlap_count"], 1)
        self.assertEqual(report["overlaps"][0]["range"], [17, 22])
        self.assertEqual(report["entries"][0]["keyframes"][0]["guide_frames"], 22)
        self.assertEqual(len(report["report_hash"]), 64)
        self.assertIs(positive[0][1], metadata)

        aligned = inspect_h3_conditioning(
            [[None, {}]],
            self.target(56, origin=17),
            timeline_origin_frame=17,
        )
        self.assertEqual(aligned["timeline_origin_frame"], 17)

    def test_rejects_out_of_range_and_malformed_h3_metadata(self):
        guide = np.zeros((1, 24, 7, 2, 3), dtype=np.float32)
        positive = [[None, {"minimax_keyframes": [{"resolved_frame_index": 110, "latent": guide}]}]]
        with self.assertRaisesRegex(ValueError, "outside"):
            inspect_h3_conditioning(positive, self.target())
        with self.assertRaisesRegex(TypeError, "minimax_refs"):
            inspect_h3_conditioning([[None, {"minimax_refs": {}}]], self.target())


if __name__ == "__main__":
    unittest.main()
