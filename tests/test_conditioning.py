import unittest

try:
    import numpy as np
except ImportError:
    np = None

from cauce.conditioning import inspect_h3_conditioning, inspect_h3_packed_sequence
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

    def test_packed_sequence_matches_official_layout_segments(self):
        target = self.target()
        guide = np.zeros((1, 24, 2, 2, 3), dtype=np.float32)
        guide_audio = np.zeros((1, 32, 2, 4), dtype=np.float32)
        ref_video = np.zeros((1, 24, 3, 4, 6), dtype=np.float32)
        positive = [[
            np.zeros((1, 12, 8), dtype=np.float32),
            {
                "minimax_keyframes": [
                    {"resolved_frame_index": 0, "latent": guide, "audio_latent": guide_audio}
                ],
                "minimax_refs": [
                    {"kind": "image", "latent": guide, "latent_h": 2, "latent_w": 3},
                    {
                        "kind": "video_audio",
                        "latent": ref_video,
                        "latent_t": 3,
                        "latent_h": 4,
                        "latent_w": 6,
                        "ref_audio_t": 5,
                    },
                ],
            },
        ]]
        report = inspect_h3_packed_sequence(positive, target, estimated_bytes_per_row=100)
        rows = report["entries"][0]["rows"]
        self.assertEqual(rows["text"], 12)
        self.assertEqual(rows["keyframe_video"], 4)
        self.assertEqual(rows["keyframe_audio"], 8)
        self.assertEqual(rows["reference_visual"], 20)
        self.assertEqual(rows["reference_audio"], 10)
        self.assertEqual(rows["target_audio"], 414)
        self.assertEqual(rows["target_video"], 74)
        self.assertEqual(report["total_rows"], 542)
        self.assertEqual(report["estimated_working_set_bytes"], 54_200)
        self.assertFalse(report["int32_attention_offset_risk"])
        self.assertEqual(len(report["report_hash"]), 64)

    def test_packed_sequence_uses_largest_scheduled_conditioning_entry(self):
        target = self.target()
        positive = [
            [np.zeros((1, 8, 4), dtype=np.float32), {}],
            [np.zeros((1, 20, 4), dtype=np.float32), {}],
        ]
        report = inspect_h3_packed_sequence(positive, target)
        self.assertEqual(report["active_entry_index"], 1)
        self.assertEqual(report["total_rows"], 20 + 414 + 74)


if __name__ == "__main__":
    unittest.main()
