import unittest

from cauce.seams import (
    make_seam_plan,
    make_seam_window,
    seam_splice_ranges,
    seam_video_token_values,
)


class SeamTests(unittest.TestCase):
    def test_default_plan_resolves_120_real_frames_inside_124_h3_frames(self):
        plan = make_seam_plan(200, 180)
        self.assertEqual(plan["context_frames_per_side"], 60)
        self.assertEqual(plan["repair_frames_per_side"], 24)
        self.assertEqual(plan["working_frames"], 124)
        self.assertEqual(plan["guard_frames_per_side"], 2)
        self.assertEqual(plan["cut_frame"], 62)
        self.assertEqual((plan["repair_start_frame"], plan["repair_end_frame"]), (38, 86))
        self.assertEqual((plan["accepted_start_frame"], plan["accepted_end_frame"]), (2, 122))

    def test_seam_mask_preserves_outer_tokens_and_generates_the_middle(self):
        plan = make_seam_plan(200, 180)
        values = seam_video_token_values(plan, feather_frames=6)
        self.assertEqual(len(values), 37)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 0.0)
        self.assertIn(1.0, values)
        self.assertTrue(any(0.0 < value < 1.0 for value in values))

    def test_seam_window_matches_the_plan_exactly(self):
        plan = make_seam_plan(200, 180)
        window = make_seam_window(plan)
        self.assertEqual(window["shape"]["pixel_frames"], plan["working_frames"])
        self.assertEqual(window["accepted_frames"], plan["working_frames"])
        self.assertEqual(window["accept_mode"], "full_render")

    def test_splice_ranges_replace_only_the_inner_second_of_each_source(self):
        plan = make_seam_plan(200, 180)
        self.assertEqual(
            seam_splice_ranges(plan),
            {
                "left_keep": (0, 176),
                "working_patch": (38, 86),
                "right_keep": (24, 180),
            },
        )
        output_frames = 176 + (86 - 38) + (180 - 24)
        self.assertEqual(output_frames, 380)

    def test_plan_rejects_sources_shorter_than_the_context(self):
        with self.assertRaisesRegex(ValueError, "each source needs at least 60 frames"):
            make_seam_plan(59, 100)


if __name__ == "__main__":
    unittest.main()
