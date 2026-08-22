import unittest

try:
    import numpy
except ImportError:  # CAUCE itself adds no local Python dependencies
    numpy = None

from cauce.seams import (
    make_seam_plan,
    make_seam_window,
    seam_splice_ranges,
    seam_video_token_values,
    seam_visible_frame_values,
    splice_seam_patch,
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
        self.assertEqual((plan["sampling_start_frame"], plan["sampling_end_frame"]), (26, 98))
        self.assertEqual(plan["sampling_overscan_frames_per_side"], 12)
        self.assertEqual((plan["accepted_start_frame"], plan["accepted_end_frame"]), (2, 122))

    def test_seam_mask_is_binary_and_covers_the_sampling_overscan(self):
        plan = make_seam_plan(200, 180)
        values = seam_video_token_values(plan)
        self.assertEqual(len(values), 37)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 0.0)
        self.assertIn(1.0, values)
        self.assertEqual(set(values), {0.0, 1.0})

    def test_cover_projection_is_a_superset_of_majority(self):
        plan = make_seam_plan(200, 180)
        cover = seam_video_token_values(plan, projection="cover")
        majority = seam_video_token_values(plan, projection="majority")
        self.assertTrue(all(left >= right for left, right in zip(cover, majority)))

    def test_visible_fields_separate_sampling_acceptance_and_output_opacity(self):
        plan = make_seam_plan(200, 180)
        sampling, acceptance, opacity = seam_visible_frame_values(
            plan,
            decoded_blend_frames=8,
            curve="cosine",
        )
        start, end = plan["repair_start_frame"], plan["repair_end_frame"]
        sampling_start = plan["sampling_start_frame"]
        self.assertEqual(len(sampling), 124)
        self.assertEqual(sampling[sampling_start - 1], 0.0)
        self.assertEqual(sampling[sampling_start], 1.0)
        self.assertEqual(sampling[start], 1.0)
        self.assertEqual(acceptance[start], 1.0)
        self.assertEqual(acceptance[start - 1], 0.0)
        self.assertEqual(opacity[start], 0.0)
        self.assertEqual(opacity[end - 1], 0.0)

    def test_overscan_must_fit_inside_context(self):
        with self.assertRaisesRegex(ValueError, "overscan"):
            make_seam_plan(200, 180, sampling_overscan_seconds_per_side=1.5)

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

    @unittest.skipIf(numpy is None, "NumPy is supplied by ComfyUI, not CAUCE")
    def test_splice_preserves_every_frame_outside_the_accepted_patch(self):
        plan = make_seam_plan(200, 180)
        left = numpy.arange(200, dtype=numpy.float32).reshape(200, 1, 1, 1)
        right = (1000 + numpy.arange(180, dtype=numpy.float32)).reshape(180, 1, 1, 1)
        proposal = numpy.full((124, 1, 1, 1), 5000.0, dtype=numpy.float32)
        joined, patch, _ = splice_seam_patch(
            left, right, proposal, plan, feather_frames=8, curve="cosine"
        )
        self.assertEqual(joined.shape[0], 380)
        numpy.testing.assert_array_equal(joined[:176], left[:176])
        numpy.testing.assert_array_equal(joined[224:], right[24:])
        self.assertEqual(float(patch[0, 0, 0, 0]), float(left[-24, 0, 0, 0]))
        self.assertEqual(float(patch[-1, 0, 0, 0]), float(right[23, 0, 0, 0]))
        self.assertEqual(float(patch[12, 0, 0, 0]), 5000.0)

    def test_plan_rejects_sources_shorter_than_the_context(self):
        with self.assertRaisesRegex(ValueError, "each source needs at least 60 frames"):
            make_seam_plan(59, 100)


if __name__ == "__main__":
    unittest.main()
