import unittest

try:
    import numpy
except ImportError:  # CAUCE itself adds no local Python dependencies
    numpy = None

from cauce.seams import (
    assemble_native_two_clip_loop,
    build_native_latent_seam_window,
    make_native_latent_seam_plan,
    make_seam_plan,
    make_seam_window,
    seam_splice_ranges,
    seam_video_token_values,
    seam_visible_frame_values,
    splice_seam_patch,
)


class TemporalInpaintTests(unittest.TestCase):
    def test_native_plan_is_phase_safe_and_uses_real_bidirectional_context(self):
        plan = make_native_latent_seam_plan(124, 124)
        self.assertEqual(plan["mode"], "native_av_latent_bidirectional")
        self.assertEqual(plan["working_frames"], 124)
        self.assertEqual(plan["working_video_tokens"], 37)
        self.assertEqual(plan["context_frames_per_side"], 39)
        self.assertEqual(plan["context_tokens_per_side"], 12)
        self.assertEqual((plan["repair_start_frame"], plan["repair_end_frame"]), (39, 85))
        self.assertEqual(plan["repair_frames_per_side"], 23)
        self.assertEqual((plan["left_target_start_token"], plan["left_target_end_token"]), (0, 12))
        self.assertEqual((plan["right_target_start_token"], plan["right_target_end_token"]), (25, 37))
        self.assertEqual(plan["left_latent_source_start_token"] % 5, 0)

    def test_native_plan_can_sample_with_overscan_and_accept_exactly_three_seconds(self):
        plan = make_native_latent_seam_plan(
            124,
            124,
            context_frames_per_side=22,
            accepted_repair_frames=72,
        )
        self.assertEqual(plan["context_tokens_per_side"], 7)
        self.assertEqual((plan["sampling_start_frame"], plan["sampling_end_frame"]), (22, 102))
        self.assertEqual(plan["sampling_total_frames"], 80)
        self.assertEqual((plan["repair_start_frame"], plan["repair_end_frame"]), (26, 98))
        self.assertEqual(plan["repair_total_frames"], 72)
        self.assertEqual(plan["repair_frames_per_side"], 36)
        self.assertEqual(plan["overscan_frames_per_side"], 4)
        self.assertEqual((plan["left_guide_start_frame"], plan["left_guide_end_frame"]), (0, 22))
        self.assertEqual((plan["right_guide_start_frame"], plan["right_guide_end_frame"]), (102, 124))
        self.assertEqual(plan["left_latent_source_start_token"] % 5, 0)

    @unittest.skipIf(numpy is None, "NumPy is supplied by ComfyUI, not CAUCE")
    def test_native_working_domain_is_62_tail_plus_62_head(self):
        plan = make_native_latent_seam_plan(124, 124)
        left = numpy.arange(124, dtype=numpy.float32).reshape(124, 1, 1, 1)
        right = (1000 + numpy.arange(124, dtype=numpy.float32)).reshape(124, 1, 1, 1)
        working = build_native_latent_seam_window(left, right, plan)
        self.assertEqual(working.shape[0], 124)
        numpy.testing.assert_array_equal(working[:62], left[-62:])
        numpy.testing.assert_array_equal(working[62:], right[:62])

    @unittest.skipIf(numpy is None, "NumPy is supplied by ComfyUI, not CAUCE")
    def test_native_two_seam_loop_preserves_duration_and_repairs_wrap_boundary(self):
        forward = make_native_latent_seam_plan(124, 124)
        wrap = make_native_latent_seam_plan(124, 124)
        first = numpy.arange(124, dtype=numpy.float32).reshape(124, 1, 1, 1)
        second = (1000 + numpy.arange(124, dtype=numpy.float32)).reshape(124, 1, 1, 1)
        forward_proposal = numpy.full((124, 1, 1, 1), 5000.0, dtype=numpy.float32)
        wrap_proposal = numpy.full((124, 1, 1, 1), 6000.0, dtype=numpy.float32)
        loop, first_fixed, second_fixed, forward_patch, wrap_patch, report = (
            assemble_native_two_clip_loop(
                first,
                second,
                forward_proposal,
                forward,
                wrap_proposal,
                wrap,
                feather_frames=4,
            )
        )
        self.assertEqual(loop.shape[0], 248)
        self.assertEqual(first_fixed.shape[0], 124)
        self.assertEqual(second_fixed.shape[0], 124)
        self.assertEqual(forward_patch.shape[0], 46)
        self.assertEqual(wrap_patch.shape[0], 46)
        numpy.testing.assert_array_equal(loop[:124], first_fixed)
        numpy.testing.assert_array_equal(loop[124:], second_fixed)
        numpy.testing.assert_array_equal(first_fixed[23:-23], first[23:-23])
        numpy.testing.assert_array_equal(second_fixed[23:-23], second[23:-23])
        # The exported loop boundary must contain the two central halves of the
        # repaired wrap patch. Preserving the original endpoints here would
        # recreate the hard second->first cut that this operation repairs.
        repair = int(wrap["repair_frames_per_side"])
        numpy.testing.assert_array_equal(first_fixed[0], wrap_patch[repair])
        numpy.testing.assert_array_equal(second_fixed[-1], wrap_patch[repair - 1])
        self.assertEqual(report["loop_frames"], 248)

    @unittest.skipIf(numpy is None, "NumPy is supplied by ComfyUI, not CAUCE")
    def test_native_three_second_repairs_leave_non_overlapping_source_centers(self):
        forward = make_native_latent_seam_plan(
            124, 124, context_frames_per_side=22, accepted_repair_frames=72
        )
        wrap = make_native_latent_seam_plan(
            124, 124, context_frames_per_side=22, accepted_repair_frames=72
        )
        first = numpy.arange(124, dtype=numpy.float32).reshape(124, 1, 1, 1)
        second = (1000 + numpy.arange(124, dtype=numpy.float32)).reshape(124, 1, 1, 1)
        proposal = numpy.full((124, 1, 1, 1), 5000.0, dtype=numpy.float32)
        loop, first_fixed, second_fixed, forward_patch, wrap_patch, report = (
            assemble_native_two_clip_loop(
                first,
                second,
                proposal,
                forward,
                proposal,
                wrap,
                feather_frames=4,
            )
        )
        self.assertEqual(loop.shape[0], 248)
        self.assertEqual(forward_patch.shape[0], 72)
        self.assertEqual(wrap_patch.shape[0], 72)
        numpy.testing.assert_array_equal(first_fixed[36:-36], first[36:-36])
        numpy.testing.assert_array_equal(second_fixed[36:-36], second[36:-36])
        self.assertEqual(report["loop_frames"], 248)

    def test_native_plan_rejects_phase_unsafe_source_lengths(self):
        with self.assertRaisesRegex(ValueError, "complete H3 runs"):
            make_native_latent_seam_plan(125, 124)

    def test_default_plan_resolves_three_second_request_to_exact_h3_interval(self):
        plan = make_seam_plan(200, 180)
        self.assertEqual(plan["context_frames_per_side"], 60)
        self.assertEqual(plan["working_frames"], 124)
        self.assertEqual(plan["guard_frames_per_side"], 2)
        self.assertEqual(plan["cut_frame"], 62)
        self.assertEqual(plan["repair_requested_frames_total"], 72)
        self.assertEqual(plan["repair_total_frames"], 72)
        self.assertEqual(plan["repair_frames_per_side"], 36)
        self.assertEqual((plan["repair_start_frame"], plan["repair_end_frame"]), (26, 98))
        self.assertEqual((plan["sampling_start_frame"], plan["sampling_end_frame"]), (26, 98))
        self.assertEqual((plan["left_guide_start_frame"], plan["left_guide_end_frame"]), (4, 26))
        self.assertEqual((plan["right_guide_start_frame"], plan["right_guide_end_frame"]), (98, 120))
        self.assertEqual((plan["accepted_start_frame"], plan["accepted_end_frame"]), (2, 122))

    def test_seam_mask_is_binary_and_exactly_covers_the_token_aligned_gap(self):
        plan = make_seam_plan(200, 180)
        values = seam_video_token_values(plan)
        self.assertEqual(len(values), 37)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 0.0)
        self.assertIn(1.0, values)
        self.assertEqual(set(values), {0.0, 1.0})
        self.assertEqual([index for index, value in enumerate(values) if value], list(range(8, 29)))

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

    def test_guides_must_use_a_native_h3_clip_length(self):
        with self.assertRaisesRegex(ValueError, "guide_frames"):
            make_seam_plan(200, 180, guide_frames=24)

    def test_seam_window_matches_the_plan_exactly(self):
        plan = make_seam_plan(200, 180)
        window = make_seam_window(plan)
        self.assertEqual(window["shape"]["pixel_frames"], plan["working_frames"])
        self.assertEqual(window["accepted_frames"], plan["working_frames"])
        self.assertEqual(window["accept_mode"], "full_render")

    def test_splice_ranges_replace_only_the_token_aligned_inpaint_interval(self):
        plan = make_seam_plan(200, 180)
        self.assertEqual(
            seam_splice_ranges(plan),
            {
                "left_keep": (0, 164),
                "working_patch": (26, 98),
                "right_keep": (36, 180),
            },
        )
        output_frames = 164 + (98 - 26) + (180 - 36)
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
        numpy.testing.assert_array_equal(joined[:164], left[:164])
        numpy.testing.assert_array_equal(joined[236:], right[36:])
        self.assertEqual(float(patch[0, 0, 0, 0]), float(left[-36, 0, 0, 0]))
        self.assertEqual(float(patch[-1, 0, 0, 0]), float(right[35, 0, 0, 0]))
        self.assertEqual(float(patch[10, 0, 0, 0]), 5000.0)

    def test_plan_rejects_sources_shorter_than_the_context(self):
        with self.assertRaisesRegex(ValueError, "each source needs at least 60 frames"):
            make_seam_plan(59, 100)


if __name__ == "__main__":
    unittest.main()
