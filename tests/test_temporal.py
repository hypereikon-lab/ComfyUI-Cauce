from fractions import Fraction
import unittest

from cauce.temporal import (
    analyze_h3_interleave_projection,
    plan_frame_interpolation,
    plan_h3_guide_retime,
)


class TemporalPlanningTests(unittest.TestCase):
    def test_endpoint_preserving_two_x_interpolation_uses_exact_sample_grid(self):
        plan = plan_frame_interpolation(124, 2, 24)
        self.assertEqual(plan["target_frame_count"], 247)
        self.assertEqual(plan["target_fps"]["fraction"], {"numerator": 48, "denominator": 1})
        self.assertEqual(plan["inserted_frame_count"], 123)
        self.assertEqual(plan["source_anchor_output_indices"], list(range(0, 247, 2)))
        self.assertEqual(
            plan["sample_span_duration"]["fraction"],
            {"numerator": 41, "denominator": 8},
        )
        self.assertEqual(len(plan["plan_hash"]), 64)

    def test_h3_interleave_report_proves_mixed_tokens_are_regenerated(self):
        report = analyze_h3_interleave_projection(124, 2)
        self.assertEqual(report["raw_interpolated_frame_count"], 247)
        self.assertEqual(report["resolved_h3_frame_count"], 260)
        self.assertEqual(report["h3_padding_frame_count"], 13)
        self.assertEqual(report["video_token_count"], 77)
        # The 1-frame token at the start of every 17-frame H3 cycle sometimes
        # lands on an even (known) output index; the surrounding 4-frame tokens
        # still mix known and missing samples.
        self.assertEqual(report["token_counts"]["preserve_only"], 8)
        self.assertGreater(report["token_counts"]["mixed_known_and_missing"], 0)
        self.assertGreater(report["token_counts"]["generate_only"], 0)
        self.assertFalse(report["exact_known_frame_preservation_possible"])
        first_mixed = next(
            token
            for token in report["tokens"]
            if token["classification"] == "mixed-known-and-missing"
        )
        self.assertEqual(first_mixed["decoded_span"], [1, 5])
        self.assertEqual(first_mixed["projected_mask"], 1.0)

    def test_multiplier_one_on_legal_h3_span_preserves_every_token(self):
        report = analyze_h3_interleave_projection(124, 1)
        self.assertEqual(report["resolved_h3_frame_count"], 124)
        self.assertEqual(report["token_counts"]["mixed_known_and_missing"], 0)
        self.assertEqual(report["token_counts"]["generate_only"], 0)
        self.assertTrue(report["exact_known_frame_preservation_possible"])

    def test_sparse_h3_guide_retime_aligns_both_resolved_endpoints(self):
        plan = plan_h3_guide_retime(124, 2, 24, 24)
        self.assertEqual(plan["requested_target_frame_count"], 247)
        self.assertEqual(plan["resolved_h3_frame_count"], 260)
        self.assertEqual(plan["h3_lattice_padding_frames"], 13)
        self.assertEqual(plan["anchors"][0]["source_frame_index"], 0)
        self.assertEqual(plan["anchors"][0]["target_frame_index"], 0)
        self.assertEqual(plan["anchors"][-1]["source_frame_index"], 123)
        self.assertEqual(plan["anchors"][-1]["target_frame_index"], 259)
        self.assertEqual(
            plan["requested_sample_span_duration"]["fraction"],
            {"numerator": 41, "denominator": 4},
        )
        self.assertFalse(plan["pixel_exact_interpolation"])

    def test_temporal_plans_reject_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            plan_frame_interpolation(1, 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            plan_frame_interpolation(2, 2, 0)
        with self.assertRaisesRegex(ValueError, "positive"):
            plan_h3_guide_retime(124, Fraction(0, 1))


if __name__ == "__main__":
    unittest.main()
