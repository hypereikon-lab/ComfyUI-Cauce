from fractions import Fraction
import unittest

from cauce.temporal import plan_h3_guide_retime


class TemporalPlanningTests(unittest.TestCase):
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

    def test_guide_retime_rejects_nonpositive_scale(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            plan_h3_guide_retime(124, Fraction(0, 1))


if __name__ == "__main__":
    unittest.main()
