import unittest

try:
    import numpy as np
except ImportError:
    np = None

from cauce.bridge import (
    apply_native_guide_bridge,
    extract_native_guide_bridge_sources,
    plan_native_guide_bridge,
)


@unittest.skipIf(np is None, "NumPy is supplied by ComfyUI, not CAUCE")
class NativeGuideBridgeTests(unittest.TestCase):
    def setUp(self):
        self.left = np.arange(60, dtype=np.float32).reshape(60, 1, 1, 1)
        self.right = (1000 + np.arange(70, dtype=np.float32)).reshape(70, 1, 1, 1)

    def test_default_plan_uses_two_22_frame_guides(self):
        plan = plan_native_guide_bridge(60, 70)
        self.assertEqual(plan["left_source_range"], [38, 60])
        self.assertEqual(plan["right_source_range"], [0, 22])
        self.assertEqual(plan["left_guide_frame_idx"], 0)
        self.assertEqual(plan["right_guide_frame_idx"], 102)
        self.assertEqual(plan["generated_center_range"], [22, 102])
        self.assertEqual(plan["generated_center_frames"], 80)
        self.assertEqual(plan["assembled_frames"], 210)

    def test_plan_is_deterministic(self):
        first = plan_native_guide_bridge(60, 70, guide_frames=39, target_frames=141)
        second = plan_native_guide_bridge(60, 70, guide_frames=39, target_frames=141)
        self.assertEqual(first, second)
        self.assertEqual(len(first["plan_hash"]), 64)

    def test_extracts_exact_tail_and_head(self):
        plan = plan_native_guide_bridge(60, 70)
        left_guide, right_guide = extract_native_guide_bridge_sources(
            self.left, self.right, plan
        )
        np.testing.assert_array_equal(left_guide, self.left[-22:])
        np.testing.assert_array_equal(right_guide, self.right[:22])

    def test_applies_only_generated_center(self):
        plan = plan_native_guide_bridge(60, 70)
        generated = (2000 + np.arange(124, dtype=np.float32)).reshape(124, 1, 1, 1)
        joined, center, report = apply_native_guide_bridge(
            self.left, self.right, generated, plan
        )
        np.testing.assert_array_equal(center, generated[22:102])
        np.testing.assert_array_equal(joined[:60], self.left)
        np.testing.assert_array_equal(joined[60:140], generated[22:102])
        np.testing.assert_array_equal(joined[140:], self.right)
        self.assertEqual(report["assembled_frames"], 210)
        self.assertEqual(report["quality_status"], "requires_visual_validation")

    def test_rejects_invalid_geometry_and_counts(self):
        with self.assertRaisesRegex(ValueError, "24 fps"):
            plan_native_guide_bridge(60, 70, fps=23.976)
        with self.assertRaisesRegex(ValueError, "guide_frames"):
            plan_native_guide_bridge(60, 70, guide_frames=21)
        with self.assertRaisesRegex(ValueError, "target_frames"):
            plan_native_guide_bridge(60, 70, target_frames=125)
        with self.assertRaisesRegex(ValueError, "at least"):
            plan_native_guide_bridge(20, 70)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            plan_native_guide_bridge(73, 73, guide_frames=73, target_frames=124)

    def test_rejects_tampered_plan_and_wrong_generated_length(self):
        plan = plan_native_guide_bridge(60, 70)
        tampered = dict(plan, target_frames=141)
        with self.assertRaisesRegex(ValueError, "hash"):
            extract_native_guide_bridge_sources(self.left, self.right, tampered)
        with self.assertRaisesRegex(ValueError, "expected 124"):
            apply_native_guide_bridge(
                self.left,
                self.right,
                np.zeros((123, 1, 1, 1), dtype=np.float32),
                plan,
            )


if __name__ == "__main__":
    unittest.main()
