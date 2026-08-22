from fractions import Fraction
import unittest

from cauce.contracts import (
    FIELD_SCHEMA,
    append_field_span,
    append_timeline_item,
    make_point,
    make_span,
    make_timeline,
    make_window,
    range_fraction,
)


class ContractTests(unittest.TestCase):
    def test_point_keeps_arbitrary_prompt(self):
        prompt = "anything at all — no ontology"
        point = make_point("p 01", "3/2", prompt)
        self.assertEqual(point["id"], "p-01")
        self.assertEqual(point["prompt"], prompt)
        self.assertEqual(point["time"], {"numerator": 3, "denominator": 2})

    def test_window_separates_hidden_head_and_accepted_media(self):
        window = make_window(
            "w1",
            10,
            5,
            context_frames=39,
            duplicate_prefix_frames=5,
            accept_mode="exact_frames",
        )
        render_start, _ = range_fraction(window["render_range"])
        accepted_start, accepted_end = range_fraction(window["accepted_range"])
        context_start, context_end = range_fraction(window["context_range"])
        prefix_start, prefix_end = range_fraction(window["duplicate_prefix_range"])
        self.assertEqual(render_start, Fraction(67, 8))
        self.assertEqual((context_start, context_end), (Fraction(67, 8), Fraction(10)))
        self.assertEqual((prefix_start, prefix_end), (Fraction(67, 8), Fraction(103, 12)))
        self.assertEqual(accepted_start, Fraction(10))
        self.assertEqual(accepted_end, Fraction(15))
        self.assertEqual(window["accepted_offset_frames"], 39)
        self.assertEqual(window["accepted_start_frame"], 39)
        self.assertEqual(window["accepted_end_frame"], 159)
        self.assertEqual(window["accepted_frames"], 120)
        self.assertFalse(window["phase_safe_parent"])
        self.assertEqual(window["shape"]["pixel_frames"], 175)

    def test_window_defaults_to_phase_safe_parent_acceptance(self):
        window = make_window("w1", 10, 5, context_frames=39)
        accepted_start, accepted_end = range_fraction(window["accepted_range"])
        self.assertEqual(window["accept_mode"], "nearest_run")
        self.assertEqual(window["accepted_start_frame"], 39)
        self.assertEqual(window["accepted_end_frame"], 158)
        self.assertEqual(window["accepted_frames"], 119)
        self.assertTrue(window["phase_safe_parent"])
        self.assertEqual(accepted_start, Fraction(10))
        self.assertEqual(accepted_end, Fraction(359, 24))

    def test_window_refuses_hidden_head_before_zero(self):
        with self.assertRaises(ValueError):
            make_window("w", 0, 5, context_frames=39)

    def test_window_accepts_visual_context_without_audio_alignment(self):
        window = make_window("w", 10, 5, context_frames=22)
        self.assertEqual(window["context_frames"], 22)

    def test_window_rejects_context_outside_the_visual_grid(self):
        with self.assertRaises(ValueError):
            make_window("w", 10, 5, context_frames=23)

    def test_field_and_timeline_are_versioned_data(self):
        field = append_field_span(
            None, channel="video", start=1, end=2, strength=0.0
        )
        self.assertEqual(field["schema"], FIELD_SCHEMA)
        timeline = make_timeline("main")
        timeline = append_timeline_item(timeline, make_point("p", 1, ""))
        timeline = append_timeline_item(
            timeline, make_span("s", "video", 1, 2, source="motion.mp4")
        )
        self.assertEqual(len(timeline["points"]), 1)
        self.assertEqual(len(timeline["spans"]), 1)
        self.assertIn("hash", timeline)


if __name__ == "__main__":
    unittest.main()
