from fractions import Fraction
import unittest

from cauce.timebase import (
    H3Shape,
    frame_to_sample,
    format_timecode,
    h3_audio_latent_frames,
    h3_av_boundaries,
    h3_visual_latent_frames,
    is_h3_av_boundary,
    is_h3_frame_count,
    reduce_intervals,
    snap_h3_frame_count,
    visual_token_count_for_span,
    visual_token_spans,
)


class TimebaseTests(unittest.TestCase):
    def test_h3_grid_and_shapes(self):
        expected = {
            5: (2, 8),
            22: (7, 37),
            39: (12, 65),
            124: (37, 207),
            362: (107, 603),
        }
        for frames, (video_t, audio_t) in expected.items():
            with self.subTest(frames=frames):
                self.assertTrue(is_h3_frame_count(frames))
                self.assertEqual(h3_visual_latent_frames(frames), video_t)
                self.assertEqual(h3_audio_latent_frames(frames), audio_t)
                shape = H3Shape.from_frames(frames)
                self.assertEqual(shape.video_latent_frames, video_t)
                self.assertEqual(shape.audio_latent_frames, audio_t)
                self.assertEqual(visual_token_spans(video_t)[-1][1], frames)

    def test_snap_is_explicit(self):
        self.assertEqual(snap_h3_frame_count(5, unit="seconds", mode="ceil"), 124)
        self.assertEqual(snap_h3_frame_count(124, unit="frames", mode="nearest"), 124)
        self.assertEqual(snap_h3_frame_count(125, unit="frames", mode="floor"), 124)
        self.assertEqual(snap_h3_frame_count(125, unit="frames", mode="ceil"), 141)

    def test_context_windows_end_on_token_boundary(self):
        self.assertEqual(visual_token_count_for_span(5), 2)
        self.assertEqual(visual_token_count_for_span(22), 7)
        self.assertEqual(visual_token_count_for_span(39), 12)
        self.assertEqual(visual_token_count_for_span(56), 17)
        with self.assertRaises(ValueError):
            visual_token_count_for_span(6)

    def test_joint_av_context_uses_shared_clock_boundaries(self):
        self.assertEqual(h3_av_boundaries(200), (39, 90, 141, 192))
        self.assertTrue(is_h3_av_boundary(39))
        self.assertFalse(is_h3_av_boundary(22))

    def test_frame_sample_mapping_has_no_float_path(self):
        self.assertEqual(frame_to_sample(24, 32000), 32000)
        self.assertEqual(frame_to_sample(1, 32000), 1333)
        self.assertEqual(frame_to_sample(3, 32000), 4000)

    def test_timecode(self):
        self.assertEqual(format_timecode(Fraction(1, 1)), "00:00:01:00")
        self.assertEqual(format_timecode(Fraction(25, 24)), "00:00:01:01")

    def test_interval_spans_override_default(self):
        targets = ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(2)))
        sources = ((Fraction(0), Fraction(1), 0.0),)
        self.assertEqual(reduce_intervals(targets, sources, 1.0), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
