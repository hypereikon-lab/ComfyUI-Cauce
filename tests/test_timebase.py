import unittest

from cauce.timebase import (
    H3Shape,
    h3_audio_latent_frames,
    h3_visual_latent_frames,
    is_h3_frame_count,
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

    def test_context_windows_end_on_token_boundary(self):
        self.assertEqual(visual_token_count_for_span(5), 2)
        self.assertEqual(visual_token_count_for_span(22), 7)
        self.assertEqual(visual_token_count_for_span(39), 12)
        self.assertEqual(visual_token_count_for_span(56), 17)
        with self.assertRaises(ValueError):
            visual_token_count_for_span(6)


if __name__ == "__main__":
    unittest.main()
