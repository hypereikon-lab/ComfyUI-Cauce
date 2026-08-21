from fractions import Fraction
import unittest

from cauce.contracts import append_field_span, make_window
from cauce.h3 import append_reference, empty_reference_set, frame_index_in_window
from cauce.masks import compile_audio_field, compile_video_field


class H3AndMaskTests(unittest.TestCase):
    def test_reference_limits_are_fail_closed(self):
        refs = empty_reference_set()
        for index in range(9):
            refs = append_reference(refs, kind="image", media=f"image-{index}")
        with self.assertRaises(ValueError):
            append_reference(refs, kind="image", media="one-too-many")

    def test_reference_duration_bounds(self):
        with self.assertRaises(ValueError):
            append_reference(
                None, kind="video", media="video", duration_seconds=1.9
            )

    def test_guide_uses_absolute_master_time(self):
        window = make_window("w", 10, 5)
        self.assertEqual(frame_index_in_window(window, 10), 0)
        self.assertEqual(frame_index_in_window(window, Fraction(241, 24)), 1)
        with self.assertRaises(ValueError):
            frame_index_in_window(window, 9)

    def test_fields_compile_on_distinct_video_and_audio_grids(self):
        window = make_window("w", 10, 5)
        field = append_field_span(
            None,
            channel="both",
            start=10,
            end=Fraction(121, 12),  # first two visible frames
            strength=0.0,
        )
        video = compile_video_field(window, field)
        audio = compile_audio_field(window, field)
        self.assertEqual(len(video), window["shape"]["video_latent_frames"])
        self.assertEqual(len(audio), window["shape"]["audio_latent_frames"])
        self.assertEqual(video[0], 0.0)
        self.assertEqual(audio[0], 0.0)
        self.assertEqual(video[-1], 1.0)
        self.assertEqual(audio[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
