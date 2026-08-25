import unittest

try:
    import numpy as np
except ImportError:
    np = None

from cauce.av_latent import (
    allocate_av_window_like,
    append_av_span,
    build_av_span_keyframes,
    extract_av_span,
    inspect_av_latent,
    plan_av_window,
    validate_av_span,
    validate_av_window_layout,
)
from cauce.contracts import content_hash
from cauce.timebase import h3_audio_token_boundary, h3_visual_latent_frames


@unittest.skipIf(np is None, "NumPy is supplied by ComfyUI, not CAUCE")
class AVLatentTests(unittest.TestCase):
    @staticmethod
    def latent(frames, *, origin=0, value=0.0):
        video_tokens = h3_visual_latent_frames(frames)
        audio_tokens = h3_audio_token_boundary(origin + frames) - h3_audio_token_boundary(
            origin
        )
        video = np.full((1, 24, video_tokens, 2, 3), value, dtype=np.float32)
        audio = np.full((1, 32, 2, audio_tokens), value, dtype=np.float32)
        return {"samples": (video, audio)}

    def setUp(self):
        self.previous = self.latent(243, value=1.0)
        self.layout = plan_av_window(
            self.previous,
            overlap_frames=22,
            extension_frames=119,
        )

    def test_inspects_packed_av_latent(self):
        report = inspect_av_latent(self.previous)
        self.assertEqual(report["frame_count"], 243)
        self.assertEqual(report["video_tokens"], 72)
        self.assertEqual(report["audio_tokens"], 405)
        self.assertEqual(report["video_shape"], [1, 24, 72, 2, 3])

    def test_plans_globally_aligned_window(self):
        self.assertEqual(self.layout["window_start_frame"], 221)
        self.assertEqual(self.layout["window_end_frame"], 362)
        self.assertEqual(self.layout["window_frame_count"], 141)
        self.assertEqual(self.layout["target_video_tokens"], 42)
        self.assertEqual(self.layout["overlap_video_tokens"], 7)
        self.assertEqual(self.layout["extension_video_tokens"], 35)
        self.assertEqual(self.layout["target_audio_tokens"], 235)
        self.assertEqual(self.layout["overlap_audio_tokens"], 37)
        self.assertEqual(self.layout["extension_audio_tokens"], 198)
        self.assertEqual(len(self.layout["layout_hash"]), 64)

    def test_allocates_window_on_global_audio_grid(self):
        target = allocate_av_window_like(self.previous, self.layout)
        video, audio = target["samples"]
        self.assertEqual(video.shape, (1, 24, 42, 2, 3))
        self.assertEqual(audio.shape, (1, 32, 2, 235))
        report = inspect_av_latent(target, timeline_origin_frame=221)
        self.assertEqual(report["timeline_end_frame"], 362)

    def test_global_audio_phase_can_differ_from_isolated_duration(self):
        previous = self.latent(22)
        layout = plan_av_window(previous, overlap_frames=5, extension_frames=51)
        self.assertEqual(layout["window_start_frame"], 17)
        self.assertEqual(layout["window_frame_count"], 56)
        self.assertEqual(layout["target_audio_tokens"], 94)
        self.assertEqual(h3_audio_token_boundary(56), 93)

        target = allocate_av_window_like(previous, layout)
        self.assertEqual(target["samples"][1].shape[-1], 94)
        sampled = self.latent(56, origin=17, value=3.0)
        suffix = extract_av_span(
            sampled,
            timeline_origin_frame=17,
            start_frame=5,
            frame_count=51,
        )
        extended, total_frames = append_av_span(previous, suffix)
        self.assertEqual(total_frames, 73)
        self.assertEqual(extended["samples"][1].shape[-1], h3_audio_token_boundary(73))

    def test_extracts_native_tail_and_generated_suffix_as_spans(self):
        tail = extract_av_span(self.previous, start_frame=221, frame_count=22)
        tail_video, tail_audio, tail_descriptor = validate_av_span(tail)
        self.assertEqual(tail_video.shape[2], 7)
        self.assertEqual(tail_audio.shape[-1], 37)
        self.assertEqual(tail_descriptor["global_start_frame"], 221)
        self.assertEqual(tail_descriptor["global_end_frame"], 243)

        sampled = self.latent(141, origin=221, value=2.0)
        suffix = extract_av_span(
            sampled,
            timeline_origin_frame=221,
            start_frame=22,
            frame_count=119,
        )
        suffix_video, suffix_audio, suffix_descriptor = validate_av_span(suffix)
        self.assertEqual(suffix_video.shape[2], 35)
        self.assertEqual(suffix_audio.shape[-1], 198)
        self.assertEqual(suffix_descriptor["global_start_frame"], 243)
        self.assertEqual(suffix_descriptor["global_end_frame"], 362)

    def test_builds_guide_then_appends_only_explicit_span(self):
        target = allocate_av_window_like(self.previous, self.layout)
        tail = extract_av_span(self.previous, start_frame=221, frame_count=22)
        keyframes = build_av_span_keyframes(
            [],
            tail,
            target,
            self.layout,
            target_frame_idx=0,
        )
        self.assertEqual(len(keyframes), 1)
        self.assertEqual(keyframes[0]["resolved_frame_index"], 0)
        self.assertEqual(
            set(keyframes[0]),
            {"resolved_frame_index", "latent", "audio_latent"},
        )
        self.assertEqual(keyframes[0]["latent"].shape[2], 7)
        self.assertEqual(keyframes[0]["audio_latent"].shape[-1], 37)

        sampled = self.latent(141, origin=221, value=2.0)
        suffix = extract_av_span(
            sampled,
            timeline_origin_frame=221,
            start_frame=22,
            frame_count=119,
        )
        extended, total_frames = append_av_span(self.previous, suffix)
        video, audio = extended["samples"]
        self.assertEqual(total_frames, 362)
        self.assertEqual(video.shape[2], 107)
        self.assertEqual(audio.shape[-1], 603)
        np.testing.assert_array_equal(video[:, :, :72], self.previous["samples"][0])
        self.assertTrue(np.all(video[:, :, 72:] == 2.0))

    def test_rejects_drift_and_tampering_but_allows_guide_composition(self):
        tampered = dict(self.layout, target_audio_tokens=234)
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_av_window_layout(tampered)
        tampered["layout_hash"] = content_hash(
            {key: value for key, value in tampered.items() if key != "layout_hash"}
        )
        with self.assertRaisesRegex(ValueError, "target_audio_tokens"):
            validate_av_window_layout(tampered)

        target = allocate_av_window_like(self.previous, self.layout)
        tail = extract_av_span(self.previous, start_frame=221, frame_count=22)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            build_av_span_keyframes(
                [],
                tail,
                target,
                self.layout,
                target_frame_idx=5,
            )
        composed = build_av_span_keyframes(
            [{"resolved_frame_index": 0, "latent": tail["video"]}],
            tail,
            target,
            self.layout,
            target_frame_idx=0,
        )
        self.assertEqual(len(composed), 2)

        tampered_span = dict(tail)
        tampered_descriptor = dict(tail["descriptor"], global_end_frame=244)
        tampered_span["descriptor"] = tampered_descriptor
        tampered_span["descriptor_hash"] = content_hash(tampered_descriptor)
        with self.assertRaisesRegex(ValueError, "global_end_frame"):
            validate_av_span(tampered_span)

        noncontiguous = extract_av_span(
            self.latent(141, origin=221),
            timeline_origin_frame=221,
            start_frame=39,
            frame_count=17,
        )
        with self.assertRaisesRegex(ValueError, "not globally contiguous"):
            append_av_span(self.previous, noncontiguous)

    def test_rejects_invalid_window_lengths_and_audio(self):
        with self.assertRaisesRegex(ValueError, r"17k\+5"):
            plan_av_window(self.previous, overlap_frames=21, extension_frames=119)
        with self.assertRaisesRegex(ValueError, "multiple of 17"):
            plan_av_window(self.previous, overlap_frames=22, extension_frames=120)
        malformed = self.latent(243)
        malformed["samples"] = (malformed["samples"][0], malformed["samples"][1][..., :-1])
        with self.assertRaisesRegex(ValueError, "audio temporal length"):
            inspect_av_latent(malformed)


if __name__ == "__main__":
    unittest.main()
