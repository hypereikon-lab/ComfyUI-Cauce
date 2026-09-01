import unittest

try:
    import numpy as np
except ImportError:
    np = None

from cauce.av_latent import (
    apply_av_denoise_interval,
    apply_video_denoise_mask,
    allocate_av_window_like,
    append_av_span,
    build_av_span_keyframes,
    clear_av_denoise_mask,
    densify_h3_video_tokens,
    extract_av_span,
    expand_av_canvas,
    extract_h3_visual_stream,
    inspect_av_latent,
    place_av_span,
    plan_av_window,
    plan_h3_temporal_densification,
    replace_av_span,
    replace_h3_video_stream,
    resize_h3_av_latent,
    split_av_latent,
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

    def test_extracts_visual_stream_without_losing_the_av_carrier_contract(self):
        visual, report = extract_h3_visual_stream(self.previous)
        self.assertEqual(visual["samples"].shape, (1, 24, 72, 2, 3))
        self.assertIsNot(visual["samples"], self.previous["samples"][0])
        self.assertTrue(report["audio_preserved_in_source_carrier"])
        self.assertTrue(report["requires_explicit_graft"])
        self.assertEqual(report["frame_count"], 243)

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

    def test_splits_a_cumulative_state_into_branchable_prefix_and_suffix(self):
        prefix, suffix, prefix_frames, suffix_frames = split_av_latent(
            self.previous,
            cut_frame=124,
        )
        self.assertEqual(prefix_frames, 124)
        self.assertEqual(suffix_frames, 119)
        self.assertEqual(prefix["samples"][0].shape[2], 37)
        self.assertEqual(prefix["samples"][1].shape[-1], 207)
        _, _, descriptor = validate_av_span(suffix)
        self.assertEqual(descriptor["global_start_frame"], 124)
        self.assertEqual(descriptor["global_end_frame"], 243)

        reconstructed, total = append_av_span(prefix, suffix)
        self.assertEqual(total, 243)
        np.testing.assert_array_equal(reconstructed["samples"][0], self.previous["samples"][0])
        np.testing.assert_array_equal(reconstructed["samples"][1], self.previous["samples"][1])
        with self.assertRaisesRegex(ValueError, r"17k\+5"):
            split_av_latent(self.previous, cut_frame=123)
        with self.assertRaisesRegex(ValueError, "non-empty suffix"):
            split_av_latent(self.previous, cut_frame=243)

    def test_places_native_spans_and_makes_rebase_explicit(self):
        source = self.latent(124, value=7.0)
        span = extract_av_span(source, start_frame=0, frame_count=124)
        target = self.latent(243, value=0.0)
        placed, report = place_av_span(target, span, target_frame_idx=119)
        self.assertTrue(report["rebased"])
        self.assertEqual(report["target_frame_range"], [119, 243])
        self.assertEqual(report["source_global_range"], [0, 124])
        self.assertEqual(placed["samples"][0].shape, target["samples"][0].shape)
        self.assertTrue(np.all(placed["samples"][0][:, :, 35:] == 7.0))
        self.assertTrue(np.all(placed["samples"][1][..., 198:] == 7.0))

        with self.assertRaisesRegex(ValueError, "audio tokens do not align"):
            place_av_span(target, span, target_frame_idx=34)

    def test_builds_hard_and_continuous_masks_on_both_native_clocks(self):
        target = self.latent(141, origin=221)
        masked, report = apply_av_denoise_interval(
            target,
            timeline_origin_frame=221,
            start_frame=22,
            frame_count=119,
        )
        video_mask, audio_mask = masked["noise_mask"]
        self.assertEqual(video_mask.shape, (1, 1, 42, 2, 3))
        self.assertEqual(audio_mask.shape, (1, 1, 2, 235))
        self.assertTrue(np.all(video_mask[:, :, :7] == 0.0))
        self.assertTrue(np.all(video_mask[:, :, 7:] == 1.0))
        self.assertTrue(np.all(audio_mask[..., :37] == 0.0))
        self.assertTrue(np.all(audio_mask[..., 37:] == 1.0))
        self.assertEqual(report["denoise_range"], [22, 141])
        self.assertEqual(report["curve"], "smoothstep")

        feathered, feather_report = apply_av_denoise_interval(
            self.latent(124),
            start_frame=39,
            frame_count=51,
            fade_in_frames=17,
            fade_out_frames=17,
            curve="smootherstep",
        )
        feather_video, feather_audio = feathered["noise_mask"]
        self.assertTrue(np.any((feather_video > 0.0) & (feather_video < 1.0)))
        self.assertTrue(np.any((feather_audio > 0.0) & (feather_audio < 1.0)))
        self.assertEqual(feather_report["video_profile"]["maximum"], 1.0)
        self.assertEqual(feather_report["audio_profile"]["maximum"], 1.0)

    def test_mask_composition_replacement_and_clear_are_deterministic(self):
        base = self.latent(243, value=1.0)
        first, _ = apply_av_denoise_interval(
            base,
            start_frame=124,
            frame_count=51,
        )
        combined, _ = apply_av_denoise_interval(
            first,
            start_frame=141,
            frame_count=17,
            combine="multiply",
        )
        combined_video, _ = combined["noise_mask"]
        self.assertTrue(np.all(combined_video[:, :, :37] == 0.0))
        self.assertTrue(np.any(combined_video[:, :, 37:] == 1.0))

        generated = self.latent(243, value=9.0)
        replacement = extract_av_span(generated, start_frame=124, frame_count=51)
        replaced, report = replace_av_span(base, replacement)
        self.assertEqual(report["replaced_global_range"], [124, 175])
        self.assertNotIn("noise_mask", replaced)
        video_start = 37
        video_end = 52
        self.assertTrue(np.all(replaced["samples"][0][:, :, :video_start] == 1.0))
        self.assertTrue(np.all(replaced["samples"][0][:, :, video_start:video_end] == 9.0))
        self.assertTrue(np.all(replaced["samples"][0][:, :, video_end:] == 1.0))

        cleared, removed = clear_av_denoise_mask(first)
        self.assertTrue(removed)
        self.assertNotIn("noise_mask", cleared)
        _, removed_again = clear_av_denoise_mask(cleared)
        self.assertFalse(removed_again)

        nonzero = self.latent(56, origin=17)
        nonzero["noise_mask"] = nonzero["samples"]
        clean_nonzero, removed_nonzero = clear_av_denoise_mask(
            nonzero,
            timeline_origin_frame=17,
        )
        self.assertTrue(removed_nonzero)
        self.assertNotIn("noise_mask", clean_nonzero)

    def test_projects_static_and_animated_masks_onto_native_video_tokens(self):
        latent = self.latent(124)
        static = np.zeros((2, 3), dtype=np.float32)
        static[:, 1:] = 0.75
        masked, report = apply_video_denoise_mask(
            latent,
            static,
            start_frame=0,
            frame_count=124,
        )
        video_mask, audio_mask = masked["noise_mask"]
        self.assertEqual(video_mask.shape, (1, 1, 37, 2, 3))
        self.assertEqual(audio_mask.shape, (1, 1, 2, 207))
        self.assertTrue(np.all(video_mask[:, :, :, :, 0] == 0.0))
        self.assertTrue(np.all(video_mask[:, :, :, :, 1:] == 0.75))
        self.assertTrue(np.all(audio_mask == 0.0))
        self.assertEqual(report["temporal_projection"], "static")
        self.assertEqual(report["video_token_range"], [0, 37])
        self.assertRegex(report["result_digest"], r"^[0-9a-f]{64}$")

        animated = np.zeros((124, 2, 3), dtype=np.float32)
        animated[1:5] = 0.5
        animated[5:] = 1.0
        projected, animated_report = apply_video_denoise_mask(
            latent,
            animated,
            start_frame=0,
            frame_count=124,
        )
        projected_video, _ = projected["noise_mask"]
        self.assertTrue(np.all(projected_video[:, :, 0] == 0.0))
        self.assertTrue(np.all(projected_video[:, :, 1] == 0.5))
        self.assertTrue(np.all(projected_video[:, :, 2:] == 1.0))
        self.assertEqual(
            animated_report["temporal_projection"],
            "amax-per-h3-visual-token",
        )

    def test_composes_spatial_and_temporal_masks_without_touching_latents(self):
        latent = self.latent(124, value=3.0)
        interval, _ = apply_av_denoise_interval(
            latent,
            start_frame=39,
            frame_count=51,
            inside_strength_audio=0.0,
        )
        spatial = np.zeros((2, 3), dtype=np.float32)
        spatial[:, 1:] = 1.0
        combined, _ = apply_video_denoise_mask(
            interval,
            spatial,
            start_frame=39,
            frame_count=51,
            combine="multiply",
        )
        video_mask, audio_mask = combined["noise_mask"]
        self.assertTrue(np.all(video_mask[:, :, :12] == 0.0))
        self.assertTrue(np.all(video_mask[:, :, 12:27, :, 0] == 0.0))
        self.assertTrue(np.all(video_mask[:, :, 12:27, :, 1:] == 1.0))
        self.assertTrue(np.all(video_mask[:, :, 27:] == 0.0))
        self.assertTrue(np.all(audio_mask == 0.0))
        np.testing.assert_array_equal(combined["samples"][0], latent["samples"][0])
        np.testing.assert_array_equal(combined["samples"][1], latent["samples"][1])

        with self.assertRaisesRegex(ValueError, "one static mask"):
            apply_video_denoise_mask(
                latent,
                np.zeros((2, 2, 3), dtype=np.float32),
                start_frame=0,
                frame_count=124,
            )

    def test_expands_h3_canvas_with_exact_placement_and_outpaint_mask(self):
        frames = 124
        video = np.full((1, 24, 37, 4, 6), 7.0, dtype=np.float32)
        audio = np.full((1, 32, 2, 207), 5.0, dtype=np.float32)
        source = {"samples": (video, audio)}
        expanded, report = expand_av_canvas(
            source,
            target_width=160,
            target_height=128,
            offset_x=32,
            offset_y=32,
        )
        expanded_video, expanded_audio = expanded["samples"]
        video_mask, audio_mask = expanded["noise_mask"]
        self.assertEqual(expanded_video.shape, (1, 24, 37, 8, 10))
        self.assertEqual(expanded_audio.shape, audio.shape)
        self.assertTrue(np.all(expanded_video[:, :, :, 2:6, 2:8] == 7.0))
        self.assertTrue(np.all(expanded_video[:, :, :, :2] == 0.0))
        self.assertTrue(np.all(video_mask[:, :, :, 2:6, 2:8] == 0.0))
        self.assertTrue(np.all(video_mask[:, :, :, :2] == 1.0))
        self.assertTrue(np.all(audio_mask == 0.0))
        np.testing.assert_array_equal(expanded_audio, audio)
        self.assertEqual(report["frame_count"], frames)
        self.assertEqual(report["source_offset"], {"x": 32, "y": 32})
        self.assertRegex(report["expansion_hash"], r"^[0-9a-f]{64}$")

        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            expand_av_canvas(
                source,
                target_width=150,
                target_height=128,
                offset_x=32,
                offset_y=32,
            )
        source_with_mask = dict(source, noise_mask=(video[:, :1], audio[:, :1]))
        with self.assertRaisesRegex(ValueError, "clear the existing"):
            expand_av_canvas(
                source_with_mask,
                target_width=160,
                target_height=128,
                offset_x=32,
                offset_y=32,
            )

    def test_plans_and_builds_native_h3_temporal_densification(self):
        source = self.latent(124)
        source_video, _ = source["samples"]
        for index in range(source_video.shape[2]):
            source_video[:, :, index] = float(index + 1)

        plan = plan_h3_temporal_densification(124, 2)
        self.assertEqual(plan["delivery_frame_count"], 247)
        self.assertEqual(plan["h3_target_frame_count"], 260)
        self.assertEqual(plan["source_video_tokens"], 37)
        self.assertEqual(plan["h3_target_video_tokens"], 77)
        self.assertEqual(plan["delivery_fps"], 48)
        self.assertEqual(plan["decoded_tail_trim_frames"], 13)
        self.assertTrue(plan["inside_h3_trained_frame_range"])

        densified, report = densify_h3_video_tokens(
            source,
            factor=2,
            feather_tokens=1,
        )
        video, audio = densified["samples"]
        video_mask, audio_mask = densified["noise_mask"]
        self.assertEqual(video.shape, (1, 24, 77, 2, 3))
        self.assertEqual(audio.shape, (1, 32, 2, 433))
        self.assertEqual(video_mask.shape, (1, 1, 77, 2, 3))
        self.assertEqual(audio_mask.shape, (1, 1, 2, 433))
        for anchor in report["anchors"]:
            source_index = anchor["source_token"]
            target_index = anchor["target_token"]
            np.testing.assert_array_equal(
                video[:, :, target_index],
                source_video[:, :, source_index],
            )
            self.assertTrue(np.all(video_mask[:, :, target_index] == 0.0))
        for target_index in report["generated_target_tokens"]:
            self.assertTrue(np.all(video[:, :, target_index] == 0.0))
            self.assertTrue(np.all(video_mask[:, :, target_index] == 1.0))
        self.assertTrue(np.all(audio == 0.0))
        self.assertTrue(np.all(audio_mask == 1.0))

    def test_resizes_only_h3_visual_state_for_same_model_second_pass(self):
        source = self.latent(124, value=2.0)
        resized, report = resize_h3_av_latent(
            source,
            target_width=96,
            target_height=64,
            method="bicubic",
            video_denoise=0.35,
            audio_denoise=0.0,
        )
        video, audio = resized["samples"]
        video_mask, audio_mask = resized["noise_mask"]
        self.assertEqual(video.shape, (1, 24, 37, 4, 6))
        self.assertEqual(audio.shape, source["samples"][1].shape)
        self.assertTrue(np.all(video == 2.0))
        np.testing.assert_array_equal(audio, source["samples"][1])
        self.assertTrue(np.all(video_mask == np.float32(0.35)))
        self.assertTrue(np.all(audio_mask == 0.0))
        self.assertEqual(report["source_width"], 48)
        self.assertEqual(report["source_height"], 32)
        self.assertEqual(report["target_width"], 96)
        self.assertEqual(report["target_height"], 64)

    def test_grafts_vae_visual_state_onto_compatible_h3_carrier(self):
        carrier = self.latent(124, value=1.0)
        encoded = {
            "samples": np.full((1, 24, 37, 4, 6), 8.0, dtype=np.float32)
        }
        grafted, report = replace_h3_video_stream(
            carrier,
            encoded,
            video_denoise=0.25,
            audio_denoise=0.0,
        )
        video, audio = grafted["samples"]
        video_mask, audio_mask = grafted["noise_mask"]
        self.assertEqual(video.shape, (1, 24, 37, 4, 6))
        self.assertTrue(np.all(video == 8.0))
        np.testing.assert_array_equal(audio, carrier["samples"][1])
        self.assertTrue(np.all(video_mask == np.float32(0.25)))
        self.assertTrue(np.all(audio_mask == 0.0))
        self.assertEqual(report["method"], "pixel-vae-second-pass")

        malformed = {"samples": np.zeros((1, 24, 36, 4, 6), dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "duration differs"):
            replace_h3_video_stream(carrier, malformed)


if __name__ == "__main__":
    unittest.main()
