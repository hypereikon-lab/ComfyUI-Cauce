import math
import sys
import types
import unittest
from unittest.mock import patch

import torch

from cauce.motion import affine_motion_map
from cauce.sigma_transport import (
    SigmaMotionSampler,
    sigma_schedule_increments,
    sigma_schedule_series,
    warp_h3_video_step,
)


class _FakeSamplerFunction:
    __name__ = "sample_res_multistep"


class _FakeSampler:
    sampler_function = _FakeSamplerFunction()


class _ExecutingFakeSampler(_FakeSampler):
    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        state = noise.clone()
        for index in range(len(sigmas) - 1):
            model_wrap(state, sigmas[index], denoise_mask=denoise_mask)
        return state


class SigmaTransportTests(unittest.TestCase):
    def test_accumulate_schedule_reaches_strength_and_increments_sum_to_it(self):
        parameters = {
            "start_percent": 0.2,
            "end_percent": 0.8,
            "strength": 0.3,
            "envelope": "accumulate",
            "easing": "smoothstep",
        }
        series = sigma_schedule_series(11, **parameters)
        increments = sigma_schedule_increments(11, **parameters)
        self.assertEqual(series[0], 0.0)
        self.assertAlmostEqual(series[-1], 0.3)
        self.assertAlmostEqual(sum(increments), 0.3)
        self.assertTrue(all(a <= b for a, b in zip(series, series[1:])))

    def test_pulse_schedule_returns_to_identity(self):
        parameters = {
            "start_percent": 0.1,
            "end_percent": 0.9,
            "strength": 0.4,
            "envelope": "pulse",
            "easing": "cosine",
        }
        series = sigma_schedule_series(21, **parameters)
        increments = sigma_schedule_increments(21, **parameters)
        self.assertEqual(series[0], 0.0)
        self.assertAlmostEqual(max(series), 0.4)
        self.assertEqual(series[-1], 0.0)
        self.assertAlmostEqual(sum(increments), 0.0)

    def test_identity_map_is_identity_for_any_increment(self):
        video = torch.randn(1, 4, 5, 6, 8)
        identity = affine_motion_map(17, 6, 8)
        warped = warp_h3_video_step(video, identity, 0.7)
        torch.testing.assert_close(warped, video, atol=2e-6, rtol=2e-6)

    def test_nonzero_map_changes_only_the_supplied_visual_tensor(self):
        video = torch.arange(1 * 2 * 5 * 6 * 8, dtype=torch.float32).reshape(1, 2, 5, 6, 8)
        motion = affine_motion_map(
            17,
            6,
            8,
            translate_x_end=20.0,
            easing="linear",
        )
        warped = warp_h3_video_step(video, motion, 0.25)
        self.assertEqual(tuple(warped.shape), tuple(video.shape))
        self.assertGreater(float(torch.mean(torch.abs(warped - video))), 0.01)
        self.assertTrue(torch.isfinite(warped).all())

    def test_wrapper_rejects_unsupported_solver(self):
        sampler = _FakeSampler()
        motion = affine_motion_map(5, 4, 6)
        built = SigmaMotionSampler(sampler, motion)
        self.assertEqual(built.sampler_name, "sample_res_multistep")

        class Unsupported:
            sampler_function = lambda: None

        with self.assertRaises(ValueError):
            SigmaMotionSampler(Unsupported(), motion)

    def test_wrapper_transports_packed_video_and_preserves_audio(self):
        video = torch.arange(1 * 2 * 5 * 6 * 8, dtype=torch.float32).reshape(
            1, 2, 5, 6, 8
        )
        audio = torch.randn(1, 3, 4, 7)
        shapes = [tuple(video.shape), tuple(audio.shape)]

        def pack_latents(streams):
            packed = torch.cat([stream.reshape(1, -1) for stream in streams], dim=1)
            return packed, [tuple(stream.shape) for stream in streams]

        def unpack_latents(packed, latent_shapes):
            streams = []
            offset = 0
            for shape in latent_shapes:
                count = math.prod(shape)
                streams.append(packed[:, offset : offset + count].reshape(shape))
                offset += count
            return streams

        comfy = types.ModuleType("comfy")
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.pack_latents = pack_latents
        comfy_utils.unpack_latents = unpack_latents
        comfy.utils = comfy_utils

        class Inner:
            latent_shapes = shapes

        class ModelWrap:
            inner_model = Inner()

            def __call__(self, state, sigma, **kwargs):
                return state

        packed, _ = pack_latents([video, audio])
        motion = affine_motion_map(
            17,
            6,
            8,
            translate_x_end=20.0,
            easing="linear",
        )
        sampler = SigmaMotionSampler(
            _ExecutingFakeSampler(),
            motion,
            start_percent=0.0,
            end_percent=0.75,
            strength=0.25,
        )
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.utils": comfy_utils}):
            result = sampler.sample(
                ModelWrap(),
                torch.linspace(1.0, 0.0, 5),
                {},
                None,
                packed,
                packed.clone(),
            )
        video_out, audio_out = unpack_latents(result, shapes)
        self.assertGreater(float(torch.mean(torch.abs(video_out - video))), 0.01)
        torch.testing.assert_close(audio_out, audio, atol=0.0, rtol=0.0)
        self.assertEqual(sampler._step_index, 4)

    def test_schedule_rejects_inverted_window(self):
        with self.assertRaises(ValueError):
            sigma_schedule_series(
                10,
                start_percent=0.8,
                end_percent=0.2,
                strength=0.2,
                envelope="accumulate",
                easing="linear",
            )


if __name__ == "__main__":
    unittest.main()
