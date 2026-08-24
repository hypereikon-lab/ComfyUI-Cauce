import unittest

import numpy as np

from cauce.motion import (
    _sample_numpy,
    WarpedH3Noise,
    affine_motion_map,
    analytic_motion_map,
    compose_motion_maps,
    depth_camera_motion_map,
    displacement_motion_map,
    identity_grid,
    integrate_advection,
    modulate_motion_map,
    motion_map_report,
    perspective_motion_map,
    resample_motion_map,
    validate_motion_map,
    vector_field,
)


class MotionMapTests(unittest.TestCase):
    def test_identity_affine_is_exact_and_fully_valid(self):
        value = affine_motion_map(9, 12, 16)
        expected = identity_grid(9, 12, 16)
        np.testing.assert_array_equal(value["grid"], expected)
        self.assertGreater(float(value["validity"].min()), 0.99999)
        validate_motion_map(value)

    def test_affine_translation_uses_pullback_coordinates(self):
        value = affine_motion_map(
            2,
            10,
            20,
            translate_x_start=0.0,
            translate_x_end=10.0,
            easing="linear",
        )
        identity = identity_grid(2, 10, 20)
        np.testing.assert_allclose(value["grid"][0], identity[0], atol=1e-6)
        # A positive forward translation pulls source samples from the left.
        np.testing.assert_allclose(
            value["grid"][-1, ..., 0], identity[-1, ..., 0] - 0.2, atol=1e-6
        )
        self.assertLess(float(value["validity"][-1].mean()), 0.91)

    def test_sine_loop_closes_affine_and_advection_maps(self):
        affine = affine_motion_map(
            17,
            12,
            16,
            scale_start=1.0,
            scale_end=1.8,
            rotation_end=90.0,
            easing="sine_loop",
        )
        np.testing.assert_allclose(affine["grid"][0], affine["grid"][-1], atol=1e-6)

        field = vector_field(
            17,
            12,
            16,
            kind="uniform",
            duration_seconds=2.0,
            speed_x_percent=12.0,
            temporal_mode="sine_loop",
        )
        advected = integrate_advection(field, method="rk4")
        np.testing.assert_allclose(
            advected["grid"][-1], identity_grid(1, 12, 16)[0], atol=1e-6
        )

    def test_composition_matches_two_sequential_samples(self):
        height, width = 18, 24
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, height, dtype=np.float32),
            np.linspace(0.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        image = np.stack((xx, yy, xx * yy), axis=-1)
        first = affine_motion_map(
            1, height, width, translate_x_end=8.0, rotation_end=9.0, easing="linear"
        )
        second = analytic_motion_map(
            1, height, width, mode="wave", amount_start=4.0, amount_end=4.0
        )
        composed = compose_motion_maps(first, second)
        sequential = _sample_numpy(
            _sample_numpy(image, first["grid"][0], "border"),
            second["grid"][0],
            "border",
        )
        one_sample = _sample_numpy(image, composed["grid"][0], "border")
        # Two image resamples blur differently at the boundary; map geometry is
        # nevertheless equivalent within one source pixel.
        self.assertLess(float(np.mean(np.abs(sequential - one_sample))), 0.01)

    def test_perspective_identity_and_corner_pin(self):
        identity = perspective_motion_map(2, 12, 16, easing="linear")
        np.testing.assert_allclose(identity["grid"], identity_grid(2, 12, 16), atol=1e-6)
        pin = perspective_motion_map(
            2,
            12,
            16,
            top_left_x_end=20.0,
            bottom_left_x_end=20.0,
            easing="linear",
        )
        self.assertGreater(
            float(np.mean(np.abs(pin["grid"][-1] - identity["grid"][-1]))), 0.05
        )

    def test_external_displacement_and_modulation_are_general(self):
        encoded = np.zeros((1, 5, 7, 3), dtype=np.float32)
        encoded[..., 0] = 1.0
        encoded[..., 1] = 0.5
        motion = displacement_motion_map(
            encoded, 3, 10, 14, magnitude_percent=10.0, encoding="centered_rg"
        )
        identity = identity_grid(3, 10, 14)
        np.testing.assert_allclose(
            motion["grid"][..., 0], identity[..., 0] + 0.2, atol=1e-6
        )
        ramped = modulate_motion_map(
            motion, strength_start=0.0, strength_end=1.0, easing="linear"
        )
        np.testing.assert_allclose(ramped["grid"][0], identity[0], atol=1e-6)
        np.testing.assert_allclose(ramped["grid"][-1], motion["grid"][-1], atol=1e-6)

        mask = np.zeros((10, 14), dtype=np.float32)
        mask[:, 7:] = 1.0
        localized = modulate_motion_map(
            motion, strength_start=1.0, strength_end=1.0, spatial_mask=mask
        )
        np.testing.assert_allclose(localized["grid"][:, :, :7], identity[:, :, :7], atol=1e-6)
        np.testing.assert_allclose(localized["grid"][:, :, 7:], motion["grid"][:, :, 7:], atol=1e-6)

    def test_resampling_preserves_endpoints_and_metadata(self):
        source = affine_motion_map(5, 8, 12, scale_end=1.4, rotation_end=20.0)
        target = resample_motion_map(source, 13, 16, 24)
        self.assertEqual(target["grid"].shape, (13, 16, 24, 2))
        self.assertEqual(target["validity"].shape, (13, 16, 24))
        report = motion_map_report(target)
        self.assertEqual(report["frames"], 13)
        self.assertEqual(report["operation"], "resample:affine")

    def test_depth_reprojection_starts_at_identity_and_marks_disocclusions(self):
        depth = np.tile(np.linspace(0.0, 1.0, 24, dtype=np.float32), (16, 1))
        value = depth_camera_motion_map(
            depth,
            9,
            16,
            24,
            translate_x_end=20.0,
            translate_z_end=20.0,
        )
        np.testing.assert_allclose(
            value["grid"][0], identity_grid(1, 16, 24)[0], atol=1e-6
        )
        self.assertGreater(float(value["validity"][0].mean()), 0.999)
        self.assertLess(float(value["validity"][-1].mean()), 0.95)

    def test_deterministic_hash_and_invalid_depth_planes(self):
        first = analytic_motion_map(7, 8, 12, mode="swirl", amount_end=45.0)
        second = analytic_motion_map(7, 8, 12, mode="swirl", amount_end=45.0)
        self.assertEqual(first["tensor_hash"], second["tensor_hash"])
        with self.assertRaises(ValueError):
            depth_camera_motion_map(np.ones((4, 4)), 2, 4, 4, near=10.0, far=1.0)

    def test_warped_noise_rejects_invalid_temporal_correlation(self):
        motion = affine_motion_map(5, 4, 6)
        self.assertEqual(WarpedH3Noise(7, motion, temporal_correlation=0.85).seed, 7)
        with self.assertRaises(ValueError):
            WarpedH3Noise(7, motion, temporal_correlation=1.01)


if __name__ == "__main__":
    unittest.main()
