import unittest

from cauce.latent_injection import resolve_injection_step, validate_flow_injection

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from cauce.latent_injection import (
        flow_preserving_video_injection,
        project_visible_mask_to_h3,
    )


class H3FlowLatentInjectionContractTests(unittest.TestCase):
    def test_step_resolution_uses_flow_coordinate_and_leaves_an_evaluation(self):
        sigmas = [1.0, 0.95, 0.80, 0.55, 0.20, 0.0]
        self.assertEqual(resolve_injection_step(sigmas, 0.0), 0)
        self.assertEqual(resolve_injection_step(sigmas, 0.45), 2)
        self.assertEqual(resolve_injection_step(sigmas, 1.0), 3)
        with self.assertRaises(ValueError):
            resolve_injection_step([1.0, 0.0], 0.5)

    def test_h3_shifted_simple_schedule_does_not_treat_mid_step_as_mid_flow(self):
        sigmas = [
            12.0 * (1.0 - 0.05 * i) / (1.0 + 11.0 * (1.0 - 0.05 * i))
            for i in range(20)
        ] + [0.0]
        self.assertEqual(resolve_injection_step(sigmas, 0.45), 17)
        self.assertAlmostEqual(1.0 - sigmas[18], 3.0 / 7.0)

    def test_percent_and_strength_are_bounded(self):
        validate_flow_injection(0.0, 0.0)
        validate_flow_injection(1.0, 1.0)
        for progress, strength in ((-0.1, 0.0), (1.1, 0.0), (0.5, -0.1), (0.5, 1.1)):
            with self.subTest(progress=progress, strength=strength):
                with self.assertRaises(ValueError):
                    validate_flow_injection(progress, strength)


@unittest.skipIf(torch is None, "PyTorch is supplied by ComfyUI, not CAUCE")
class H3FlowLatentInjectionTests(unittest.TestCase):
    def test_zero_strength_is_exact_identity(self):
        current = torch.randn(1, 3, 4, 5, 6)
        clean = torch.randn_like(current)
        guide = torch.randn_like(current)
        mask = torch.ones(1, 1, 4, 5, 6)
        result = flow_preserving_video_injection(
            current, clean, guide, mask, sigma=0.4, strength=0.0
        )
        self.assertIs(result, current)

    def test_full_injection_preserves_the_implied_noise_endpoint(self):
        clean = torch.randn(1, 2, 3, 4, 5)
        guide = torch.randn_like(clean)
        noise = torch.randn_like(clean)
        sigma = 0.35
        current = sigma * noise + (1.0 - sigma) * clean
        mask = torch.ones(1, 1, 3, 4, 5)
        result = flow_preserving_video_injection(
            current, clean, guide, mask, sigma=sigma, strength=1.0
        )
        expected = sigma * noise + (1.0 - sigma) * guide
        torch.testing.assert_close(result, expected)

    def test_fractional_mask_is_local_and_continuous(self):
        current = torch.zeros(1, 1, 2, 2, 3)
        clean = torch.zeros_like(current)
        guide = torch.ones_like(current)
        mask = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 1, 1, 3)
        mask = mask.expand(1, 1, 2, 2, 3)
        result = flow_preserving_video_injection(
            current, clean, guide, mask, sigma=0.25, strength=0.8
        )
        torch.testing.assert_close(result, mask * 0.6)

    def test_visible_mask_projects_over_the_h3_causal_token_spans(self):
        # 7 H3 visual tokens cover 22 visible frames: 1,4,4,4,4,1,4.
        mask = torch.zeros(22, 2, 3)
        mask[1:5] = 0.5
        mask[5:9] = 1.0
        projected = project_visible_mask_to_h3(
            mask, tokens=7, height=2, width=3, projection="mean"
        )
        self.assertEqual(tuple(projected.shape), (1, 1, 7, 2, 3))
        torch.testing.assert_close(
            projected[0, 0, :, 0, 0],
            torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
