import math
import unittest
from types import SimpleNamespace

from cauce.h3_zenith_rope import H3ZenithRoPEPatch, hybrid_phases, zenith_ray

try:
    import torch
except ImportError:
    torch = None


class ZenithProjectionTests(unittest.TestCase):
    def test_zenith_golden_vectors(self):
        fixtures = [((.5, .5), (0, 1, 0)), ((1, .5), (1, 0, 0)),
                    ((.5, 0), (0, 0, 1)), ((0, .5), (-1, 0, 0)),
                    ((.5, 1), (0, 0, -1)),
                    ((.75, .5), (math.sqrt(.5), math.sqrt(.5), 0))]
        for uv, expected in fixtures:
            for actual, wanted in zip(zenith_ray(*uv), expected):
                self.assertAlmostEqual(actual, wanted, places=12)
        self.assertIsNone(zenith_ray(0, 0))

    def test_pixel_center_ray_angle_and_unit_norm(self):
        # First/last centers of 24 tokens are +/-23/24, never +/-1.
        for i in range(24):
            ray = zenith_ray((i + .5) / 24, .5)
            self.assertAlmostEqual(sum(x*x for x in ray), 1, places=12)
            self.assertAlmostEqual(math.acos(ray[1]), abs(2*(i+.5)/24-1)*math.pi/2)


@unittest.skipIf(torch is None, 'PyTorch required for phase integration tests')
class ZenithPhaseTests(unittest.TestCase):
    def setUp(self):
        self.n = 6
        frame = torch.tensor([[0, i*32/6, j*32/6] for i in range(6) for j in range(6)], dtype=torch.float64)
        self.positions = torch.cat([torch.ones(3, 3), frame, torch.ones(2, 3), frame, frame])
        self.positions[-36:, 0] = 17
        self.segments = [(0,3,'text'), (3,39,'cond'), (39,41,'audio'), (41,113,'video')]
        self.inv = torch.logspace(0, -3, 16)
        self.stock = self.rope(self.positions, 'cpu')

    def rope(self, p, device):
        half = (p.float().to(device).unsqueeze(-1)*self.inv).reshape(-1,48)
        return torch.cat([half,half], dim=1)

    def apply(self, strength=1, include=True, positions=None):
        return hybrid_phases(self.stock, self.positions if positions is None else positions,
                             self.segments, self.inv, 12, 12, strength, 8, include)

    def test_zero_returns_identical_tensor(self):
        actual, _ = self.apply(0)
        self.assertIs(actual, self.stock)

    def test_only_requested_spatial_bands_and_rows_change(self):
        actual, report = self.apply()
        self.assertGreater(report['max_phase_delta_radians'], 0)
        self.assertEqual(report['low_frequency_indices'], list(range(8,16)))
        keep = list(range(0,24)) + list(range(32,40))
        keep += [x+48 for x in keep]
        self.assertTrue(torch.equal(actual[:,keep],self.stock[:,keep]))
        self.assertTrue(torch.equal(actual[:3],self.stock[:3]))
        self.assertTrue(torch.equal(actual[39:41],self.stock[39:41]))
        self.assertTrue(torch.equal(actual[:,:48],actual[:,48:]))
        for i in (3,41,77):  # top-left exterior
            self.assertTrue(torch.equal(actual[i],self.stock[i]))
        self.assertTrue(torch.equal(actual[41:77,16:48],actual[77:113,16:48]))
        self.assertTrue(torch.equal(actual[3:39,16:48],actual[41:77,16:48]))

    def test_keyframe_opt_out_and_unrelated_reference_preservation(self):
        actual, _ = self.apply(include=False)
        self.assertTrue(torch.equal(actual[3:39],self.stock[3:39]))
        segments = [(a,b,'ref_img' if k=='cond' else k) for a,b,k in self.segments]
        actual, _ = hybrid_phases(self.stock,self.positions,segments,self.inv,12,12,1,8)
        self.assertTrue(torch.equal(actual[3:39],self.stock[3:39]))

    def test_half_strength_is_half_phase_delta(self):
        full, _ = self.apply(1)
        half, _ = self.apply(.5)
        torch.testing.assert_close(half-self.stock,(full-self.stock)*.5,atol=1e-6,rtol=1e-5)

    def test_stacked_warp_is_rejected(self):
        positions = self.positions.clone()
        positions[50,1] += 1
        with self.assertRaisesRegex(ValueError,'nonstandard'):
            self.apply(positions=positions)

    def test_wrapper_restores_instance_on_success_and_error(self):
        test = self
        class Model:
            rope = SimpleNamespace(inv_freq=test.inv)
            def rope_freqs(self,p,device): return test.rope(p,device)
        model = Model()
        patch = H3ZenithRoPEPatch(.5,8)
        def executor(x,*args,**kwargs):
            output = model.rope_freqs(test.positions,'cpu')
            if kwargs.get('fail'): raise RuntimeError('injected error')
            return output
        executor.class_obj = model
        kw = dict(minimax_payload={'layout':SimpleNamespace(segments=self.segments)})
        x = [torch.empty(1,24,2,12,12)]
        result = patch(executor,x,None,None,{},**kw)
        self.assertFalse(torch.equal(result,self.stock))
        self.assertNotIn('rope_freqs',vars(model))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            patch(executor,x,None,None,{},fail=True,**kw)
        self.assertNotIn('rope_freqs',vars(model))
        self.assertTrue(torch.equal(model.rope_freqs(self.positions,'cpu'),self.stock))
        previous = model.rope_freqs
        model.rope_freqs = previous
        patch(executor,x,None,None,{},**kw)
        self.assertIs(vars(model)['rope_freqs'],previous)


if __name__ == '__main__':
    unittest.main()
