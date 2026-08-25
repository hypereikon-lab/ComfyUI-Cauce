import unittest

try:
    import numpy as np
except ImportError:
    np = None

from cauce.assembly import accept_decoded_range


@unittest.skipIf(np is None, "NumPy is supplied by ComfyUI, not CAUCE")
class AssemblyTests(unittest.TestCase):
    def test_accept_decoded_range_is_exact(self):
        images = np.arange(20, dtype=np.float32).reshape(20, 1, 1, 1)
        accepted, count = accept_decoded_range(images, 3, 7)
        self.assertEqual(count, 7)
        np.testing.assert_array_equal(accepted, images[3:10])

    def test_accept_decoded_range_rejects_overrun(self):
        images = np.zeros((5, 1, 1, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "range ends at 6"):
            accept_decoded_range(images, 2, 4)


if __name__ == "__main__":
    unittest.main()
