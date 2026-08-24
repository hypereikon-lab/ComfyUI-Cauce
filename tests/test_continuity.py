import unittest

try:
    import numpy
except ImportError:
    numpy = None

from cauce.continuity import accept_decoded_range


class ContinuityTests(unittest.TestCase):
    @unittest.skipIf(numpy is None, "NumPy is supplied by ComfyUI, not CAUCE")
    def test_accept_decoded_range_is_exact(self):
        images = numpy.arange(20, dtype=numpy.float32).reshape(20, 1, 1, 1)
        accepted, count = accept_decoded_range(images, 3, 7)
        self.assertEqual(count, 7)
        numpy.testing.assert_array_equal(accepted, images[3:10])

    @unittest.skipIf(numpy is None, "NumPy is supplied by ComfyUI, not CAUCE")
    def test_accept_decoded_range_rejects_overrun(self):
        images = numpy.zeros((5, 1, 1, 1), dtype=numpy.float32)
        with self.assertRaisesRegex(ValueError, "range ends at 6"):
            accept_decoded_range(images, 2, 4)


if __name__ == "__main__":
    unittest.main()
