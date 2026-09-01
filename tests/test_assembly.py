import unittest

try:
    import numpy as np
except ImportError:
    np = None

from cauce.assembly import accept_decoded_range, restore_decoded_anchors


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

    def test_restore_decoded_anchors_preserves_only_generated_gaps(self):
        source = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
        generated = np.full((7, 1, 1, 1), 100.0, dtype=np.float32)

        restored, delivery, anchors, gaps = restore_decoded_anchors(
            generated,
            source,
            factor=2,
        )

        self.assertEqual((delivery, anchors, gaps), (7, 4, 3))
        np.testing.assert_array_equal(restored[::2], source)
        np.testing.assert_array_equal(restored[1::2], generated[1::2])
        np.testing.assert_array_equal(generated, np.full_like(generated, 100.0))

    def test_restore_decoded_anchors_crops_tail_padding(self):
        source = np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1)
        generated = np.full((8, 1, 1, 1), 9.0, dtype=np.float32)

        restored, delivery, anchors, gaps = restore_decoded_anchors(
            generated,
            source,
            factor=3,
        )

        self.assertEqual(restored.shape[0], 7)
        self.assertEqual((delivery, anchors, gaps), (7, 3, 4))
        np.testing.assert_array_equal(restored[::3], source)

    def test_restore_decoded_anchors_rejects_mismatched_geometry(self):
        source = np.zeros((3, 2, 2, 3), dtype=np.float32)
        generated = np.zeros((5, 3, 2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "matching geometry"):
            restore_decoded_anchors(generated, source, factor=2)


if __name__ == "__main__":
    unittest.main()
