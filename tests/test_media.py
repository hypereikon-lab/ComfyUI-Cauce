import unittest

from cauce.media import select_image_frame


class FakeImages:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values), 1, 1, 3)

    def __getitem__(self, item):
        return self.values[item]


class MediaTests(unittest.TestCase):
    def test_select_last_image_preserves_batch_slice(self):
        result, resolved = select_image_frame(FakeImages(["a", "b", "c"]), -1)
        self.assertEqual(result, ["c"])
        self.assertEqual(resolved, 2)

    def test_select_image_rejects_out_of_range_index(self):
        with self.assertRaisesRegex(ValueError, "outside an image batch"):
            select_image_frame(FakeImages(["a", "b"]), 2)

    def test_select_image_rejects_empty_batch(self):
        with self.assertRaisesRegex(ValueError, "empty image batch"):
            select_image_frame(FakeImages([]), -1)


if __name__ == "__main__":
    unittest.main()
