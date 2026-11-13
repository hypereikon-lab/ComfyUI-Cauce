import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from cauce.geometry_persistence import save_geometry_archive


class GeometryPersistenceTests(unittest.TestCase):
    def geometry(self):
        points = np.zeros((2, 3, 4, 3), np.float32)
        points[0, ..., 0] = 3
        points[0, ..., 2] = 4
        points[1, ..., 2] = 20
        return {"points": points, "mask": np.ones((2, 3, 4), bool),
                "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)}

    def test_preserves_metric_scale_order_and_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            path, receipt = save_geometry_archive(self.geometry(), root, "nested/view", sample_index_start=12)
            self.assertEqual(receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                for i, expected in enumerate((5, 20)):
                    entry = manifest["samples"][i]
                    payload = archive.read(entry["filename"])
                    self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
                    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["radial"], np.full((3, 4), expected))
                        self.assertEqual(int(saved["sample_index"]), i + 12)
                        self.assertTrue(saved["mask"].all())
            with self.assertRaises(FileExistsError):
                save_geometry_archive(self.geometry(), root, "nested/view")

    def test_invalid_shape_and_escape_leave_no_archive(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                save_geometry_archive(self.geometry(), root, "../escape")
            geometry = self.geometry()
            geometry["mask"] = geometry["mask"][:1]
            with self.assertRaises(ValueError):
                save_geometry_archive(geometry, root, "broken")
            self.assertEqual(list(Path(root).iterdir()), [])
