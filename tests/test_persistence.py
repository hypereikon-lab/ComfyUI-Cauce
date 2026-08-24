from pathlib import Path
import tempfile
import unittest

from cauce.persistence import resolve_latest_or_indexed, safe_output_path


class PersistenceTests(unittest.TestCase):
    def test_output_path_is_relative_and_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                safe_output_path(root, "cauce/latents/clip"),
                (root / "cauce" / "latents" / "clip").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "relative"):
                safe_output_path(root, str(root / "absolute"))
            with self.assertRaisesRegex(ValueError, "escapes"):
                safe_output_path(root, "../outside")

    def test_resolve_latest_or_indexed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "latents"
            folder.mkdir()
            first = folder / "clip_00001.safetensors"
            second = folder / "clip_00002.safetensors"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            self.assertEqual(
                resolve_latest_or_indexed(root, "latents", artifact_index=1),
                first.resolve(),
            )
            self.assertIn(
                resolve_latest_or_indexed(root, "latents"),
                {first.resolve(), second.resolve()},
            )


if __name__ == "__main__":
    unittest.main()
