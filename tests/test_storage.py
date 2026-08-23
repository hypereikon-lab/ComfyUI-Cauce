import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

from cauce.storage import (
    apply_storage_plan,
    build_storage_plan,
    resolve_scoped_directory,
    storage_plan_report,
    validate_storage_plan,
)
class StorageCoreTests(unittest.TestCase):
    def test_inventory_is_recursive_filtered_stable_and_preserves_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "clips").mkdir()
            (root / "clips" / "a.mp4").write_bytes(b"aaa")
            (root / "clips" / "b.png").write_bytes(b"bb")
            (root / "_output_images_will_be_put_here").write_text("")

            first = build_storage_plan(
                root,
                root_kind="output",
                include_glob="*.mp4,*.png",
            )
            second = build_storage_plan(
                root,
                root_kind="output",
                include_glob="*.mp4,*.png",
            )

            self.assertEqual(first["plan_id"], second["plan_id"])
            self.assertEqual(first["confirmation"], second["confirmation"])
            self.assertEqual(
                [entry["relative_path"] for entry in first["entries"]],
                ["clips/a.mp4", "clips/b.png"],
            )
            self.assertEqual(first["summary"]["total_bytes"], 5)
            self.assertIn(
                {
                    "relative_path": "_output_images_will_be_put_here",
                    "reason": "protected_marker",
                },
                first["skipped"],
            )
            self.assertEqual(storage_plan_report(first)["plan_id"], first["plan_id"])

    def test_scope_rejects_absolute_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            outside = Path(temp) / "outside"
            root.mkdir()
            outside.mkdir()
            with self.assertRaises(ValueError):
                resolve_scoped_directory(root, "../outside")
            with self.assertRaises(ValueError):
                resolve_scoped_directory(root, str(outside))

            link = root / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(ValueError):
                resolve_scoped_directory(root, "linked")

    def test_cleanup_requires_exact_confirmation_and_only_deletes_unchanged_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            planned = root / "planned.mp4"
            changed = root / "changed.mp4"
            planned.write_bytes(b"planned")
            changed.write_bytes(b"old")
            plan = build_storage_plan(root, root_kind="output")

            preview = apply_storage_plan(root, plan, armed=False, confirmation="")
            self.assertEqual(preview["status"], "not_armed")
            self.assertTrue(planned.exists())
            with self.assertRaises(ValueError):
                apply_storage_plan(root, plan, armed=True, confirmation="DELETE OUTPUT")

            changed.write_bytes(b"changed after plan")
            stat = changed.stat()
            os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            extra = root / "extra.mp4"
            extra.write_bytes(b"not in plan")

            receipt = apply_storage_plan(
                root,
                plan,
                armed=True,
                confirmation=plan["confirmation"],
            )
            self.assertEqual(receipt["status"], "partial")
            self.assertFalse(planned.exists())
            self.assertTrue(changed.exists())
            self.assertTrue(extra.exists())
            self.assertEqual(receipt["deleted_count"], 1)
            self.assertEqual(
                receipt["skipped"],
                [{"relative_path": "changed.mp4", "reason": "changed_since_plan"}],
            )

    def test_cleanup_removes_only_empty_parents_of_deleted_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target_dir = root / "nested" / "leaf"
            unrelated = root / "unrelated-empty"
            target_dir.mkdir(parents=True)
            unrelated.mkdir()
            (target_dir / "clip.mp4").write_bytes(b"video")
            plan = build_storage_plan(root, root_kind="output")
            receipt = apply_storage_plan(
                root,
                plan,
                armed=True,
                confirmation=plan["confirmation"],
                remove_empty_directories=True,
            )
            self.assertFalse((root / "nested").exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(
                set(receipt["removed_directories"]),
                {"nested/leaf", "nested"},
            )

    def test_tampered_plan_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "clip.mp4").write_bytes(b"video")
            plan = build_storage_plan(root, root_kind="output")
            plan["entries"][0]["relative_path"] = "different.mp4"
            with self.assertRaises(ValueError):
                validate_storage_plan(plan)
            with self.assertRaises(ValueError):
                apply_storage_plan(
                    root,
                    plan,
                    armed=True,
                    confirmation=plan["confirmation"],
                )

    def test_tampered_confirmation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "clip.mp4").write_bytes(b"video")
            plan = build_storage_plan(root, root_kind="output")
            plan["confirmation"] = "DELETE OUTPUT EASY"
            with self.assertRaises(ValueError):
                validate_storage_plan(plan)


class StorageNodeTests(unittest.TestCase):
    def test_nodes_inventory_delete_and_write_receipt_outside_storage_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            input_root = base / "input"
            output_root = base / "output"
            user_root = base / "user"
            input_root.mkdir()
            output_root.mkdir()
            user_root.mkdir()
            (input_root / "upload.mp4").write_bytes(b"upload")

            fake = types.ModuleType("folder_paths")
            fake.get_input_directory = lambda: str(input_root)
            fake.get_output_directory = lambda: str(output_root)
            fake.get_user_directory = lambda: str(user_root)
            previous = sys.modules.get("folder_paths")
            sys.modules["folder_paths"] = fake
            package_name = "comfyui_cauce_storage_test_plugin"
            repository = Path(__file__).resolve().parents[1]
            spec = importlib.util.spec_from_file_location(
                package_name,
                repository / "__init__.py",
                submodule_search_locations=[str(repository)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            try:
                assert spec.loader is not None
                spec.loader.exec_module(module)
                maintenance = sys.modules[f"{package_name}.cauce_nodes.maintenance"]
                inventory_result = maintenance.CauceStorageInventory().inventory(
                    "input", ".", "*", "", True, 0, True
                )
                plan, report, confirmation, file_count, total_gib = inventory_result[
                    "result"
                ]
                self.assertEqual(file_count, 1)
                self.assertGreater(total_gib, 0)
                self.assertEqual(json.loads(report)["confirmation"], confirmation)

                preview = maintenance.CauceStorageCleanup().cleanup(
                    plan, False, "", True
                )
                preview_receipt = json.loads(preview["result"][0])
                self.assertEqual(preview_receipt["status"], "staged")
                self.assertTrue(Path(preview_receipt["staged_plan_path"]).is_file())
                self.assertTrue((input_root / "upload.mp4").exists())

                (input_root / "arrived-after-review.mp4").write_bytes(b"new")
                changed_inventory = maintenance.CauceStorageInventory().inventory(
                    "input", ".", "*", "", True, 0, True
                )
                changed_plan, _, changed_confirmation, _, _ = changed_inventory[
                    "result"
                ]
                with self.assertRaises(ValueError):
                    maintenance.CauceStorageCleanup().cleanup(
                        changed_plan, True, changed_confirmation, True
                    )
                self.assertTrue((input_root / "upload.mp4").exists())
                self.assertTrue((input_root / "arrived-after-review.mp4").exists())

                applied = maintenance.CauceStorageCleanup().cleanup(
                    plan, True, confirmation, True
                )
                encoded, deleted_count, freed_gib, receipt_path = applied["result"]
                self.assertEqual(json.loads(encoded)["status"], "completed")
                self.assertEqual(deleted_count, 1)
                self.assertGreater(freed_gib, 0)
                self.assertFalse((input_root / "upload.mp4").exists())
                self.assertTrue((input_root / "arrived-after-review.mp4").exists())
                receipt = Path(receipt_path)
                self.assertTrue(receipt.is_file())
                self.assertTrue(receipt.is_relative_to(user_root.resolve()))
                self.assertFalse(receipt.is_relative_to(input_root.resolve()))
                self.assertFalse(receipt.is_relative_to(output_root.resolve()))
            finally:
                for key in list(sys.modules):
                    if key == package_name or key.startswith(package_name + "."):
                        sys.modules.pop(key, None)
                if previous is None:
                    sys.modules.pop("folder_paths", None)
                else:
                    sys.modules["folder_paths"] = previous


if __name__ == "__main__":
    unittest.main()
