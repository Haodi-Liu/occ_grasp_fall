import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.backfill_bimanual_joint_target_fk_pose import (
    _validate_dry_run_manifest_output)


def _args(dataset_root, manifest):
    return SimpleNamespace(
        apply=False,
        dataset_root=dataset_root,
        manifest=manifest,
    )


class TestDryRunManifestOutputValidation(unittest.TestCase):

    def test_accepts_new_writable_path_outside_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / 'dataset'
            dataset_root.mkdir()
            manifest = root / 'audit' / 'dry_run.jsonl'
            args = _args(dataset_root, manifest)

            _validate_dry_run_manifest_output(args)

            self.assertEqual(args.manifest, manifest.absolute())

    def test_rejects_path_inside_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            args = _args(
                dataset_root, dataset_root / 'audit' / 'dry_run.jsonl')

            with self.assertRaisesRegex(ValueError, 'outside dataset root'):
                _validate_dry_run_manifest_output(args)

    def test_rejects_existing_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / 'dataset'
            dataset_root.mkdir()
            manifest = root / 'dry_run.jsonl'
            manifest.touch()
            args = _args(dataset_root, manifest)

            with self.assertRaisesRegex(ValueError, 'already exists'):
                _validate_dry_run_manifest_output(args)

    @patch(
        'tools.backfill_bimanual_joint_target_fk_pose.os.access',
        return_value=False)
    def test_rejects_unwritable_existing_parent(self, _):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / 'dataset'
            dataset_root.mkdir()
            args = _args(
                dataset_root, root / 'audit' / 'dry_run.jsonl')

            with self.assertRaisesRegex(ValueError, 'not writable'):
                _validate_dry_run_manifest_output(args)


if __name__ == '__main__':
    unittest.main()
