#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.backend.bimanual_action_commands import ArmForwardKinematics
from rlbench.backend.bimanual_action_commands import SCHEMA_ID, SIDES
from rlbench.backend.bimanual_action_commands import joint_target_fk_pose_key
from rlbench.backend.bimanual_action_commands import (
    validate_bimanual_action_command_sources)
from rlbench.backend.bimanual_action_commands import (
    validate_bimanual_action_commands)
from rlbench.backend.const import LOW_DIM_PICKLE
from rlbench.environment import Environment


GIB = 1024 ** 3
DEFAULT_EXPECTED_EPISODES = 720
DEFAULT_EXPECTED_FRAMES = 265924


@dataclass(frozen=True)
class EpisodePlan:
    path: Path
    relative_path: str
    frame_count: int
    size_bytes: int
    sha256: str
    status: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Backfill FK poses derived from recorded bimanual joint targets. '
            'The default mode is dataset-read-only.'))
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--ttt', type=Path, required=True)
    parser.add_argument(
        '--expected-episodes', type=int,
        default=DEFAULT_EXPECTED_EPISODES)
    parser.add_argument(
        '--expected-frames', type=int, default=DEFAULT_EXPECTED_FRAMES)
    parser.add_argument('--fk-self-check-samples', type=int, default=100)
    parser.add_argument(
        '--first-frame-position-tol', type=float, default=0.005)
    parser.add_argument(
        '--first-frame-angle-tol', type=float, default=0.01)
    parser.add_argument('--existing-pose-atol', type=float, default=1e-6)
    parser.add_argument(
        '--min-free-reserve-gib', type=float, default=20.0,
        help='Free space that must remain after all planned rewrites.')
    parser.add_argument('--backup-root', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--restore', action='store_true')
    return parser.parse_args()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_pickle(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def _is_within(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _nearest_existing_parent(path):
    path = Path(path).expanduser().absolute()
    while not path.exists():
        if path.parent == path:
            raise ValueError('No existing parent for %s.' % path)
        path = path.parent
    return path


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(path, record, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'x' if exclusive else 'a'
    with path.open(mode, encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _read_jsonl(path):
    records = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    'Invalid JSON at %s:%d.' % (path, line_number)) from exc
    return records


def _pose_error(expected, observed):
    expected = np.asarray(expected, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if (expected.shape != (7,) or observed.shape != (7,)
            or not np.isfinite(expected).all()
            or not np.isfinite(observed).all()):
        raise ValueError('Expected two finite poses with shape (7,).')

    position_error = float(np.linalg.norm(expected[:3] - observed[:3]))
    expected_norm = float(np.linalg.norm(expected[3:]))
    observed_norm = float(np.linalg.norm(observed[3:]))
    if expected_norm <= 1e-12 or observed_norm <= 1e-12:
        raise ValueError('Pose contains a zero-norm quaternion.')
    expected_q = expected[3:] / expected_norm
    observed_q = observed[3:] / observed_norm
    cosine = float(np.clip(
        abs(np.dot(expected_q, observed_q)), 0.0, 1.0))
    angle_error = float(2.0 * np.arccos(cosine))
    return position_error, angle_error


def _sample_joint_target(arm, rng):
    cyclics, intervals = arm.get_joint_intervals()
    values = []
    for cyclic, interval in zip(cyclics, intervals):
        if cyclic:
            lower, upper = -math.pi, math.pi
        else:
            lower = float(interval[0])
            upper = lower + float(interval[1])
        margin = 0.02 * (upper - lower)
        values.append(rng.uniform(lower + margin, upper - margin))
    return np.asarray(values, dtype=np.float64)


def _self_check_fk(arm, fk, samples, rng, side):
    original_positions = arm.get_joint_positions()
    original_targets = arm.get_joint_target_positions()
    max_position_error = 0.0
    max_angle_error = 0.0
    try:
        for _ in range(samples):
            target = _sample_joint_target(arm, rng)
            arm.set_joint_positions(target)
            expected = fk.pose(target)
            observed = arm.get_tip().get_pose()
            position_error, angle_error = _pose_error(expected, observed)
            max_position_error = max(max_position_error, position_error)
            max_angle_error = max(max_angle_error, angle_error)
    finally:
        arm.set_joint_positions(original_positions)
        arm.set_joint_target_positions(original_targets)

    tolerance = 1e-5
    if (max_position_error > tolerance
            or max_angle_error > tolerance):
        raise ValueError(
            '%s FK self-check failed: position=%g, angle=%g.' %
            (side, max_position_error, max_angle_error))
    return {
        'samples': samples,
        'max_position_error': max_position_error,
        'max_angle_error': max_angle_error,
    }


def build_fk_models(ttt_path, self_check_samples):
    observation_config = ObservationConfig()
    action_mode = BimanualMoveArmThenGripper(
        BimanualJointPosition(), BimanualDiscrete())
    environment = Environment(
        action_mode=action_mode,
        obs_config=observation_config,
        robot_setup='dual_panda',
        headless=True,
        ttt_file=str(ttt_path))

    environment.launch()
    try:
        arms = {
            'right': environment._robot.right_arm,
            'left': environment._robot.left_arm,
        }
        fks = {
            side: ArmForwardKinematics(arms[side]) for side in SIDES
        }
        rng = np.random.RandomState(0)
        self_check = {
            side: _self_check_fk(
                arms[side], fks[side], self_check_samples, rng, side)
            for side in SIDES
        }
    finally:
        environment.shutdown()
    return fks, self_check


def _discover_episodes(dataset_root):
    paths = []
    for path in dataset_root.rglob(LOW_DIM_PICKLE):
        if path.is_symlink():
            raise ValueError('Refusing symlinked pickle: %s.' % path)
        resolved = path.resolve()
        if not _is_within(resolved, dataset_root):
            raise ValueError('Pickle escapes dataset root: %s.' % path)
        paths.append(resolved)
    return sorted(paths, key=lambda path: str(path.relative_to(dataset_root)))


def _classify_demo(demo, fks, pose_atol):
    presence = []
    for obs in demo:
        for side in SIDES:
            presence.append(joint_target_fk_pose_key(side) in obs.misc)

    if not any(presence):
        return 'pending'
    if not all(presence):
        raise ValueError('FK pose fields are only partially present.')

    validate_bimanual_action_commands(demo)
    for step_idx, obs in enumerate(demo):
        for side in SIDES:
            source_key = '%s_executed_demo_joint_position_action' % side
            pose_key = joint_target_fk_pose_key(side)
            expected = fks[side].pose(obs.misc[source_key])
            actual = np.asarray(obs.misc[pose_key], dtype=np.float32)
            if not np.allclose(
                    actual, expected, rtol=0.0, atol=pose_atol):
                raise ValueError(
                    '%s does not match its source at step %d.' %
                    (pose_key, step_idx))
    return 'already_correct'


def _check_first_frame(demo, fks, position_tol, angle_tol):
    first = demo[0]
    errors = {}
    for side in SIDES:
        arm_observation = getattr(first, side)
        expected = fks[side].pose(arm_observation.joint_positions)
        observed = arm_observation.gripper_pose
        position_error, angle_error = _pose_error(expected, observed)
        if position_error > position_tol or angle_error > angle_tol:
            raise ValueError(
                '%s first-frame model mismatch: position=%g, angle=%g.' %
                (side, position_error, angle_error))
        errors[side] = {
            'position_error': position_error,
            'angle_error': angle_error,
        }
    return errors


def preflight_dataset(args, fks):
    dataset_root = args.dataset_root.resolve()
    episode_paths = _discover_episodes(dataset_root)
    if len(episode_paths) != args.expected_episodes:
        raise ValueError(
            'Expected %d episodes, found %d.' %
            (args.expected_episodes, len(episode_paths)))

    plans = []
    total_frames = 0
    first_frame_max = {
        side: {'position_error': 0.0, 'angle_error': 0.0}
        for side in SIDES
    }
    for path in episode_paths:
        demo = _load_pickle(path)
        validate_bimanual_action_command_sources(demo)
        frame_errors = _check_first_frame(
            demo, fks, args.first_frame_position_tol,
            args.first_frame_angle_tol)
        status = _classify_demo(demo, fks, args.existing_pose_atol)
        frame_count = len(demo)
        total_frames += frame_count
        for side in SIDES:
            for key in ('position_error', 'angle_error'):
                first_frame_max[side][key] = max(
                    first_frame_max[side][key], frame_errors[side][key])
        plans.append(EpisodePlan(
            path=path,
            relative_path=str(path.relative_to(dataset_root)),
            frame_count=frame_count,
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            status=status,
        ))

    if total_frames != args.expected_frames:
        raise ValueError(
            'Expected %d frames, found %d.' %
            (args.expected_frames, total_frames))
    return plans, total_frames, first_frame_max


def _validate_output_locations(dataset_root, backup_root, manifest):
    dataset_root = dataset_root.resolve()
    if _is_within(backup_root, dataset_root):
        raise ValueError('Backup root must be outside the dataset root.')
    if _is_within(manifest, dataset_root):
        raise ValueError('Manifest must be outside the dataset root.')
    backup_parent = _nearest_existing_parent(backup_root)
    if os.stat(str(dataset_root)).st_dev != os.stat(str(backup_parent)).st_dev:
        raise ValueError(
            'Backup root must share a filesystem with the dataset root.')


def _check_free_space(dataset_root, pending_plans, reserve_gib):
    pending_bytes = sum(plan.size_bytes for plan in pending_plans)
    largest_pickle = max(
        [plan.size_bytes for plan in pending_plans] or [0])
    reserve_bytes = int(reserve_gib * GIB)
    required = int(math.ceil(pending_bytes * 1.10))
    required += largest_pickle + reserve_bytes
    available = shutil.disk_usage(str(dataset_root)).free
    if available < required:
        raise ValueError(
            'Insufficient free space: available=%d, required=%d.' %
            (available, required))
    return {
        'available_bytes': available,
        'rewrite_guard_bytes': required - reserve_bytes,
        'reserve_bytes': reserve_bytes,
        'required_bytes': required,
    }


def add_fk_fields(demo, fks):
    for obs in demo:
        for side in SIDES:
            source_key = '%s_executed_demo_joint_position_action' % side
            obs.misc[joint_target_fk_pose_key(side)] = (
                fks[side].pose(obs.misc[source_key]))


def _source_fingerprints(ttt_path):
    repository = Path(__file__).resolve().parents[1]
    source_paths = (
        repository / 'rlbench/backend/bimanual_action_commands.py',
        repository / 'rlbench/backend/scene.py',
        Path(__file__).resolve(),
    )
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=str(repository), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', 'status', '--porcelain'], cwd=str(repository), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None

    return {
        'git_commit': commit,
        'git_dirty': dirty,
        'ttt_path': str(ttt_path),
        'ttt_sha256': _sha256(ttt_path),
        'source_sha256': {
            str(path.relative_to(repository)): _sha256(path)
            for path in source_paths
        },
    }


def _manifest_header(args, fingerprints, self_check, mode):
    return {
        'type': 'header',
        'mode': mode,
        'created_at': _utc_now(),
        'schema_id': SCHEMA_ID,
        'dataset_root': str(args.dataset_root.resolve()),
        'expected_episodes': args.expected_episodes,
        'expected_frames': args.expected_frames,
        'first_frame_position_tol': args.first_frame_position_tol,
        'first_frame_angle_tol': args.first_frame_angle_tol,
        'existing_pose_atol': args.existing_pose_atol,
        'min_free_reserve_gib': args.min_free_reserve_gib,
        'fk_self_check': self_check,
        'fingerprints': fingerprints,
    }


def _print_summary(plans, total_frames, first_frame_max, space=None):
    pending = sum(plan.status == 'pending' for plan in plans)
    already_correct = sum(
        plan.status == 'already_correct' for plan in plans)
    summary = {
        'episodes': len(plans),
        'frames': total_frames,
        'pending': pending,
        'already_correct': already_correct,
        'first_frame_max': first_frame_max,
    }
    if space is not None:
        summary['space'] = space
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _load_apply_manifest(path):
    records = _read_jsonl(path)
    if not records or records[0].get('type') != 'header':
        raise ValueError('Manifest does not start with a header.')
    episode_records = {
        record['relative_path']: record
        for record in records[1:]
        if record.get('type') == 'episode'
    }
    return records[0], episode_records


def _prepare_apply_manifest(args, header):
    manifest = args.manifest.resolve()
    if manifest.exists():
        existing_header, records = _load_apply_manifest(manifest)
        required_header_values = {
            'mode': 'apply',
            'schema_id': SCHEMA_ID,
            'dataset_root': str(args.dataset_root.resolve()),
        }
        for key, expected in required_header_values.items():
            if existing_header.get(key) != expected:
                raise ValueError(
                    'Manifest %s mismatch: expected %r, got %r.' %
                    (key, expected, existing_header.get(key)))
        expected_ttt = header['fingerprints']['ttt_sha256']
        actual_ttt = existing_header['fingerprints']['ttt_sha256']
        if actual_ttt != expected_ttt:
            raise ValueError('Manifest TTT fingerprint mismatch.')
        return records

    _append_jsonl(manifest, header, exclusive=True)
    return {}


def _ensure_backup(plan, backup_root):
    backup_path = backup_root / plan.relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        if _sha256(backup_path) != plan.sha256:
            raise ValueError('Existing backup hash mismatch: %s.' % backup_path)
        return backup_path
    os.link(str(plan.path), str(backup_path))
    _fsync_directory(backup_path.parent)
    if _sha256(backup_path) != plan.sha256:
        raise ValueError('New backup hash mismatch: %s.' % backup_path)
    return backup_path


def _atomic_write_demo(plan, demo):
    descriptor, temp_name = tempfile.mkstemp(
        prefix='.%s.' % plan.path.name,
        suffix='.tmp', dir=str(plan.path.parent))
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with temp_path.open('wb') as stream:
            pickle.dump(demo, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        candidate = _load_pickle(temp_path)
        validate_bimanual_action_commands(candidate)
        shutil.copystat(str(plan.path), str(temp_path))
        if _sha256(plan.path) != plan.sha256:
            raise ValueError(
                'Source changed after preflight: %s.' % plan.path)
        os.replace(str(temp_path), str(plan.path))
        _fsync_directory(plan.path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def apply_backfill(args, plans, total_frames, first_frame_max,
                   fks, header):
    dataset_root = args.dataset_root.resolve()
    backup_root = args.backup_root.expanduser().absolute()
    manifest = args.manifest.expanduser().absolute()
    _validate_output_locations(dataset_root, backup_root, manifest)
    pending_plans = [plan for plan in plans if plan.status == 'pending']
    space = _check_free_space(
        dataset_root, pending_plans, args.min_free_reserve_gib)
    _print_summary(plans, total_frames, first_frame_max, space)

    existing_records = _prepare_apply_manifest(args, header)
    backup_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(backup_root.parent)

    remaining_bytes = sum(plan.size_bytes for plan in pending_plans)
    largest_pickle = max(
        [plan.size_bytes for plan in pending_plans] or [0])
    reserve_bytes = int(args.min_free_reserve_gib * GIB)

    for plan in plans:
        record = existing_records.get(plan.relative_path)
        backup_path = backup_root / plan.relative_path
        if plan.status == 'already_correct':
            if record is not None:
                if record.get('after_sha256') != plan.sha256:
                    raise ValueError(
                        'Manifest/current hash mismatch: %s.' % plan.path)
            elif backup_path.exists():
                before_sha = _sha256(backup_path)
                recovered = {
                    'type': 'episode',
                    'relative_path': plan.relative_path,
                    'frame_count': plan.frame_count,
                    'before_sha256': before_sha,
                    'after_sha256': plan.sha256,
                    'status': 'recovered_manifest',
                    'completed_at': _utc_now(),
                }
                _append_jsonl(manifest, recovered)
            continue

        available = shutil.disk_usage(str(dataset_root)).free
        remaining_guard = int(math.ceil(remaining_bytes * 1.10))
        required = remaining_guard + largest_pickle + reserve_bytes
        if available < required:
            raise ValueError(
                'Free space fell below the safe threshold before %s.' %
                plan.relative_path)
        if _sha256(plan.path) != plan.sha256:
            raise ValueError('Source changed after preflight: %s.' % plan.path)

        backup_path = _ensure_backup(plan, backup_root)
        demo = _load_pickle(plan.path)
        validate_bimanual_action_command_sources(demo)
        add_fk_fields(demo, fks)
        validate_bimanual_action_commands(demo)
        _atomic_write_demo(plan, demo)
        after_sha = _sha256(plan.path)
        record = {
            'type': 'episode',
            'relative_path': plan.relative_path,
            'frame_count': plan.frame_count,
            'before_sha256': _sha256(backup_path),
            'after_sha256': after_sha,
            'status': 'applied',
            'completed_at': _utc_now(),
        }
        _append_jsonl(manifest, record)
        remaining_bytes -= plan.size_bytes

    print('Backfill apply complete.')


def write_dry_run_manifest(args, header, summary):
    if args.manifest is None:
        return
    manifest = args.manifest.expanduser().absolute()
    if _is_within(manifest, args.dataset_root):
        raise ValueError('Dry-run manifest must be outside dataset root.')
    _append_jsonl(manifest, header, exclusive=True)
    _append_jsonl(manifest, {
        'type': 'summary',
        'created_at': _utc_now(),
        **summary,
    })


def restore_from_manifest(args):
    if args.manifest is None or args.backup_root is None:
        raise ValueError('--restore requires --manifest and --backup-root.')
    manifest = args.manifest.expanduser().absolute()
    header, episode_records = _load_apply_manifest(manifest)
    dataset_root = args.dataset_root.resolve()
    backup_root = args.backup_root.expanduser().absolute()
    if header.get('schema_id') != SCHEMA_ID:
        raise ValueError('Restore manifest schema mismatch.')
    if header.get('dataset_root') != str(dataset_root):
        raise ValueError('Restore manifest dataset root mismatch.')
    if len(episode_records) != args.expected_episodes:
        raise ValueError(
            'Expected %d applied records, found %d.' %
            (args.expected_episodes, len(episode_records)))

    restore_plans = []
    for relative_path, record in sorted(episode_records.items()):
        current = dataset_root / relative_path
        backup = backup_root / relative_path
        if current.is_symlink() or backup.is_symlink():
            raise ValueError('Refusing symlink during restore.')
        before_sha = _sha256(backup)
        current_sha = _sha256(current)
        if before_sha != record['before_sha256']:
            raise ValueError('Backup hash mismatch: %s.' % backup)
        if current_sha not in (
                record['before_sha256'], record['after_sha256']):
            raise ValueError('Current hash mismatch: %s.' % current)
        restore_plans.append((current, backup, record, current_sha))

    restore_manifest = manifest.with_name(
        '%s.restore.%s.jsonl' %
        (manifest.stem, datetime.now().strftime('%Y%m%dT%H%M%S')))
    _append_jsonl(restore_manifest, {
        'type': 'header',
        'mode': 'restore',
        'created_at': _utc_now(),
        'schema_id': SCHEMA_ID,
        'source_manifest': str(manifest),
        'dataset_root': str(dataset_root),
    }, exclusive=True)

    for current, backup, record, current_sha in restore_plans:
        if current_sha != record['before_sha256']:
            descriptor, temp_name = tempfile.mkstemp(
                prefix='.%s.restore.' % current.name,
                suffix='.tmp', dir=str(current.parent))
            os.close(descriptor)
            temp_path = Path(temp_name)
            temp_path.unlink()
            try:
                os.link(str(backup), str(temp_path))
                os.replace(str(temp_path), str(current))
                _fsync_directory(current.parent)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        _append_jsonl(restore_manifest, {
            'type': 'episode',
            'relative_path': record['relative_path'],
            'restored_sha256': _sha256(current),
            'completed_at': _utc_now(),
        })
    print('Restore complete: %s' % restore_manifest)


def _validate_dry_run_manifest_output(args):
    if args.apply or args.manifest is None:
        return
    manifest = args.manifest.expanduser().absolute()
    if _is_within(manifest, args.dataset_root):
        raise ValueError('Dry-run manifest must be outside dataset root.')
    if manifest.exists():
        raise ValueError(
            'Dry-run manifest already exists: %s.' % manifest)
    writable_parent = _nearest_existing_parent(manifest.parent)
    if not os.access(str(writable_parent), os.W_OK | os.X_OK):
        raise ValueError(
            'Dry-run manifest parent is not writable: %s.' %
            writable_parent)
    args.manifest = manifest


def _validate_args(args):
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.ttt = args.ttt.expanduser().resolve()
    if not args.dataset_root.is_dir():
        raise ValueError('Dataset root is not a directory.')
    if not args.ttt.is_file():
        raise ValueError('TTT is not a file.')
    if args.expected_episodes <= 0 or args.expected_frames <= 0:
        raise ValueError('Expected counts must be positive.')
    if args.fk_self_check_samples < 0:
        raise ValueError('FK self-check sample count cannot be negative.')
    if args.min_free_reserve_gib < 0.0:
        raise ValueError('Free-space reserve cannot be negative.')
    if args.restore and not args.apply:
        raise ValueError('--restore must be paired with --apply.')
    if args.apply and (args.backup_root is None or args.manifest is None):
        raise ValueError('--apply requires --backup-root and --manifest.')
    _validate_dry_run_manifest_output(args)


def main():
    args = parse_args()
    _validate_args(args)
    if args.restore:
        restore_from_manifest(args)
        return

    fks, self_check = build_fk_models(
        args.ttt, args.fk_self_check_samples)
    fingerprints = _source_fingerprints(args.ttt)
    plans, total_frames, first_frame_max = preflight_dataset(args, fks)
    mode = 'apply' if args.apply else 'dry_run'
    header = _manifest_header(args, fingerprints, self_check, mode)

    if args.apply:
        apply_backfill(
            args, plans, total_frames, first_frame_max, fks, header)
    else:
        pending_plans = [
            plan for plan in plans if plan.status == 'pending'
        ]
        space = _check_free_space(
            args.dataset_root, pending_plans, args.min_free_reserve_gib)
        summary = _print_summary(
            plans, total_frames, first_frame_max, space)
        write_dry_run_manifest(args, header, summary)


if __name__ == '__main__':
    main()
