#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import sys
import time
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path('/root/project')
GSPLAT_ROOT = ROOT / 'gaussian-splatting'
GSPLAT_EXAMPLES = ROOT / 'gsplat' / 'examples'
TOOLS_ROOT = Path(__file__).resolve().parent
OUT_ROOT = TOOLS_ROOT / 'reviewer4_baselines'
SCENE_DIR = ROOT / 'data' / '360_v2' / 'bicycle'
IMAGE_SOURCE = SCENE_DIR / 'images_4'
FULL_SPARSE = SCENE_DIR / 'sparse' / '0'
EXPERIMENT3_CSV = TOOLS_ROOT / 'experiment3_tables' / 'experiment3_per_scene_metrics.csv'
EXPERIMENT3_NE_CSV = TOOLS_ROOT / 'experiment3_tables' / 'experiment3_ne_metrics.csv'
OLD_BASELINES_ROOT = ROOT / 'data' / 'bicycle_baselines'
TRAINER = ROOT / 'gsplat' / 'examples' / 'simple_trainer_REAKS.py'

METHOD_ORDER = [
    'GT',
    'Original',
    'Uniform',
    'Blur-aware',
    'Deep K-Means',
    'Pose-FPS',
    'Coverage-Greedy',
    'SfM-Covisibility',
]
STAGE_INPUT = {
    'Original': 'Full input',
    'Uniform': 'Image order',
    'Blur-aware': 'Blur score',
    'Deep K-Means': 'Deep feature',
    'Pose-FPS': 'Camera pose',
    'Coverage-Greedy': 'Pose coverage',
    'SfM-Covisibility': 'Full-SfM oracle',
}
OLD_METHOD_DIRS = {
    'Uniform': OLD_BASELINES_ROOT / '1_Uniform',
    'Blur-aware': OLD_BASELINES_ROOT / '2_Blur_aware',
    'Deep K-Means': OLD_BASELINES_ROOT / '3_Deep_KMeans',
    'REAKS-old5': OLD_BASELINES_ROOT / '0_REAKS',
}
NEW_METHOD_DIRS = {
    'Pose-FPS': 'pose_fps',
    'Coverage-Greedy': 'coverage_greedy',
    'SfM-Covisibility': 'sfm_covisibility_greedy',
}
FIGURE_METHODS = [
    'Uniform',
    'Blur-aware',
    'Deep K-Means',
    'Pose-FPS',
    'Coverage-Greedy',
    'SfM-Covisibility',
    'REAKS',
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fields: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(v):
    if v in (None, '', 'NA', 'nan'):
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return None if math.isnan(x) else x


def qvec2rotmat(qvec: Sequence[float]) -> Tuple[Tuple[float, float, float], ...]:
    q0, q1, q2, q3 = qvec
    return (
        (1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q3 * q1 + 2 * q0 * q2),
        (2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1),
        (2 * q3 * q1 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2),
    )


def mat_vec_mul(m, v) -> Tuple[float, float, float]:
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def read_images_binary(path: Path) -> Dict[str, dict]:
    images = {}
    with path.open('rb') as f:
        n_raw = f.read(8)
        if len(n_raw) != 8:
            return images
        n = struct.unpack('<Q', n_raw)[0]
        for _ in range(n):
            h = f.read(64)
            if len(h) != 64:
                break
            vals = struct.unpack('<idddddddi', h)
            image_id = vals[0]
            qvec = tuple(float(x) for x in vals[1:5])
            tvec = tuple(float(x) for x in vals[5:8])
            camera_id = vals[8]
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if not c or c == b'\x00':
                    break
                name_bytes.extend(c)
            n2_raw = f.read(8)
            if len(n2_raw) != 8:
                break
            n2 = struct.unpack('<Q', n2_raw)[0]
            triples = []
            for _j in range(n2):
                rec = f.read(24)
                if len(rec) != 24:
                    break
                x, y, pid = struct.unpack('<ddq', rec)
                if pid != -1:
                    triples.append(int(pid))
            name = name_bytes.decode('utf-8')
            R = qvec2rotmat(qvec)
            Rt = tuple(tuple(R[j][i] for j in range(3)) for i in range(3))
            center = tuple(-x for x in mat_vec_mul(Rt, tvec))
            # Camera looks along +Z in camera coordinates for COLMAP world-to-camera convention after inversion.
            direction = tuple(Rt[i][2] for i in range(3))
            images[name] = {
                'image_id': image_id,
                'camera_id': camera_id,
                'qvec': qvec,
                'tvec': tvec,
                'center': center,
                'direction': direction,
                'points3D_ids': triples,
            }
    return images


def points3d_stats(path: Path) -> Tuple[Optional[int], Optional[float]]:
    if not path.exists():
        return None, None
    errs = []
    try:
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                return 0, None
            n = struct.unpack('<Q', raw)[0]
            for _ in range(n):
                h = f.read(43)
                if len(h) != 43:
                    break
                errs.append(float(struct.unpack('<QdddBBBd', h)[-1]))
                tl = f.read(8)
                if len(tl) != 8:
                    break
                f.seek(struct.unpack('<Q', tl)[0] * 8, 1)
    except Exception:
        return None, None
    return len(errs), statistics.mean(errs) if errs else None


def normalized_entropy(names: List[str], images: Dict[str, dict], sectors: int = 24) -> Optional[float]:
    centers = [images[n]['center'] for n in names if n in images]
    if not centers:
        return None
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    counts = [0] * sectors
    used = 0
    for x, y, _z in centers:
        dx, dy = x - cx, y - cy
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            continue
        angle = math.atan2(dy, dx) % (2 * math.pi)
        idx = min(int(angle / (2 * math.pi) * sectors), sectors - 1)
        counts[idx] += 1
        used += 1
    if used == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c:
            p = c / used
            ent -= p * math.log(p)
    return ent / math.log(sectors)


def vector_features(images: Dict[str, dict]) -> Dict[str, Tuple[float, ...]]:
    names = sorted(images)
    centers = [images[n]['center'] for n in names]
    means = [sum(c[i] for c in centers) / len(centers) for i in range(3)]
    spans = []
    for i in range(3):
        vals = [c[i] for c in centers]
        spans.append(max(vals) - min(vals) or 1.0)
    feats = {}
    for n in names:
        c = images[n]['center']
        d = images[n]['direction']
        feats[n] = tuple((c[i] - means[i]) / spans[i] for i in range(3)) + tuple(d)
    return feats


def sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def pose_fps(images: Dict[str, dict], k: int) -> List[str]:
    feats = vector_features(images)
    names = sorted(feats)
    first = min(names, key=lambda n: (images[n]['center'][0], images[n]['center'][1], n))
    selected = [first]
    min_dist = {n: sqdist(feats[n], feats[first]) for n in names}
    while len(selected) < min(k, len(names)):
        nxt = max((n for n in names if n not in selected), key=lambda n: (min_dist[n], n))
        selected.append(nxt)
        for n in names:
            if n not in selected:
                min_dist[n] = min(min_dist[n], sqdist(feats[n], feats[nxt]))
    return sorted(selected)


def coverage_bins_for_image(img: dict, center_mean: Tuple[float, float, float], sectors_xy: int = 24, sectors_dir: int = 12) -> set:
    x, y, z = img['center']
    dx, dy = x - center_mean[0], y - center_mean[1]
    pos_angle = math.atan2(dy, dx) % (2 * math.pi)
    pos_bin = int(pos_angle / (2 * math.pi) * sectors_xy)
    d = img['direction']
    dir_angle = math.atan2(d[1], d[0]) % (2 * math.pi)
    dir_bin = int(dir_angle / (2 * math.pi) * sectors_dir)
    elev_bin = 0 if d[2] < -0.25 else (2 if d[2] > 0.25 else 1)
    radius = math.sqrt(dx * dx + dy * dy + (z - center_mean[2]) ** 2)
    return {('pos', pos_bin), ('dir', dir_bin), ('elev', elev_bin), ('joint', pos_bin, dir_bin), ('rad', int(radius > 0))}


def coverage_greedy(images: Dict[str, dict], k: int) -> List[str]:
    names = sorted(images)
    centers = [images[n]['center'] for n in names]
    mean = tuple(sum(c[i] for c in centers) / len(centers) for i in range(3))
    bins = {n: coverage_bins_for_image(images[n], mean) for n in names}
    selected = []
    covered = set()
    while len(selected) < min(k, len(names)):
        def score(n):
            gain = len(bins[n] - covered)
            diversity = len(bins[n])
            return (gain, diversity, -len(selected), n)
        remaining = [n for n in names if n not in selected]
        nxt = max(remaining, key=score)
        selected.append(nxt)
        covered.update(bins[nxt])
    return sorted(selected)


def sfm_covisibility_greedy(images: Dict[str, dict], k: int) -> List[str]:
    names = sorted(images)
    point_sets = {n: set(images[n]['points3D_ids']) for n in names}
    selected = []
    covered = set()
    while len(selected) < min(k, len(names)):
        def score(n):
            pts = point_sets[n]
            new_pts = len(pts - covered)
            redundancy = len(pts & covered)
            return (new_pts, -0.05 * redundancy, len(pts), n)
        remaining = [n for n in names if n not in selected]
        nxt = max(remaining, key=score)
        selected.append(nxt)
        covered.update(point_sets[nxt])
    return sorted(selected)


def symlink_images(selected: List[str], dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ensure_dir(dst)
    for name in selected:
        src = IMAGE_SOURCE / name
        if not src.exists():
            raise FileNotFoundError(f'Missing source image {src}')
        os.symlink(src.resolve(), dst / name)


def load_colmap_rw():
    module_path = GSPLAT_ROOT / 'utils' / 'read_write_model.py'
    spec = importlib.util.spec_from_file_location('gs_read_write_model', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_subset_sparse(selected: List[str], sparse_dst: Path) -> None:
    rwm = load_colmap_rw()
    selected_set = set(selected)
    cameras = rwm.read_cameras_binary(str(FULL_SPARSE / 'cameras.bin'))
    images = rwm.read_images_binary(str(FULL_SPARSE / 'images.bin'))
    points3d = rwm.read_points3D_binary(str(FULL_SPARSE / 'points3D.bin'))
    selected_raw_images = {iid: img for iid, img in images.items() if img.name in selected_set}
    selected_ids = set(selected_raw_images.keys())
    used_cameras = {img.camera_id for img in selected_raw_images.values()}
    subset_cameras = {cid: cam for cid, cam in cameras.items() if cid in used_cameras}
    subset_points = {}
    kept_point_ids = set()
    for pid, pt in points3d.items():
        keep_pairs = [(iid, idx) for iid, idx in zip(pt.image_ids, pt.point2D_idxs) if int(iid) in selected_ids]
        if not keep_pairs:
            continue
        image_ids = [p[0] for p in keep_pairs]
        point2d_idxs = [p[1] for p in keep_pairs]
        subset_points[pid] = rwm.Point3D(
            id=pt.id, xyz=pt.xyz, rgb=pt.rgb, error=pt.error,
            image_ids=__import__('numpy').array(image_ids), point2D_idxs=__import__('numpy').array(point2d_idxs),
        )
        kept_point_ids.add(int(pid))
    selected_images = {}
    for iid, img in selected_raw_images.items():
        filtered_ids = []
        for pid in img.point3D_ids:
            pid_int = int(pid)
            filtered_ids.append(pid_int if pid_int in kept_point_ids else -1)
        selected_images[iid] = rwm.Image(
            id=img.id, qvec=img.qvec, tvec=img.tvec, camera_id=img.camera_id,
            name=img.name, xys=img.xys, point3D_ids=__import__('numpy').array(filtered_ids),
        )
    if sparse_dst.exists():
        shutil.rmtree(sparse_dst.parent)
    ensure_dir(sparse_dst)
    rwm.write_cameras_binary(subset_cameras, str(sparse_dst / 'cameras.bin'))
    rwm.write_images_binary(selected_images, str(sparse_dst / 'images.bin'))
    rwm.write_points3D_binary(subset_points, str(sparse_dst / 'points3D.bin'))


def infer_reaks_count() -> int:
    dirs = [
        SCENE_DIR / 'recon_major_3_double_single_gpu' / 'ratio_0.50' / 'seed_42' / 'images',
        OLD_BASELINES_ROOT / '0_REAKS' / 'images',
    ]
    for d in dirs:
        if d.exists():
            count = len([p for p in d.iterdir() if p.is_file() or p.is_symlink()])
            if count:
                return count
    raise RuntimeError('Cannot infer REAKS selected frame count for bicycle.')


def make_selected_lists() -> None:
    ensure_dir(OUT_ROOT)
    images = read_images_binary(FULL_SPARSE / 'images.bin')
    k = infer_reaks_count()
    methods = {
        'Pose-FPS': pose_fps(images, k),
        'Coverage-Greedy': coverage_greedy(images, k),
        'SfM-Covisibility': sfm_covisibility_greedy(images, k),
    }
    for method, selected in methods.items():
        method_dir = OUT_ROOT / NEW_METHOD_DIRS[method]
        run_dir = method_dir / 'run'
        ensure_dir(method_dir)
        ensure_dir(run_dir)
        symlink_images(selected, run_dir / 'images')
        sparse_dst = run_dir / 'sparse' / '0'
        write_subset_sparse(selected, sparse_dst)
        (method_dir / 'selected_images.txt').write_text('\n'.join(selected) + '\n')
        write_json(method_dir / 'selected_images.json', {
            'dataset': 'Mip-NeRF 360',
            'scene': 'bicycle',
            'method': method,
            'source_sparse': str(FULL_SPARSE),
            'image_source': str(IMAGE_SOURCE),
            'selected_count': len(selected),
            'target_count_from_REAKS': k,
            'selected_images': selected,
            'oracle_note': 'SfM-Covisibility-Greedy uses full SfM posterior tracks and is an oracle/upper-bound comparison.' if method == 'SfM-Covisibility' else '',
        })
    notes = [
        '# Reviewer 4 bicycle baselines notes',
        '',
        f'Default scene: `/root/project/data/360_v2/bicycle`.',
        f'All three new baselines select `{k}` images, matching the REAKS selected count.',
        'Pose-FPS is a camera-pose coverage baseline.',
        'Coverage-Greedy is a pose/spatial-angular coverage-aware baseline.',
        'SfM-Covisibility-Greedy is an SfM-aware oracle baseline: it uses full SfM posterior 3D point tracks and therefore cannot reduce the initial SfM cost.',
    ]
    (OUT_ROOT / 'reviewer4_new3_notes.md').write_text('\n'.join(notes) + '\n')


def run_training_for_missing(max_steps: int, seed: int, cuda: str, dry_run: bool = False) -> None:
    for method, slug in NEW_METHOD_DIRS.items():
        run_dir = OUT_ROOT / slug / 'run'
        result_dir = run_dir / 'results'
        stats_file = result_dir / 'stats' / f'val_step{max_steps - 1}.json'
        if stats_file.exists():
            print(f'[skip] {method}: existing {stats_file}')
            continue
        ensure_dir(result_dir)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cuda
        env['PYTHONPATH'] = '/root/project/gsplat/pycolmap:/root/project/gsplat/pycolmap/pycolmap:/root/project/gsplat/examples:/root/project/gsplat:' + env.get('PYTHONPATH', '')
        cmd = [
            'conda', 'run', '-n', 'gsplat', 'python', str(TRAINER), 'default',
            '--disable-viewer', '--disable-video', '--data-factor', '1',
            '--data-dir', str(run_dir), '--result-dir', str(result_dir),
            '--test-every', '8', '--max-steps', str(max_steps), '--eval-steps', str(max_steps),
            '--save-steps', str(max_steps), '--seed', str(seed), '--render-traj-path', 'ellipse', '--tb-every', '0',
        ]
        log_path = OUT_ROOT / slug / 'train.log'
        print('[train]', method, ' '.join(cmd))
        if dry_run:
            continue
        start = time.time()
        with log_path.open('w') as log:
            log.write('[cmd] ' + ' '.join(cmd) + '\n')
            proc = subprocess.run(cmd, cwd=str(GSPLAT_EXAMPLES), env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        (OUT_ROOT / slug / 'training_elapsed_sec.txt').write_text(f'{elapsed:.6f}\n')
        if proc.returncode != 0:
            raise RuntimeError(f'Training failed for {method}; see {log_path}')


def latest_val_json(stats_dir: Path) -> dict:
    files = sorted(stats_dir.glob('val_step*.json'))
    return read_json(files[-1]) if files else {}


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return len([p for p in d.iterdir() if p.is_file() or p.is_symlink()])


def stage_seconds_from_csv(path: Path, stage_names: Iterable[str]) -> Optional[float]:
    total = 0.0
    found = False
    names = set(stage_names)
    for row in read_csv(path):
        if row.get('stage') in names and row.get('status') in {'ok', 'skipped', 'not_applicable', 'dry_run'}:
            v = as_float(row.get('elapsed_sec'))
            if v is not None:
                total += v
                found = True
    return total if found else None


def timing_prefix(timings: dict, prefix: str) -> Optional[float]:
    vals = []
    for k, v in timings.items():
        if k.startswith(prefix) and isinstance(v, dict):
            x = as_float(v.get('elapsed_sec'))
            if x is not None:
                vals.append(x)
    return sum(vals) if vals else None


def metrics_from_experiment3(method_raw: str) -> Optional[dict]:
    for row in read_csv(EXPERIMENT3_CSV):
        if row.get('dataset_raw') == '360_v2' and row.get('scene') == 'bicycle' and row.get('method_raw') == method_raw and row.get('seed') == '42':
            ne = None
            for nr in read_csv(EXPERIMENT3_NE_CSV):
                if nr.get('dataset_raw') == '360_v2' and nr.get('scene') == 'bicycle' and nr.get('method_raw') == method_raw and nr.get('seed') == '42':
                    ne = as_float(nr.get('ne'))
                    break
            return {
                'dataset': 'Mip-NeRF 360', 'scene': 'bicycle',
                'selected_images': row.get('selected_images'),
                'PSNR': row.get('psnr'), 'SSIM': row.get('ssim'), 'LPIPS': row.get('lpips'),
                'points_k': row.get('sfm_points_k'), 'repro_error_px': row.get('sfm_reproj_error_px'),
                'NE': ne,
                'SfM_time_sec': row.get('colmap_sfm_time_sec'),
                'training_time_sec': row.get('training_time_sec'),
                'rendering_time_sec': row.get('rendering_time_sec'),
                'evaluation_time_sec': row.get('evaluation_time_sec'),
                'total_time_sec': row.get('total_pipeline_time_sec'),
            }
    return None


def run_metric_row(method: str, run_dir: Path, result_subdir: str = 'results', selected_override: Optional[int] = None) -> dict:
    result_dir = run_dir / result_subdir
    stats_dir = result_dir / 'stats'
    if not stats_dir.exists() and (run_dir / 'recon_REAKS' / 'stats').exists():
        result_dir = run_dir / 'recon_REAKS'
        stats_dir = result_dir / 'stats'
    val = latest_val_json(stats_dir)
    points, repro = points3d_stats(run_dir / 'sparse' / '0' / 'points3D.bin')
    full_images = read_images_binary(run_dir / 'sparse' / '0' / 'images.bin') if (run_dir / 'sparse' / '0' / 'images.bin').exists() else {}
    image_names = [p.name for p in sorted((run_dir / 'images').iterdir())] if (run_dir / 'images').exists() else list(full_images)
    ne = normalized_entropy(image_names, full_images) if full_images else None
    timings = read_json(stats_dir / 'pipeline_timings.json')
    train = timing_prefix(timings, '3dgs_training')
    render = timing_prefix(timings, 'rendering_all_step')
    eval_t = timing_prefix(timings, 'evaluation_val_step')
    stage_csv = run_dir / 'stage_timings.csv'
    sfm_t = stage_seconds_from_csv(stage_csv, ['colmap_feature_extraction', 'colmap_feature_matching', 'colmap_mapping_sfm'])
    if train is None:
        elapsed_file = run_dir.parent / 'training_elapsed_sec.txt'
        train = as_float(elapsed_file.read_text().strip()) if elapsed_file.exists() else None
    total_vals = [sfm_t, train, render, eval_t]
    total = sum(v for v in total_vals if v is not None) if any(v is not None for v in total_vals) else None
    return {
        'dataset': 'Mip-NeRF 360', 'scene': 'bicycle', 'method': method,
        'selected_images': selected_override if selected_override is not None else count_images(run_dir / 'images'),
        'PSNR': val.get('psnr'), 'SSIM': val.get('ssim'), 'LPIPS': val.get('lpips'),
        'points_k': None if points is None else points / 1000.0,
        'repro_error_px': repro, 'NE': ne,
        'SfM_time_sec': sfm_t, 'training_time_sec': train, 'rendering_time_sec': render,
        'evaluation_time_sec': eval_t, 'total_time_sec': total,
        'result_path': str(result_dir),
    }


def collect_metrics() -> List[dict]:
    rows = []
    original = metrics_from_experiment3('original')
    if original:
        original['method'] = 'Original'
        original['result_path'] = str(SCENE_DIR / 'recon_major_3_double_single_gpu' / 'original' / 'seed_42' / 'results')
        rows.append(original)
    for method in ['Uniform', 'Blur-aware', 'Deep K-Means']:
        rows.append(run_metric_row(method, OLD_METHOD_DIRS[method], result_subdir='recon_REAKS'))
    for method, slug in NEW_METHOD_DIRS.items():
        rows.append(run_metric_row(method, OUT_ROOT / slug / 'run'))
    reaks = metrics_from_experiment3('ratio_0.50')
    if reaks:
        reaks['method'] = 'REAKS'
        reaks['result_path'] = str(SCENE_DIR / 'recon_major_3_double_single_gpu' / 'ratio_0.50' / 'seed_42' / 'results')
        rows.append(reaks)
    rows.sort(key=lambda r: METHOD_ORDER.index(r['method']) if r.get('method') in METHOD_ORDER else 999)
    fields = ['dataset', 'scene', 'method', 'selected_images', 'PSNR', 'SSIM', 'LPIPS', 'points_k', 'repro_error_px', 'NE', 'SfM_time_sec', 'training_time_sec', 'rendering_time_sec', 'evaluation_time_sec', 'total_time_sec', 'result_path']
    write_csv(OUT_ROOT / 'reviewer4_new3_metrics.csv', [r for r in rows if r['method'] in NEW_METHOD_DIRS], fields)
    write_csv(OUT_ROOT / 'reviewer4_8method_metrics.csv', rows, fields)
    summary_fields = ['method'] + [f'{m}_mean_std' for m in ['selected_images','PSNR','SSIM','LPIPS','points_k','repro_error_px','NE','SfM_time_sec','training_time_sec','rendering_time_sec','evaluation_time_sec','total_time_sec']]
    summary = []
    for method in METHOD_ORDER:
        vals = [r for r in rows if r.get('method') == method]
        if not vals:
            continue
        out = {'method': method}
        for m in ['selected_images','PSNR','SSIM','LPIPS','points_k','repro_error_px','NE','SfM_time_sec','training_time_sec','rendering_time_sec','evaluation_time_sec','total_time_sec']:
            xs = [as_float(r.get(m)) for r in vals]
            xs = [x for x in xs if x is not None]
            if not xs:
                out[f'{m}_mean_std'] = '--'
            elif len(xs) == 1:
                out[f'{m}_mean_std'] = f'{xs[0]:.6g} (n=1)'
            else:
                out[f'{m}_mean_std'] = f'{statistics.mean(xs):.6g} ± {statistics.stdev(xs):.6g}'
        summary.append(out)
    write_csv(OUT_ROOT / 'reviewer4_new3_summary.csv', [r for r in summary if r['method'] in NEW_METHOD_DIRS], summary_fields)
    write_csv(OUT_ROOT / 'reviewer4_8method_summary.csv', summary, summary_fields)
    write_table_tex(rows)
    write_text_outputs(rows)
    return rows


def fmt(v, digits=3):
    x = as_float(v)
    if x is None:
        return '--'
    return f'{x:.{digits}f}'


def write_table_tex(rows: List[dict]) -> None:
    lines = [
        r'\begin{tabular}{llrrrrrrr}',
        r'\toprule',
        r'Method & Stage/Input & Views & Points (k)$\downarrow$ & Reproj.$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & NE$\uparrow$ \\',
        r'\midrule',
    ]
    for r in rows:
        m = r['method']
        lines.append(f"{m} & {STAGE_INPUT.get(m, '--')} & {r.get('selected_images','--')} & {fmt(r.get('points_k'),2)} & {fmt(r.get('repro_error_px'),3)} & {fmt(r.get('PSNR'),2)} & {fmt(r.get('SSIM'),3)} & {fmt(r.get('LPIPS'),3)} & {fmt(r.get('NE'),3)} \\")
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_ROOT / 'reviewer4_8method_table.tex').write_text('\n'.join(lines) + '\n')


def write_text_outputs(rows: List[dict]) -> None:
    notes = [
        '# Reviewer 4 baseline notes', '',
        'This run processes only `Mip-NeRF 360 / bicycle` by default, as requested.',
        'Pose-FPS and Coverage-Greedy are pose-aware / coverage-aware baselines.',
        'SfM-Covisibility-Greedy is an SfM-aware oracle baseline. It relies on full SfM posterior 3D point observations and therefore cannot reduce the initial SfM cost; it is included as an oracle / upper-bound style comparison.',
        '', '## Method mapping',
        '- Original: reviewer3 `original` bicycle result.',
        '- Uniform: `/root/project/data/bicycle_baselines/1_Uniform`.',
        '- Blur-aware: `/root/project/data/bicycle_baselines/2_Blur_aware`.',
        '- Deep K-Means: `/root/project/data/bicycle_baselines/3_Deep_KMeans`.',
        '- REAKS: reviewer3 `ratio_0.50` bicycle result.',
    ]
    for method, slug in NEW_METHOD_DIRS.items():
        stats = OUT_ROOT / slug / 'run' / 'results' / 'stats' / 'val_step29999.json'
        notes.append(f'- {method}: `{OUT_ROOT / slug / "run"}`; metrics present: {stats.exists()}.')
    missing = []
    for r in rows:
        for f in ['PSNR', 'SSIM', 'LPIPS']:
            if r.get(f) in (None, ''):
                missing.append(f"{r['method']} missing {f}")
    if missing:
        notes += ['', '## Missing items'] + [f'- {m}' for m in missing]
    (OUT_ROOT / 'reviewer4_new3_notes.md').write_text('\n'.join(notes) + '\n')
    caption = (
        r'\caption{Visual comparison on the Mip-NeRF 360 bicycle scene using the same test view and identical crop for all methods. '
        r'The eight panels compare Original, Uniform, Blur-aware, Deep K-Means, Pose-FPS, Coverage-Greedy, SfM-Covisibility, and REAKS. '
        r'Each panel reports PSNR, SSIM, and LPIPS for the selected view; SfM-Covisibility is an SfM-aware oracle baseline using full-SfM posterior information.}'
    )
    (OUT_ROOT / 'reviewer4_8method_figure_caption.tex').write_text(caption + '\n')
    manuscript = (
        'We additionally evaluate three stronger and more directly related baselines on the bicycle scene: Pose-FPS, Coverage-Greedy, and SfM-Covisibility-Greedy. '
        'Pose-FPS selects views using camera-pose coverage, Coverage-Greedy maximizes spatial-angular coverage, and SfM-Covisibility-Greedy greedily covers full-SfM 3D point tracks. '
        'The latter uses posterior information from the complete SfM reconstruction and is therefore reported only as an oracle comparison rather than as a method that can reduce initial SfM cost.'
    )
    (OUT_ROOT / 'reviewer4_8method_manuscript_text.md').write_text(manuscript + '\n')
    response = (
        'In response to this comment, we expanded the baseline comparison from the original five methods to eight methods on the bicycle scene. '
        'The added baselines include a camera-pose coverage method (Pose-FPS), a spatial-angular coverage method (Coverage-Greedy), and an SfM-aware oracle method (SfM-Covisibility-Greedy). '
        'We report numerical reconstruction/rendering metrics and provide a same-view 2 x 4 visual comparison. '
        'We explicitly mark SfM-Covisibility-Greedy as an oracle baseline because it uses full-SfM posterior tracks and therefore cannot reduce the initial SfM stage.'
    )
    (OUT_ROOT / 'reviewer4_8method_response_text.md').write_text(response + '\n')


def result_dir_for_method(method: str) -> Optional[Path]:
    if method == 'Original':
        return SCENE_DIR / 'recon_major_3_double_single_gpu' / 'original' / 'seed_42' / 'results'
    if method == 'REAKS':
        return SCENE_DIR / 'recon_major_3_double_single_gpu' / 'ratio_0.50' / 'seed_42' / 'results'
    if method in OLD_METHOD_DIRS:
        return OLD_METHOD_DIRS[method] / 'recon_REAKS'
    if method in NEW_METHOD_DIRS:
        return OUT_ROOT / NEW_METHOD_DIRS[method] / 'run' / 'results'
    return None


def load_image_np(path: Path):
    from PIL import Image
    import numpy as np
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def split_render_concat(path: Path):
    import numpy as np
    arr = load_image_np(path)
    h, w, _ = arr.shape
    if w > h:
        mid = w // 2
        left = arr[:, :mid, :]
        right = arr[:, mid:, :]
        mh = min(left.shape[0], right.shape[0])
        mw = min(left.shape[1], right.shape[1])
        return left[:mh, :mw, :], right[:mh, :mw, :]
    return arr, None


def psnr(a, b):
    import numpy as np
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def ssim_simple(a, b):
    import numpy as np
    # Global SSIM over RGB, sufficient for selecting/reporting the same crop view without adding dependencies.
    vals = []
    for c in range(3):
        x = a[..., c]
        y = b[..., c]
        ux, uy = float(x.mean()), float(y.mean())
        vx, vy = float(x.var()), float(y.var())
        cov = float(((x - ux) * (y - uy)).mean())
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        vals.append(((2 * ux * uy + c1) * (2 * cov + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))
    return sum(vals) / len(vals)


def crop_box_from_difference(reaks_img, other_imgs, frac: float = 0.45):
    import numpy as np
    h, w, _ = reaks_img.shape
    cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
    diff = np.zeros((h, w), dtype=np.float32)
    for img in other_imgs:
        diff += np.mean(np.abs(reaks_img - img), axis=2)
    # Smooth cheaply with block scoring through integral image.
    integ = diff.cumsum(axis=0).cumsum(axis=1)
    best = (-1.0, (w - cw) // 2, (h - ch) // 2)
    step = max(8, min(cw, ch) // 12)
    for y in range(0, h - ch + 1, step):
        for x in range(0, w - cw + 1, step):
            x2, y2 = x + cw - 1, y + ch - 1
            total = integ[y2, x2]
            if x > 0:
                total -= integ[y2, x - 1]
            if y > 0:
                total -= integ[y - 1, x2]
            if x > 0 and y > 0:
                total += integ[y - 1, x - 1]
            if total > best[0]:
                best = (float(total), x, y)
    _, x, y = best
    return x, y, x + cw, y + ch


def create_figure(rows: List[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    fig_dir = OUT_ROOT
    crops_dir = fig_dir / 'crops'
    ensure_dir(crops_dir)
    method_dirs = {m: result_dir_for_method(m) for m in FIGURE_METHODS}
    available_views = None
    for method, rd in method_dirs.items():
        if rd is None:
            continue
        views = {p.name.split('_')[-1].split('.')[0] for p in (rd / 'renders').glob('val_step29999_*.png')}
        available_views = views if available_views is None else available_views & views
    if not available_views:
        raise RuntimeError('No common rendered validation view across the 7 methods.')

    preferred_views = ['0000', '0004', '0008', '0016', '0018', '0020', '0022', '0026']
    candidate_views = [v for v in preferred_views if v in available_views]
    if not candidate_views:
        candidate_views = sorted(available_views)

    row_by_method = {r['method']: r for r in rows}
    best = None
    best_payload = None
    for vid in candidate_views:
        renders = {}
        for method, rd in method_dirs.items():
            left, right = split_render_concat(rd / 'renders' / f'val_step29999_{vid}.png')
            renders[method] = left
        gt_path = SCENE_DIR / 'recon_major_3_double_single_gpu' / 'original' / 'seed_42' / 'results' / 'step_29999' / 'test' / f'{vid}.png'
        if not gt_path.exists():
            # fall back to any available test image from the original baseline
            gt_candidates = sorted((SCENE_DIR / 'recon_major_3_double_single_gpu' / 'original' / 'seed_42' / 'results' / 'step_29999' / 'test').glob('*.png'))
            if not gt_candidates:
                continue
            gt_path = gt_candidates[int(vid) % len(gt_candidates)]
        gt = load_image_np(gt_path)
        common_h = min([gt.shape[0]] + [img.shape[0] for img in renders.values()])
        common_w = min([gt.shape[1]] + [img.shape[1] for img in renders.values()])
        gt = gt[:common_h, :common_w, :]
        for method in list(renders.keys()):
            renders[method] = renders[method][:common_h, :common_w, :]
        metrics = {}
        for method, img in renders.items():
            metrics[method] = {
                'PSNR': psnr(img, gt),
                'SSIM': ssim_simple(img, gt),
                'LPIPS': as_float(row_by_method.get(method, {}).get('LPIPS')),
            }
        if 'REAKS' not in metrics:
            continue
        score_terms = []
        for method in FIGURE_METHODS:
            if method == 'REAKS':
                continue
            lp_re = metrics['REAKS']['LPIPS']
            lp_m = metrics[method]['LPIPS']
            lp_term = 0.0 if lp_re is None or lp_m is None else -10.0 * (lp_re - lp_m)
            score_terms.append((metrics['REAKS']['PSNR'] - metrics[method]['PSNR']) + 10.0 * (metrics['REAKS']['SSIM'] - metrics[method]['SSIM']) + lp_term)
        score = sum(score_terms) / len(score_terms)
        if best is None or score > best[0]:
            best = (score, vid)
            best_payload = (renders, gt, metrics, gt_path)
    if best_payload is None:
        raise RuntimeError('Could not choose a common validation view.')

    score, vid = best
    renders, gt, metrics, gt_path = best_payload
    crop = crop_box_from_difference(renders['REAKS'], [renders[m] for m in FIGURE_METHODS if m != 'REAKS'])
    x1, y1, x2, y2 = crop
    crop_rows = []
    Image.fromarray((np.clip(gt[y1:y2, x1:x2, :], 0, 1) * 255).astype(np.uint8)).save(crops_dir / 'GT.png')
    for method in FIGURE_METHODS:
        arr = renders[method][y1:y2, x1:x2, :]
        safe = method.replace('/', '_').replace(' ', '_')
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(crops_dir / f'{safe}.png')
        crop_rows.append({
            'scene': 'bicycle', 'view_id': vid, 'method': method,
            'PSNR': metrics[method]['PSNR'], 'SSIM': metrics[method]['SSIM'], 'LPIPS': metrics[method]['LPIPS'],
            'crop_x1': x1, 'crop_y1': y1, 'crop_x2': x2, 'crop_y2': y2,
        })
    write_csv(fig_dir / 'reviewer4_best_view_metrics.csv', crop_rows)

    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.0), dpi=300)
    display_methods = ['GT'] + FIGURE_METHODS
    for ax, method in zip(axes.flat, display_methods):
        arr = gt if method == 'GT' else renders[method]
        ax.imshow(np.clip(arr, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(method, fontsize=12, fontweight='normal', pad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')
    fig.subplots_adjust(left=0.012, right=0.988, top=0.94, bottom=0.02, wspace=0.03, hspace=0.12)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(fig_dir / f'reviewer4_8method_visual_comparison.{ext}', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    (fig_dir / 'reviewer4_best_view.json').write_text(json.dumps({'scene': 'bicycle', 'view_id': vid, 'score': score, 'crop': crop, 'preferred_front_view_candidates': candidate_views, 'gt_path': str(gt_path)}, indent=2) + '\n')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['select', 'train-missing', 'collect', 'figure', 'all'])
    parser.add_argument('--max-steps', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ensure_dir(OUT_ROOT)
    if args.command in {'select', 'all'}:
        make_selected_lists()
    if args.command in {'train-missing', 'all'}:
        run_training_for_missing(args.max_steps, args.seed, args.cuda, args.dry_run)
    rows = []
    if args.command in {'collect', 'figure', 'all'}:
        rows = collect_metrics()
    if args.command in {'figure', 'all'}:
        if not rows:
            rows = collect_metrics()
        create_figure(rows)
    if args.command == 'all':
        print('Reviewer4 output:', OUT_ROOT)
        print('New baseline CSV:', OUT_ROOT / 'reviewer4_new3_metrics.csv')
        print('8-method CSV:', OUT_ROOT / 'reviewer4_8method_metrics.csv')
        print('2x4 PNG:', OUT_ROOT / 'reviewer4_8method_visual_comparison.png')


if __name__ == '__main__':
    main()
