#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path[:0] = [
    '/root/project/gsplat/pycolmap',
    '/root/project/gsplat/pycolmap/pycolmap',
    '/root/project/gsplat/examples',
    '/root/project/gsplat',
]

from datasets.colmap import Parser, Dataset  # noqa: E402
from datasets.normalize import transform_cameras  # noqa: E402
import simple_trainer_REAKS as trainer  # noqa: E402

ROOT = Path('/root/project')
TOOLS_ROOT = Path(__file__).resolve().parent
OUT = TOOLS_ROOT / 'reviewer4_baselines'
SCENE = ROOT / 'data' / '360_v2' / 'bicycle'
CAMERA_NAME = '_DSC8679.JPG'
STEP = 29999

# Figure controls. Increase FIG_LABEL_FONTSIZE if method names look too small
# after \includegraphics[width=0.98\textwidth].
FIG_LABEL_FONTSIZE = 17
FIG_LABEL_Y = -0.055
FIG_WIDTH_IN = 13.8
FIG_HEIGHT_IN = 6.5
FIG_DPI = 300
FIG_WSPACE = 0.025
FIG_HSPACE = 0.24
FIG_BOTTOM = 0.055
FIG_TOP = 0.985

METHODS = {
    'Uniform': {
        'data_dir': ROOT / 'data' / 'bicycle_baselines' / '1_Uniform',
        'result_dir': ROOT / 'data' / 'bicycle_baselines' / '1_Uniform' / 'recon_REAKS',
    },
    'Blur-aware': {
        'data_dir': ROOT / 'data' / 'bicycle_baselines' / '2_Blur_aware',
        'result_dir': ROOT / 'data' / 'bicycle_baselines' / '2_Blur_aware' / 'recon_REAKS',
    },
    'Deep K-Means': {
        'data_dir': ROOT / 'data' / 'bicycle_baselines' / '3_Deep_KMeans',
        'result_dir': ROOT / 'data' / 'bicycle_baselines' / '3_Deep_KMeans' / 'recon_REAKS',
    },
    'Pose-FPS': {
        'data_dir': OUT / 'pose_fps' / 'run',
        'result_dir': OUT / 'pose_fps' / 'run' / 'results',
    },
    'Coverage-Greedy': {
        'data_dir': OUT / 'coverage_greedy' / 'run',
        'result_dir': OUT / 'coverage_greedy' / 'run' / 'results',
    },
    'SfM-Covisibility': {
        'data_dir': OUT / 'sfm_covisibility_greedy' / 'run',
        'result_dir': OUT / 'sfm_covisibility_greedy' / 'run' / 'results',
    },
    'REAKS': {
        'data_dir': ROOT / 'data' / 'bicycle_baselines' / '0_REAKS',
        'result_dir': ROOT / 'data' / 'bicycle_baselines' / '0_REAKS' / 'recon_REAKS',
    },
}


def safe_name(name: str) -> str:
    return name.replace('/', '_').replace(' ', '_')


def make_cfg(data_dir: Path, result_dir: Path, ckpt: Path) -> trainer.Config:
    cfg = trainer.Config()
    cfg.disable_viewer = True
    cfg.disable_video = True
    cfg.data_dir = str(data_dir)
    cfg.data_factor = 1
    cfg.result_dir = str(result_dir / '_same_camera_tmp')
    cfg.test_every = 8
    cfg.max_steps = 30000
    cfg.eval_steps = [30000]
    cfg.save_steps = [30000]
    cfg.ckpt = [str(ckpt)]
    cfg.render_traj_path = 'ellipse'
    return cfg


def load_runner(data_dir: Path, result_dir: Path):
    ckpt = result_dir / 'ckpts' / f'ckpt_{STEP}_rank0.pt'
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    cfg = make_cfg(data_dir, result_dir, ckpt)
    runner = trainer.Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    payload = torch.load(str(ckpt), map_location=runner.device, weights_only=True)
    for k in runner.splats.keys():
        runner.splats[k].data = payload['splats'][k].to(runner.device)
    return runner


def original_name_from_method_name(name: str, full_names: list[str]) -> str | None:
    stem = Path(name).stem
    if stem.isdigit():
        idx = int(stem)
        return full_names[idx] if 0 <= idx < len(full_names) else None
    return name if name in full_names else None


def estimate_sim3(full_parser: Parser, method_parser: Parser) -> tuple[float, np.ndarray, np.ndarray, int]:
    full_names = list(full_parser.image_names)
    full_c2w = {n: full_parser.camtoworlds[i] for i, n in enumerate(full_names)}
    src = []
    dst = []
    for i, name in enumerate(method_parser.image_names):
        orig = original_name_from_method_name(name, full_names)
        if orig is None or orig not in full_c2w:
            continue
        src.append(full_c2w[orig][:3, 3])
        dst.append(method_parser.camtoworlds[i][:3, 3])
    if len(src) < 3:
        raise RuntimeError(f'Not enough common cameras for Sim3 alignment: {len(src)}')
    X = np.asarray(src, dtype=np.float64)
    Y = np.asarray(dst, dtype=np.float64)
    mux = X.mean(axis=0)
    muy = Y.mean(axis=0)
    Xc = X - mux
    Yc = Y - muy
    H = (Xc.T @ Yc) / len(X)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    varx = np.mean(np.sum(Xc * Xc, axis=1))
    scale = float(np.sum(S) / max(varx, 1e-12))
    t = muy - scale * (R @ mux)
    return scale, R, t, len(src)


def apply_sim3_to_c2w(c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = c2w.copy()
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = scale * (R @ c2w[:3, 3]) + t
    return out


def render_one(method: str, data_dir: Path, result_dir: Path, full_parser: Parser, raw_c2w: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
    runner = load_runner(data_dir, result_dir)
    scale, R, t, n_common = estimate_sim3(full_parser, runner.parser)
    method_c2w = apply_sim3_to_c2w(raw_c2w, scale, R, t)
    c2w_t = torch.from_numpy(method_c2w[None]).float().to(runner.device)
    K_t = torch.from_numpy(K[None]).float().to(runner.device)
    with torch.no_grad():
        colors, _, _ = runner.rasterize_splats(
            camtoworlds=c2w_t,
            Ks=K_t,
            width=width,
            height=height,
            sh_degree=runner.cfg.sh_degree,
            near_plane=runner.cfg.near_plane,
            far_plane=runner.cfg.far_plane,
            render_mode='RGB',
        )
    arr = colors[0].clamp(0, 1).detach().cpu().numpy()
    del runner
    torch.cuda.empty_cache()
    meta = {'alignment_common_cameras': n_common, 'sim3_scale': scale}
    return arr, meta


def crop_box(gt: np.ndarray, renders: dict[str, np.ndarray]) -> tuple[int, int, int, int]:
    # Use a fixed paper-friendly crop around the full bicycle in _DSC8679.JPG.
    h, w = gt.shape[:2]
    x1 = int(w * 0.10)
    x2 = int(w * 0.72)
    y1 = int(h * 0.08)
    y2 = int(h * 0.78)
    return x1, y1, x2, y2


def make_figure(gt: np.ndarray, renders: dict[str, np.ndarray], crop: tuple[int, int, int, int]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    same_dir = OUT / 'same_camera'
    crops_dir = same_dir / 'crops'
    same_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = crop
    display = ['GT', 'Uniform', 'Blur-aware', 'Deep K-Means', 'Pose-FPS', 'Coverage-Greedy', 'SfM-Covisibility', 'REAKS']
    arrays = {'GT': gt, **renders}
    for name, arr in arrays.items():
        Image.fromarray((np.clip(arr[y1:y2, x1:x2], 0, 1) * 255).astype(np.uint8)).save(crops_dir / f'{safe_name(name)}.png')

    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI)
    for ax, name in zip(axes.flat, display):
        ax.imshow(np.clip(arrays[name][y1:y2, x1:x2], 0, 1))
        ax.text(
            0.5,
            FIG_LABEL_Y,
            name,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=FIG_LABEL_FONTSIZE,
            fontweight='normal',
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_edgecolor('black')
    fig.subplots_adjust(left=0.012, right=0.988, top=FIG_TOP, bottom=FIG_BOTTOM, wspace=FIG_WSPACE, hspace=FIG_HSPACE)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(same_dir / f'reviewer4_same_camera_2x4.{ext}', bbox_inches='tight', pad_inches=0.03)
    # Also overwrite the earlier conventional names so the latest requested figure is easy to find.
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'reviewer4_8method_visual_comparison.{ext}', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    same_dir = OUT / 'same_camera'
    same_dir.mkdir(parents=True, exist_ok=True)
    full_parser = Parser(str(SCENE), factor=4, normalize=False, test_every=8)
    if CAMERA_NAME not in full_parser.image_names:
        raise RuntimeError(f'{CAMERA_NAME} not found')
    full_idx = full_parser.image_names.index(CAMERA_NAME)
    val_dataset = Dataset(full_parser, split='val')
    val_positions = {int(idx): i for i, idx in enumerate(val_dataset.indices)}
    if full_idx not in val_positions:
        raise RuntimeError(f'{CAMERA_NAME} is not a test_every=8 validation camera')
    data = val_dataset[val_positions[full_idx]]
    gt = (data['image'].numpy() / 255.0).astype(np.float32)
    raw_c2w = data['camtoworld'].numpy()
    K = data['K'].numpy()
    height, width = gt.shape[:2]
    imageio.imwrite(same_dir / 'GT_full.png', (gt * 255).astype(np.uint8))

    renders = {}
    render_meta = {}
    for method, spec in METHODS.items():
        print(f'[render] {method}')
        arr, rmeta = render_one(method, Path(spec['data_dir']), Path(spec['result_dir']), full_parser, raw_c2w, K, width, height)
        renders[method] = arr
        render_meta[method] = rmeta
        imageio.imwrite(same_dir / f'{safe_name(method)}_full.png', (np.clip(arr, 0, 1) * 255).astype(np.uint8))
    crop = crop_box(gt, renders)
    make_figure(gt, renders, crop)
    meta = {
        'scene': 'bicycle',
        'camera_name': CAMERA_NAME,
        'full_sorted_index': full_idx,
        'validation_position': val_positions[full_idx],
        'crop': crop,
        'methods': list(METHODS.keys()),
        'alignment': render_meta,
        'note': 'All method panels are rendered from the same raw COLMAP camera pose, transformed into each method normalized coordinate system.',
    }
    (same_dir / 'same_camera_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    (OUT / 'reviewer4_best_view.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
