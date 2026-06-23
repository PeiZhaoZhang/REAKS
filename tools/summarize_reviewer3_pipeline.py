#!/usr/bin/env python3
"""Summarize Reviewer #3 REAKS pipeline outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def read_stage_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def latest_val_stats(stats_dir: Path) -> Dict:
    vals = sorted(stats_dir.glob("val_step*.json"))
    if not vals:
        return {}
    return read_json(vals[-1])


def seconds_from_stage(rows: List[Dict[str, str]], stage: str) -> float:
    total = 0.0
    for row in rows:
        if row.get("stage") == stage and row.get("status") in {"ok", "skipped", "not_applicable", "dry_run"}:
            try:
                total += float(row.get("elapsed_sec") or 0.0)
            except ValueError:
                pass
    return total


def timing_entry(timings: Dict, prefix: str) -> float:
    for key, value in timings.items():
        if key.startswith(prefix) and isinstance(value, dict):
            try:
                return float(value.get("elapsed_sec", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def reaks_stage_seconds(reaks_dir: Path, stage: str) -> float:
    rows = read_stage_csv(reaks_dir / "reaks_stage_timings.csv")
    total = 0.0
    for row in rows:
        if row.get("stage") == stage:
            try:
                total += float(row.get("elapsed_sec") or 0.0)
            except ValueError:
                pass
    return total



def read_colmap_points3d_bin(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"sfm_points": "", "sfm_points_k": "", "sfm_reproj_error_px": ""}
    errors = []
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            return {"sfm_points": 0, "sfm_points_k": 0.0, "sfm_reproj_error_px": ""}
        num_points = struct.unpack("<Q", raw)[0]
        for _ in range(num_points):
            header = f.read(43)
            if len(header) != 43:
                break
            # POINT3D_ID, XYZ, RGB, ERROR
            unpacked = struct.unpack("<QdddBBBd", header)
            errors.append(float(unpacked[-1]))
            track_len_raw = f.read(8)
            if len(track_len_raw) != 8:
                break
            track_len = struct.unpack("<Q", track_len_raw)[0]
            f.seek(track_len * 8, 1)
    mean_error = statistics.mean(errors) if errors else ""
    return {
        "sfm_points": len(errors),
        "sfm_points_k": len(errors) / 1000.0,
        "sfm_reproj_error_px": mean_error,
    }

def collect_runs(roots: Iterable[Path], tag: str) -> List[Path]:
    runs = []
    for root in roots:
        if not root.exists():
            continue
        runs.extend(sorted(root.glob(f"*/{tag}/*/seed_*")))
    return runs


def safe_div(a, b):
    try:
        a = float(a)
        b = float(b)
        return a / b if b else ""
    except (TypeError, ValueError):
        return ""


def run_to_row(run_dir: Path, tag: str) -> Dict[str, object]:
    method = run_dir.parent.name
    seed = run_dir.name.replace("seed_", "")
    scene_dir = run_dir.parents[2]
    scene = scene_dir.name
    dataset = scene_dir.parent.name
    ratio = method.replace("ratio_", "") if method.startswith("ratio_") else "0.50" if method == "original" else "1.00"

    stage_rows = read_stage_csv(run_dir / "stage_timings.csv")
    timings = read_json(run_dir / "results" / "stats" / "pipeline_timings.json")
    metrics = latest_val_stats(run_dir / "results" / "stats")
    metadata = read_json(run_dir / "reaks" / "reaks_timing_metadata.json")
    if not metadata:
        metadata = read_json(run_dir / "selection_metadata.json")
    sfm = read_colmap_points3d_bin(run_dir / "sparse" / "0" / "points3D.bin")

    reaks_feature = reactions_graph = reactions_clustering = reactions_key = 0.0
    if method.startswith("ratio_"):
        reaks_feature = reaks_stage_seconds(run_dir / "reaks", "feature_extraction")
        reactions_graph = reaks_stage_seconds(run_dir / "reaks", "similarity_graph")
        reactions_clustering = reaks_stage_seconds(run_dir / "reaks", "spectral_clustering")
        reactions_key = reaks_stage_seconds(run_dir / "reaks", "keyframe_selection")

    colmap_feature = seconds_from_stage(stage_rows, "colmap_feature_extraction")
    colmap_matching = seconds_from_stage(stage_rows, "colmap_feature_matching")
    colmap_sfm = seconds_from_stage(stage_rows, "colmap_mapping_sfm")
    train_command = seconds_from_stage(stage_rows, "3dgs_training_command")
    train_internal = timing_entry(timings, "3dgs_training")
    rendering = timing_entry(timings, "rendering_all_step") + timing_entry(timings, "trajectory_rendering_step")
    evaluation = timing_entry(timings, "evaluation_val_step")

    total_pipeline = (
        reaks_feature
        + reactions_graph
        + reactions_clustering
        + reactions_key
        + colmap_feature
        + colmap_matching
        + colmap_sfm
        + (train_internal or train_command)
        + rendering
        + evaluation
    )

    input_count = metadata.get("input_count", "")
    selected_count = metadata.get("selected_count", "")
    if not selected_count:
        selected_count = len([p for p in (run_dir / "images").glob("*") if p.is_file() or p.is_symlink()])
    if not input_count:
        try:
            input_count = round(float(selected_count) / float(ratio)) if float(ratio) > 0 else selected_count
        except (TypeError, ValueError):
            input_count = selected_count
    train_timing = timings.get("3dgs_training") or {}

    return {
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "ratio": ratio,
        "seed": seed,
        "input_images": input_count,
        "selected_images": selected_count,
        "retention_actual": safe_div(selected_count, input_count),
        "psnr": metrics.get("psnr", ""),
        "ssim": metrics.get("ssim", ""),
        "lpips": metrics.get("lpips", ""),
        "num_GS": metrics.get("num_GS", ""),
        "sfm_points": sfm.get("sfm_points", ""),
        "sfm_points_k": sfm.get("sfm_points_k", ""),
        "sfm_reproj_error_px": sfm.get("sfm_reproj_error_px", ""),
        "render_time_per_image_sec": metrics.get("ellipse_time", ""),
        "reaks_feature_extraction_sec": reaks_feature,
        "reaks_similarity_graph_sec": reactions_graph,
        "reaks_clustering_sec": reactions_clustering,
        "reaks_keyframe_selection_sec": reactions_key,
        "reaks_preprocessing_sec": reaks_feature + reactions_graph + reactions_clustering + reactions_key,
        "colmap_feature_extraction_sec": colmap_feature,
        "colmap_feature_matching_sec": colmap_matching,
        "colmap_sfm_mapping_sec": colmap_sfm,
        "colmap_total_sec": colmap_feature + colmap_matching + colmap_sfm,
        "3dgs_training_sec": train_internal or train_command,
        "rendering_sec": rendering,
        "evaluation_sec": evaluation,
        "total_pipeline_sec": total_pipeline,
        "hardware": (timings.get("setup") or {}).get("hardware", ""),
        "max_gpu_mem_gb": train_timing.get("max_gpu_mem_gb", ""),
        "max_steps": (timings.get("setup") or {}).get("max_steps", ""),
    }


def numeric(value) -> Optional[float]:
    if value == "" or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def mean_std(values: List[float]) -> str:
    if not values:
        return ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6g} +/- {std:.6g}"


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = (row["dataset"], row["method"], row["ratio"])
        groups.setdefault(key, []).append(row)

    metrics = [
        "psnr",
        "ssim",
        "lpips",
        "reaks_preprocessing_sec",
        "colmap_total_sec",
        "3dgs_training_sec",
        "rendering_sec",
        "evaluation_sec",
        "total_pipeline_sec",
        "max_gpu_mem_gb",
    ]
    out = []
    for (dataset, method, ratio), items in sorted(groups.items()):
        row = {
            "dataset": dataset,
            "method": method,
            "ratio": ratio,
            "num_runs": len(items),
            "num_scenes": len({x["scene"] for x in items}),
        }
        for metric in metrics:
            vals = [v for v in (numeric(x.get(metric)) for x in items) if v is not None]
            row[metric] = mean_std(vals)
        out.append(row)
    return out


def markdown_table(rows: List[Dict[str, object]]) -> str:
    headers = [
        "Dataset",
        "Method",
        "Runs",
        "REAKS prep (s)",
        "COLMAP/SfM (s)",
        "3DGS train (s)",
        "Render (s)",
        "Eval (s)",
        "Total pipeline (s)",
        "Max GPU mem (GB)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("method", "")),
                    str(row.get("num_runs", "")),
                    str(row.get("reaks_preprocessing_sec", "")),
                    str(row.get("colmap_total_sec", "")),
                    str(row.get("3dgs_training_sec", "")),
                    str(row.get("rendering_sec", "")),
                    str(row.get("evaluation_sec", "")),
                    str(row.get("total_pipeline_sec", "")),
                    str(row.get("max_gpu_mem_gb", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", default="recon_major_3")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = collect_runs(args.roots, args.tag)
    rows = [run_to_row(run_dir, args.tag) for run_dir in run_dirs]

    per_scene_fields = [
        "dataset",
        "scene",
        "method",
        "ratio",
        "seed",
        "input_images",
        "selected_images",
        "retention_actual",
        "psnr",
        "ssim",
        "lpips",
        "num_GS",
        "sfm_points",
        "sfm_points_k",
        "sfm_reproj_error_px",
        "render_time_per_image_sec",
        "reaks_feature_extraction_sec",
        "reaks_similarity_graph_sec",
        "reaks_clustering_sec",
        "reaks_keyframe_selection_sec",
        "reaks_preprocessing_sec",
        "colmap_feature_extraction_sec",
        "colmap_feature_matching_sec",
        "colmap_sfm_mapping_sec",
        "colmap_total_sec",
        "3dgs_training_sec",
        "rendering_sec",
        "evaluation_sec",
        "total_pipeline_sec",
        "hardware",
        "max_gpu_mem_gb",
        "max_steps",
    ]
    write_csv(args.out_dir / "reviewer3_per_scene_results.csv", rows, per_scene_fields)

    summary_rows = summarize(rows)
    summary_fields = [
        "dataset",
        "method",
        "ratio",
        "num_runs",
        "num_scenes",
        "psnr",
        "ssim",
        "lpips",
        "reaks_preprocessing_sec",
        "colmap_total_sec",
        "3dgs_training_sec",
        "rendering_sec",
        "evaluation_sec",
        "total_pipeline_sec",
        "max_gpu_mem_gb",
    ]
    write_csv(args.out_dir / "reviewer3_mean_std.csv", summary_rows, summary_fields)
    (args.out_dir / "reviewer3_total_pipeline_table.md").write_text(markdown_table(summary_rows))
    print(f"Found {len(rows)} runs. Wrote summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
