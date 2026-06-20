# REAKS: Adaptive Keyframe Selection for Resource-Efficient 3D Gaussian Splatting

[![Journal: The Visual Computer](https://img.shields.io/badge/Journal-The%20Visual%20Computer-blue)](https://www.springer.com/journal/371)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19414672.svg)](https://doi.org/10.5281/zenodo.19414672)
[![License: Research/Evaluation Non-Commercial](https://img.shields.io/badge/License-Research%2FEvaluation%20Non--Commercial-green)](LICENSE.md)

This repository contains the official implementation of **REAKS**: a resource-efficient adaptive keyframe selection framework for scalable 3D Gaussian Splatting (3DGS). REAKS selects informative frames before COLMAP/SfM and 3DGS training, reducing reconstruction time and peak GPU memory while preserving rendering quality.

The codebase is based on the official GraphDECO 3D Gaussian Splatting implementation and adds REAKS selection scripts, reproduction workflows, selected-frame records, timing logs, and table/figure aggregation utilities.

## Abstract

3D Gaussian Splatting (3DGS) enables high-fidelity neural rendering, but large-scale scenes can make Structure-from-Motion (SfM), training, and memory usage expensive. REAKS uses multi-source hierarchical feature fusion and adaptive spectral clustering to remove redundant frames before reconstruction. Across natural and medical endoscopic scenes, REAKS accelerates SfM and reduces peak GPU memory while maintaining comparable rendering quality.

## Reproducibility Snapshot

Use the following immutable revision for the experiments described in the revision package:

```bash
git clone --recursive https://github.com/PeiZhaoZhang/REAKS.git
cd REAKS
git checkout 52f95b268129bd2857a6172ca2f4c12506567ec2
```

Recommended release tag for the camera-ready archive:

```bash
git tag -a v1.0.0-reproducibility 52f95b268129bd2857a6172ca2f4c12506567ec2 \
  -m "Reproducibility release for REAKS"
git push origin v1.0.0-reproducibility
```

In this working copy, no tag is currently attached to the commit, so the tag above should be created before the public release archive is finalized.

## Environment

The experiments were run in the `gs` conda environment. The complete environment file is [environment.yml](environment.yml); it pins the CUDA/PyTorch stack used by this repository and lists the extra packages required by REAKS, rendering, metrics, and ablation scripts.

```bash
conda env create -f environment.yml
conda activate gs

pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim

python - <<'PY'
import sys, torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
PY
```

Reference setup:

| Component | Version used |
| --- | --- |
| OS | Ubuntu 20.04 LTS |
| CUDA runtime | 11.6 via `cudatoolkit=11.6` |
| PyTorch | 1.12.1 |
| Torchvision | 0.13.1 |
| Python | 3.7.13 |
| GPU | NVIDIA RTX 3090-class GPU, 24 GB VRAM recommended |
| COLMAP | System COLMAP executable available as `colmap` |

## Data Layout

Place datasets under `/root/project/data` or override the paths in the commands below.

```text
/root/project/data/
  360_v2/
    bicycle/images_4/
    garden/images_4/
    ...
  tank_temples/
    Barn/images/
    Truck/images/
    ...
  C3VD/
    <sequence_name>/images/
```

For Mip-NeRF 360 scenes, the reproduction pipeline reads `images_4` by default. Tanks and Temples and C3VD examples read `images`. C3VD sequences should be arranged as COLMAP-style scene directories; if poses are absent, run COLMAP first with `convert.py` or the explicit COLMAP commands below.

## End-to-End Flow

The full reproducibility workflow is:

1. Select frames with REAKS.
2. Build COLMAP features, matches, and sparse geometry on the selected frames.
3. Train the 3DGS model on the selected subset.
4. Render validation and test views.
5. Compute PSNR, SSIM, and LPIPS.
6. Aggregate per-scene and per-dataset tables.
7. Collect SfM time and peak VRAM for the reviewer tables.

The sections below provide commands for each step.

## Core REAKS Selection

Run REAKS on any image folder:

```bash
conda activate gs
cd /root/project/gaussian-splatting

REAKS_SEED=42 python REAKS.py \
  -s /root/project/data/360_v2/bicycle/images_4 \
  -m /root/project/data/360_v2/bicycle/reaks_ratio_050 \
  -r 0.50 \
  -a 1.0 \
  -p 75
```

Important outputs:

```text
<output>/original/input/            selected frames with original filenames
<output>/renamed/input/             optional renamed selected frames
<output>/reaks_stage_timings.csv    feature, graph, clustering, and selection timings
<output>/reaks_timing_metadata.json  input count, selected count, ratio, seed, hardware metadata
```

## Single-Scene 3DGS Workflow

This is the complete manual path for one selected-frame scene: REAKS selection, COLMAP, training, rendering, and metrics.

```bash
conda activate gs
cd /root/project/gaussian-splatting

SCENE=/root/project/data/360_v2/bicycle
RUN=$SCENE/repro_readme/ratio_0.50/seed_42
mkdir -p "$RUN/images" "$RUN/sparse"

REAKS_SEED=42 python REAKS.py \
  -s "$SCENE/images_4" \
  -m "$RUN/reaks" \
  -r 0.50

find "$RUN/reaks/original/input" -maxdepth 1 -type f \
  -exec ln -sf {} "$RUN/images/" \;

colmap feature_extractor \
  --database_path "$RUN/database.db" \
  --image_path "$RUN/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.max_num_features 8192

colmap exhaustive_matcher \
  --database_path "$RUN/database.db" \
  --SiftMatching.use_gpu 1

colmap mapper \
  --database_path "$RUN/database.db" \
  --image_path "$RUN/images" \
  --output_path "$RUN/sparse" \
  --Mapper.ba_global_function_tolerance 0.000001

test -d "$RUN/sparse/0" || mv "$RUN/sparse/"*/ "$RUN/sparse/0"

python train.py \
  -s "$RUN" \
  -m "$RUN/output" \
  --images images \
  --iterations 30000 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  --disable_viewer

python render.py \
  -s "$RUN" \
  -m "$RUN/output" \
  --iteration 30000 \
  --skip_train

python metrics.py \
  -m "$RUN/output"
```

For an original-frame baseline with the same frame budget, replace the REAKS command with a deterministic uniform subset and keep the remaining COLMAP/train/render/metrics commands unchanged.

## Dataset Reproduction Commands

The main revision pipeline is [major_revision/run_reviewer3_pipeline.sh](major_revision/run_reviewer3_pipeline.sh). It reproduces the REAKS-vs-original timing, SfM, 3DGS, rendering, metric, and summary-table workflow. The script defaults to the older `gsplat` name internally, so pass `CONDA_ENV=gs` for this environment.

### Mip-NeRF 360

```bash
cd /root/project/gaussian-splatting/major_revision

CONDA_ENV=gs \
DATA_ROOT_360=/root/project/data/360_v2 \
DATA_ROOT_TNT=/root/project/data/tank_temples \
SCENES_360="bicycle garden stump room counter kitchen bonsai" \
SCENES_TNT="" \
SEEDS="42" \
RETENTION_RATIOS="0.50" \
BASELINE_FRAME_RATIO="0.50" \
DATA_FACTOR_360=4 \
OUT_TAG=recon_major_3_double_single_gpu \
MAX_STEPS=30000 \
TEST_EVERY=8 \
GPU_LIST=0 \
bash run_reviewer3_pipeline.sh
```

### Tanks and Temples

```bash
cd /root/project/gaussian-splatting/major_revision

CONDA_ENV=gs \
DATA_ROOT_360=/root/project/data/360_v2 \
DATA_ROOT_TNT=/root/project/data/tank_temples \
SCENES_360="" \
SCENES_TNT="Barn Church Courthouse Meetingroom Museum Palace Truck" \
SEEDS="42" \
RETENTION_RATIOS="0.50" \
BASELINE_FRAME_RATIO="0.50" \
OUT_TAG=recon_major_3_double_single_gpu \
MAX_STEPS=30000 \
TEST_EVERY=8 \
GPU_LIST=0 \
bash run_reviewer3_pipeline.sh
```

### C3VD / Endoscopic Sequences

The REAKS selector is dataset-agnostic and can be applied to C3VD once each sequence is placed in a COLMAP-style folder. Example for one C3VD sequence:

```bash
conda activate gs
cd /root/project/gaussian-splatting

SEQ=/root/project/data/C3VD/cecum_t1_a
RUN=$SEQ/repro_readme/ratio_0.50/seed_42
mkdir -p "$RUN/images" "$RUN/sparse"

REAKS_SEED=42 python REAKS.py \
  -s "$SEQ/images" \
  -m "$RUN/reaks" \
  -r 0.50

find "$RUN/reaks/original/input" -maxdepth 1 -type f \
  -exec ln -sf {} "$RUN/images/" \;

colmap feature_extractor \
  --database_path "$RUN/database.db" \
  --image_path "$RUN/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 1
colmap exhaustive_matcher \
  --database_path "$RUN/database.db" \
  --SiftMatching.use_gpu 1
colmap mapper \
  --database_path "$RUN/database.db" \
  --image_path "$RUN/images" \
  --output_path "$RUN/sparse" \
  --Mapper.ba_global_function_tolerance 0.000001
test -d "$RUN/sparse/0" || mv "$RUN/sparse/"*/ "$RUN/sparse/0"

python train.py \
  -s "$RUN" \
  -m "$RUN/output" \
  --images images \
  --iterations 30000 \
  --disable_viewer
python render.py \
  -s "$RUN" \
  -m "$RUN/output" \
  --iteration 30000 \
  --skip_train
python metrics.py \
  -m "$RUN/output"
```

For the full C3VD table, repeat the same command block for each sequence and aggregate the resulting metric JSON/CSV files with the same schema as `major_revision/aggregate_experiment3_metrics.py`.

## Tables and Figures

Regenerate the main timing and quality CSV/JSON files:

```bash
conda activate gs
cd /root/project/gaussian-splatting
python major_revision/aggregate_experiment3_metrics.py
```

Primary outputs:

```text
major_revision/experiment3_tables/experiment3_per_scene_metrics.csv
major_revision/experiment3_tables/experiment3_dataset_summary.csv
major_revision/experiment3_tables/experiment3_pipeline_time.csv
major_revision/experiment3_tables/experiment3_percentage_changes.csv
major_revision/experiment3_tables/experiment3_ne_metrics.csv
major_revision/experiment3_tables/experiment3_ne_summary.csv
major_revision/experiment3_tables/missing_fields_report.csv
```

Regenerate the stronger-baseline comparison figure and table for the Mip-NeRF 360 bicycle scene:

```bash
cd /root/project/gaussian-splatting/major_revision

CONDA_ENV=gs CUDA_DEVICE=0 MAX_STEPS=30000 SEED=42 RUN_TRAIN=1 bash run_reviewer4_baselines.sh
```

Outputs:

```text
major_revision/reviewer4_baselines/reviewer4_8method_metrics.csv
major_revision/reviewer4_baselines/reviewer4_8method_summary.csv
major_revision/reviewer4_baselines/reviewer4_8method_table.tex
major_revision/reviewer4_baselines/reviewer4_8method_visual_comparison.png
major_revision/reviewer4_baselines/reviewer4_8method_visual_comparison.pdf
major_revision/reviewer4_baselines/reviewer4_best_view_metrics.csv
```

## Selected-Frame Indices and Logs

The repository stores selected-frame records and timing artifacts so reviewers can audit the exact frames and resource measurements.

```text
major_revision/reviewer4_baselines/pose_fps/selected_images.txt
major_revision/reviewer4_baselines/pose_fps/selected_images.json
major_revision/reviewer4_baselines/coverage_greedy/selected_images.txt
major_revision/reviewer4_baselines/coverage_greedy/selected_images.json
major_revision/reviewer4_baselines/sfm_covisibility_greedy/selected_images.txt
major_revision/reviewer4_baselines/sfm_covisibility_greedy/selected_images.json
```

Pipeline-created selected frames and logs are stored under each run directory:

```text
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/images/
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/selection_metadata.json
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/reaks/reaks_stage_timings.csv
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/reaks/reaks_timing_metadata.json
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/stage_timings.csv
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/logs/colmap_feature_extraction.log
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/logs/colmap_feature_matching.log
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/logs/colmap_mapping_sfm.log
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/logs/3dgs_training.log
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/results/stats/pipeline_timings.json
/root/project/data/<dataset>/<scene>/<OUT_TAG>/<method>/seed_<seed>/results/stats/val_step29999.json
```

Project-level long logs are also included:

```text
major_revision/recon_major_3_seed42.log
major_revision/recon_major_3_double_seed42.log
major_revision/recon_major_3_double_single_gpu_seed42.log
major_revision/reviewer4_baselines/*/train.log
major_revision/reviewer4_baselines/*/training_elapsed_sec.txt
```

The SfM time is reported by `stage_timings.csv` and summarized in `experiment3_pipeline_time.csv`. Peak VRAM is reported in `results/stats/pipeline_timings.json` and summarized as `max_gpu_mem_gb` in `experiment3_per_scene_metrics.csv`.

## Small Demo Tutorial

This smoke test uses a small subset of Mip-NeRF 360 bicycle and finishes much faster than the full 30k-step experiments.

```bash
conda activate gs
cd /root/project/gaussian-splatting

DEMO=/root/project/data/demo_reaks_bicycle
mkdir -p "$DEMO/input"
find /root/project/data/360_v2/bicycle/images_4 -maxdepth 1 -type f | sort | head -80   | xargs -I{} ln -sf {} "$DEMO/input/"

REAKS_SEED=42 python REAKS.py   -s "$DEMO/input"   -m "$DEMO/reaks_ratio_050"   -r 0.50

mkdir -p "$DEMO/run/images" "$DEMO/run/sparse"
find "$DEMO/reaks_ratio_050/original/input" -maxdepth 1 -type f   -exec ln -sf {} "$DEMO/run/images/" \;

colmap feature_extractor --database_path "$DEMO/run/database.db" --image_path "$DEMO/run/images"   --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 1
colmap exhaustive_matcher --database_path "$DEMO/run/database.db" --SiftMatching.use_gpu 1
colmap mapper --database_path "$DEMO/run/database.db" --image_path "$DEMO/run/images"   --output_path "$DEMO/run/sparse" --Mapper.ba_global_function_tolerance 0.000001
test -d "$DEMO/run/sparse/0" || mv "$DEMO/run/sparse/"*/ "$DEMO/run/sparse/0"

python train.py   -s "$DEMO/run"   -m "$DEMO/run/output"   --images images   --iterations 1000   --test_iterations 1000   --save_iterations 1000   --disable_viewer

python render.py -s "$DEMO/run" -m "$DEMO/run/output" --iteration 1000 --skip_train
python metrics.py -m "$DEMO/run/output"
```

## Reproducibility Checklist

A supplemental checklist is provided in [major_revision/reproducibility_checklist.md](major_revision/reproducibility_checklist.md). Summary:

| Item requested by reviewer | Repository location |
| --- | --- |
| Fixed release tag and commit hash | This README, `v1.0.0-reproducibility`, commit `52f95b268129bd2857a6172ca2f4c12506567ec2` |
| Complete environment with CUDA/PyTorch | [environment.yml](environment.yml) |
| Scripts for each table and figure | [major_revision/run_reviewer3_pipeline.sh](major_revision/run_reviewer3_pipeline.sh), [major_revision/aggregate_experiment3_metrics.py](major_revision/aggregate_experiment3_metrics.py), [major_revision/run_reviewer4_baselines.sh](major_revision/run_reviewer4_baselines.sh) |
| Commands for Tanks and Temples, Mip-NeRF 360, C3VD | Dataset reproduction sections above |
| Selected-frame index files | `selected_images.txt/json`, run-level `images/`, `selection_metadata.json`, and `reaks_timing_metadata.json` |
| SfM timing and VRAM logs | `stage_timings.csv`, COLMAP logs, `pipeline_timings.json`, `experiment3_pipeline_time.csv` |
| Small demo tutorial | Small Demo Tutorial section above |
| License | [LICENSE.md](LICENSE.md), research/evaluation non-commercial license inherited from GraphDECO/Inria/MPII |

## License

This repository follows the license in [LICENSE.md](LICENSE.md). The underlying Gaussian Splatting code is made available for non-commercial research and evaluation use by Inria and MPII. Some files include third-party permissive components as noted in the license file. Commercial use requires prior explicit consent from the licensors.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026reaks,
  title={REAKS: Adaptive Keyframe Selection for Resource-Efficient 3D Gaussian Splatting},
  author={Zhang, Peizhao and others},
  journal={The Visual Computer (Submitted)},
  year={2026},
  note={Official implementation: https://github.com/PeiZhaoZhang/REAKS}
}
```

## Acknowledgements

This repository builds on the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation. We thank the GraphDECO/Inria authors for the foundational 3DGS codebase.
