#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Reviewer #3 rigorous timing pipeline for REAKS + COLMAP/SfM + 3DGS.
# Run examples:
#   bash run_reviewer3_pipeline.sh
#   SCENES_360="bicycle garden" SCENES_TNT="Barn Courthouse" SEEDS="42 43 44" bash run_reviewer3_pipeline.sh
#   RUN_COLMAP=0 RUN_TRAIN=0 bash run_reviewer3_pipeline.sh

DATA_ROOT_360=${DATA_ROOT_360:-/root/project/data/360_v2}
DATA_ROOT_TNT=${DATA_ROOT_TNT:-/root/project/data/tank_temples}
OUT_TAG=${OUT_TAG:-recon_major_3_double_single_gpu}
REAKS_SCRIPT=${REAKS_SCRIPT:-/root/project/gaussian-splatting/REAKS.py}
TRAINER=${TRAINER:-/root/project/gsplat/examples/simple_trainer_REAKS.py}
SUMMARY_SCRIPT=${SUMMARY_SCRIPT:-$SCRIPT_DIR/summarize_reviewer3_pipeline.py}
CONDA_ENV=${CONDA_ENV:-gsplat}

DATA_FACTOR_360=${DATA_FACTOR_360:-4}
RETENTION_RATIOS=${RETENTION_RATIOS:-0.50}
BASELINE_FRAME_RATIO=${BASELINE_FRAME_RATIO:-0.50}
SEEDS=${SEEDS:-42}
SCENES_360=${SCENES_360-bicycle}
SCENES_TNT=${SCENES_TNT-Barn}

RUN_FULL_BASELINE=${RUN_FULL_BASELINE:-1}
RUN_REAKS=${RUN_REAKS:-1}
RUN_COLMAP=${RUN_COLMAP:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_SUMMARY=${RUN_SUMMARY:-1}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}

COLMAP_BIN=${COLMAP_BIN:-colmap}
COLMAP_CAMERA=${COLMAP_CAMERA:-OPENCV}
COLMAP_USE_GPU=${COLMAP_USE_GPU:-1}
COLMAP_MATCHER=${COLMAP_MATCHER:-exhaustive_matcher}
MAX_NUM_FEATURES=${MAX_NUM_FEATURES:-8192}

MAX_STEPS=${MAX_STEPS:-30000}
TEST_EVERY=${TEST_EVERY:-8}
RENDER_TRAJ_PATH=${RENDER_TRAJ_PATH:-ellipse}
TB_EVERY=${TB_EVERY:-0}
GPU_LIST=${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1}}
GPU_LIST=${GPU_LIST//,/ }
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export PYTHONPATH=/root/project/gsplat/pycolmap:/root/project/gsplat/pycolmap/pycolmap:/root/project/gsplat/examples:/root/project/gsplat:${PYTHONPATH:-}

PROJECT_SUMMARY_DIR=${PROJECT_SUMMARY_DIR:-$SCRIPT_DIR/${OUT_TAG}_summary}
mkdir -p "$PROJECT_SUMMARY_DIR"

now_sec() {
    python -c 'import time; print(f"{time.time():.6f}")'
}

elapsed_sec() {
    python - "$1" <<'PY'
import sys, time
start = float(sys.argv[1])
print(f"{time.time() - start:.6f}")
PY
}

csv_escape() {
    python - "$1" <<'PY'
import csv, io, sys
buf = io.StringIO()
csv.writer(buf).writerow([sys.argv[1]])
print(buf.getvalue().strip())
PY
}

init_timing_csv() {
    local csv_path=$1
    if [[ ! -f "$csv_path" ]]; then
        echo "dataset,scene,method,ratio,seed,stage,elapsed_sec,status,detail" > "$csv_path"
    fi
}

append_timing() {
    local csv_path=$1 dataset=$2 scene=$3 method=$4 ratio=$5 seed=$6 stage=$7 elapsed=$8 status=$9 detail=${10:-}
    init_timing_csv "$csv_path"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape "$dataset")" "$(csv_escape "$scene")" "$(csv_escape "$method")" \
        "$(csv_escape "$ratio")" "$(csv_escape "$seed")" "$(csv_escape "$stage")" \
        "$(csv_escape "$elapsed")" "$(csv_escape "$status")" "$(csv_escape "$detail")" >> "$csv_path"
}

run_stage() {
    local csv_path=$1 dataset=$2 scene=$3 method=$4 ratio=$5 seed=$6 stage=$7 log_path=$8
    shift 8
    local start elapsed status
    start=$(now_sec)
    echo "[stage] $dataset/$scene $method seed=$seed $stage"
    echo "[cmd] $*" > "$log_path"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY_RUN: $*" >> "$log_path"
        elapsed=$(elapsed_sec "$start")
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "$stage" "$elapsed" "dry_run" "$log_path"
        return 0
    fi
    set +e
    "$@" >> "$log_path" 2>&1
    status=$?
    set -e
    elapsed=$(elapsed_sec "$start")
    if [[ $status -eq 0 ]]; then
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "$stage" "$elapsed" "ok" "$log_path"
    else
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "$stage" "$elapsed" "failed" "$log_path"
        echo "Stage failed: $stage. See $log_path" >&2
        return $status
    fi
}

prepare_tank_scene() {
    local scene_dir=$1
    local image_dir="$scene_dir/images"
    mkdir -p "$image_dir"
    shopt -s nullglob
    local moved=0
    for f in "$scene_dir"/*.jpg "$scene_dir"/*.jpeg "$scene_dir"/*.png "$scene_dir"/*.JPG "$scene_dir"/*.PNG; do
        local dst="$image_dir/$(basename "$f")"
        if [[ -e "$dst" ]]; then
            continue
        fi
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "DRY_RUN: mv $f $dst"
        else
            mv "$f" "$dst"
        fi
        moved=$((moved + 1))
    done
    shopt -u nullglob
    echo "Prepared Tank scene $(basename "$scene_dir"): moved $moved root images into images/." >&2
}

link_images_from_dir() {
    local src_dir=$1 dst_dir=$2
    rm -rf "$dst_dir"
    mkdir -p "$dst_dir"
    shopt -s nullglob
    local count=0
    for f in "$src_dir"/*; do
        [[ -f "$f" ]] || continue
        case "${f,,}" in
            *.jpg|*.jpeg|*.png)
                ln -s "$(realpath "$f")" "$dst_dir/$(basename "$f")"
                count=$((count + 1))
                ;;
        esac
    done
    shopt -u nullglob
    if [[ $count -eq 0 ]]; then
        echo "No images found in $src_dir" >&2
        return 1
    fi
}

link_uniform_subset_from_dir() {
    local src_dir=$1 dst_dir=$2 keep_ratio=$3
    rm -rf "$dst_dir"
    mkdir -p "$dst_dir"
    python - "$src_dir" "$dst_dir" "$keep_ratio" <<'PY'
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
ratio = float(sys.argv[3])
if not (0.0 < ratio <= 1.0):
    raise SystemExit(f"keep_ratio must be in (0, 1], got {ratio}")
images = sorted(
    p for p in src.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
if not images:
    raise SystemExit(f"No images found in {src}")
keep = max(1, int(round(len(images) * ratio)))
if keep >= len(images):
    selected = images
elif keep == 1:
    selected = [images[len(images) // 2]]
else:
    selected_indices = sorted({round(i * (len(images) - 1) / (keep - 1)) for i in range(keep)})
    selected = [images[i] for i in selected_indices]
for p in selected:
    target = dst / p.name
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(p.resolve(), target)
metadata = {
    "source_path": str(src),
    "selection_method": "uniform_original_frame_subset",
    "retention_ratio": ratio,
    "input_count": len(images),
    "selected_count": len(selected),
}
(dst.parent / "selection_metadata.json").write_text(__import__("json").dumps(metadata, indent=2) + "\n")
print(f"Uniform baseline subset: selected {len(selected)} / {len(images)} images from {src}")
PY
}

image_source_for_scene() {
    local dataset=$1 scene_dir=$2
    if [[ "$dataset" == "360_v2" && -d "$scene_dir/images_${DATA_FACTOR_360}" ]]; then
        echo "$scene_dir/images_${DATA_FACTOR_360}"
    else
        echo "$scene_dir/images"
    fi
}

run_colmap_pipeline() {
    local csv_path=$1 dataset=$2 scene=$3 method=$4 ratio=$5 seed=$6 run_dir=$7
    local image_dir="$run_dir/images"
    local sparse_done="$run_dir/sparse/0/images.bin"
    local db_path="$run_dir/database.db"
    local sparse_root="$run_dir/sparse"
    local log_dir="$run_dir/logs"
    mkdir -p "$sparse_root" "$log_dir"

    if [[ "$SKIP_EXISTING" == "1" && -f "$sparse_done" ]]; then
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "colmap_feature_extraction" "0" "skipped" "$sparse_done exists"
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "colmap_feature_matching" "0" "skipped" "$sparse_done exists"
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "colmap_mapping_sfm" "0" "skipped" "$sparse_done exists"
        return 0
    fi

    rm -f "$db_path"
    rm -rf "$sparse_root"
    mkdir -p "$sparse_root"

    run_stage "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "colmap_feature_extraction" "$log_dir/colmap_feature_extraction.log" \
        "$COLMAP_BIN" feature_extractor \
        --database_path "$db_path" \
        --image_path "$image_dir" \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model "$COLMAP_CAMERA" \
        --SiftExtraction.use_gpu "$COLMAP_USE_GPU" \
        --SiftExtraction.max_num_features "$MAX_NUM_FEATURES"

    run_stage "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "colmap_feature_matching" "$log_dir/colmap_feature_matching.log" \
        "$COLMAP_BIN" "$COLMAP_MATCHER" \
        --database_path "$db_path" \
        --SiftMatching.use_gpu "$COLMAP_USE_GPU"

    run_stage "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "colmap_mapping_sfm" "$log_dir/colmap_mapping_sfm.log" \
        "$COLMAP_BIN" mapper \
        --database_path "$db_path" \
        --image_path "$image_dir" \
        --output_path "$sparse_root" \
        --Mapper.ba_global_function_tolerance 0.000001

    if [[ ! -f "$sparse_done" ]]; then
        local first_model
        first_model=$(find "$sparse_root" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1 || true)
        if [[ -n "$first_model" && "$first_model" != "$sparse_root/0" ]]; then
            rm -rf "$sparse_root/0"
            mv "$first_model" "$sparse_root/0"
        fi
    fi
    [[ -f "$sparse_done" ]] || { echo "COLMAP did not produce $sparse_done" >&2; return 1; }
}

run_training() {
    local csv_path=$1 dataset=$2 scene=$3 method=$4 ratio=$5 seed=$6 run_dir=$7
    local result_dir="$run_dir/results"
    local log_dir="$run_dir/logs"
    mkdir -p "$result_dir" "$log_dir"
    if [[ "$SKIP_EXISTING" == "1" && -f "$result_dir/stats/val_step$(printf '%04d' $((MAX_STEPS - 1))).json" ]]; then
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "3dgs_training_command" "0" "skipped" "existing eval stats"
        return 0
    fi
    run_stage "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "3dgs_training_command" "$log_dir/3dgs_training.log" \
        conda run -n "$CONDA_ENV" python "$TRAINER" default \
        --disable-viewer \
        --disable-video \
        --data-factor 1 \
        --data-dir "$run_dir" \
        --result-dir "$result_dir" \
        --test-every "$TEST_EVERY" \
        --max-steps "$MAX_STEPS" \
        --eval-steps "$MAX_STEPS" \
        --save-steps "$MAX_STEPS" \
        --seed "$seed" \
        --render-traj-path "$RENDER_TRAJ_PATH" \
        --tb-every "$TB_EVERY"
}

run_one_method() {
    local dataset=$1 scene=$2 scene_dir=$3 method=$4 ratio=$5 seed=$6 image_src=$7 assigned_gpu=${8:-0}
    export CUDA_VISIBLE_DEVICES="$assigned_gpu"
    local run_dir="$scene_dir/$OUT_TAG/$method/seed_${seed}"
    local csv_path="$run_dir/stage_timings.csv"
    mkdir -p "$run_dir/logs"
    init_timing_csv "$csv_path"

    if [[ "$method" == original ]]; then
        link_uniform_subset_from_dir "$image_src" "$run_dir/images" "$BASELINE_FRAME_RATIO"
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "baseline_original_frame_sampling" "0" "ok" "uniform subset from doubled candidate pool"
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "reaks_feature_extraction" "0" "not_applicable" "original-frame baseline"
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "reaks_clustering" "0" "not_applicable" "original-frame baseline"
        append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "reaks_keyframe_selection" "0" "not_applicable" "original-frame baseline"
    else
        local reaks_dir="$run_dir/reaks"
        if [[ "$SKIP_EXISTING" == "1" && -f "$reaks_dir/reaks_timing_metadata.json" ]]; then
            append_timing "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "reaks_total" "0" "skipped" "existing REAKS metadata"
        else
            rm -rf "$reaks_dir"
            export REAKS_SEED="$seed"
            run_stage "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "reaks_total" "$run_dir/logs/reaks.log" \
                conda run -n "$CONDA_ENV" python "$REAKS_SCRIPT" \
                -s "$image_src" \
                -m "$reaks_dir" \
                -r "$ratio"
        fi
        link_images_from_dir "$reaks_dir/original/input" "$run_dir/images"
    fi

    if [[ "$RUN_COLMAP" == "1" ]]; then
        run_colmap_pipeline "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "$run_dir"
    fi
    if [[ "$RUN_TRAIN" == "1" ]]; then
        run_training "$csv_path" "$dataset" "$scene" "$method" "$ratio" "$seed" "$run_dir"
    fi
}

prepare_scene() {
    local dataset=$1 scene=$2 scene_dir=$3
    if [[ "$dataset" == "tank_temples" ]]; then
        prepare_tank_scene "$scene_dir"
    fi
    local image_src
    image_src=$(image_source_for_scene "$dataset" "$scene_dir")
    [[ -d "$image_src" ]] || { echo "Missing image source: $image_src" >&2; return 1; }
    echo "$image_src"
}

declare -A GPU_PID
GPU_IDS=($GPU_LIST)
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "GPU_LIST is empty" >&2
    exit 1
fi

free_finished_gpus() {
    local gpu pid
    for gpu in "${GPU_IDS[@]}"; do
        pid=${GPU_PID[$gpu]:-}
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            unset "GPU_PID[$gpu]"
        fi
    done
}

wait_for_free_gpu() {
    local gpu pid
    while true; do
        free_finished_gpus
        for gpu in "${GPU_IDS[@]}"; do
            pid=${GPU_PID[$gpu]:-}
            if [[ -z "$pid" ]]; then
                echo "$gpu"
                return 0
            fi
        done
        wait -n || true
    done
}

launch_run() {
    local dataset=$1 scene=$2 scene_dir=$3 method=$4 ratio=$5 seed=$6 image_src=$7
    local gpu
    gpu=$(wait_for_free_gpu)
    echo "[launch] GPU=$gpu $dataset/$scene $method seed=$seed"
    run_one_method "$dataset" "$scene" "$scene_dir" "$method" "$ratio" "$seed" "$image_src" "$gpu" &
    GPU_PID[$gpu]=$!
}

queue_scene() {
    local dataset=$1 scene=$2 scene_dir=$3 image_src=$4 seed ratio
    echo "=== Queue scene: $dataset/$scene, source images: $image_src ==="
    for seed in $SEEDS; do
        if [[ "$RUN_FULL_BASELINE" == "1" ]]; then
            launch_run "$dataset" "$scene" "$scene_dir" "original" "$BASELINE_FRAME_RATIO" "$seed" "$image_src"
        fi
        if [[ "$RUN_REAKS" == "1" ]]; then
            for ratio in $RETENTION_RATIOS; do
                launch_run "$dataset" "$scene" "$scene_dir" "ratio_${ratio}" "$ratio" "$seed" "$image_src"
            done
        fi
    done
}

for scene in $SCENES_360; do
    scene_dir="$DATA_ROOT_360/$scene"
    image_src=$(prepare_scene "360_v2" "$scene" "$scene_dir")
    queue_scene "360_v2" "$scene" "$scene_dir" "$image_src"
done

for scene in $SCENES_TNT; do
    scene_dir="$DATA_ROOT_TNT/$scene"
    image_src=$(prepare_scene "tank_temples" "$scene" "$scene_dir")
    queue_scene "tank_temples" "$scene" "$scene_dir" "$image_src"
done

for gpu in "${GPU_IDS[@]}"; do
    pid=${GPU_PID[$gpu]:-}
    if [[ -n "$pid" ]]; then
        wait "$pid"
    fi
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
    python "$SUMMARY_SCRIPT" \
        --roots "$DATA_ROOT_360" "$DATA_ROOT_TNT" \
        --out-dir "$PROJECT_SUMMARY_DIR" \
        --tag "$OUT_TAG"
    echo "Summary written to $PROJECT_SUMMARY_DIR"
fi
