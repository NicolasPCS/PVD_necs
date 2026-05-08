#!/bin/bash

set -euo pipefail

SCORE_SCRIPT="ComputeScores.py"

REF_DIR="/home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics"
ABLATION_DIR="/home/ncaytuir/data-local/PVD_necs/MyScripts/AblationResults"

OUT_ORIGINAL="results_original_models_normalized.json"
OUT_MIRRORED="results_original_models_mirrored_objects_normalized.json"
OUT_SYMMETRICAL="results_symmetrical_models_normalized.json"

get_reference_pth () {
    CLASS=$1

    case "$CLASS" in
        airplane)
            echo "${REF_DIR}/reference_pth_airplane.pth"
            ;;
        car)
            echo "${REF_DIR}/reference_pth_car.pth"
            ;;
        chair)
            echo "${REF_DIR}/reference_pth_chair.pth"
            ;;
        *)
            echo "Unknown class: $CLASS" >&2
            exit 1
            ;;
    esac
}

run_score () {
    MODEL=$1
    CLASS=$2
    SAMPLE_PTH=$3
    OUT_JSON=$4

    REFERENCE_PTH=$(get_reference_pth "$CLASS")

    echo "========================================"
    echo "Model     : $MODEL"
    echo "Class     : $CLASS"
    echo "Sample    : $SAMPLE_PTH"
    echo "Reference : $REFERENCE_PTH"
    echo "Output    : $OUT_JSON"
    echo "========================================"

    python "$SCORE_SCRIPT" \
        --sample_pth "$SAMPLE_PTH" \
        --reference_pth "$REFERENCE_PTH" \
        --out_pth "$OUT_JSON"
}

# ============================================================
# DPM - ORIGINAL
# ============================================================

#run_score "dpm" "airplane" "/home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_dpm_airplane.pth" "$OUT_ORIGINAL"
#run_score "dpm" "car"      "/home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_dpm_car.pth"      "$OUT_ORIGINAL"
#run_score "dpm" "chair"    "/home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_dpm_chair.pth"    "$OUT_ORIGINAL"

# DPM - MIRRORED OBJECTS
run_score "dpm_mirrored" "airplane" "${ABLATION_DIR}/dpm_airplane.pth" "$OUT_MIRRORED"
run_score "dpm_mirrored" "car"      "${ABLATION_DIR}/dpm_car.pth"      "$OUT_MIRRORED"
run_score "dpm_mirrored" "chair"    "${ABLATION_DIR}/dpm_chair.pth"    "$OUT_MIRRORED"

# DPM - SYMMETRICAL / OURS
#run_score "dpm_ours" "airplane" "/home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_ours_airplane.pth" "$OUT_SYMMETRICAL"
#run_score "dpm_ours" "car"      "/home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_ours_car.pth"      "$OUT_SYMMETRICAL"
#run_score "dpm_ours" "chair"    "/home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_ours_chair.pth"    "$OUT_SYMMETRICAL"

# ============================================================
# PVD - ORIGINAL
# ============================================================

#run_score "pvd" "airplane" "/home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/airplane/generated_pvd_airplane.pth" "$OUT_ORIGINAL"
#run_score "pvd" "car"      "/home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/car/generated_pvd_car.pth"           "$OUT_ORIGINAL"
#run_score "pvd" "chair"    "/home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/chair/generated_pvd_chair.pth"       "$OUT_ORIGINAL"

# PVD - MIRRORED OBJECTS
run_score "pvd_mirrored" "airplane" "${ABLATION_DIR}/pvd_airplane.pth" "$OUT_MIRRORED"
run_score "pvd_mirrored" "car"      "${ABLATION_DIR}/pvd_car.pth"      "$OUT_MIRRORED"
run_score "pvd_mirrored" "chair"    "${ABLATION_DIR}/pvd_chair.pth"    "$OUT_MIRRORED"

# PVD - SYMMETRICAL / OURS
#run_score "pvd_ours" "airplane" "/home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/airplane/generated_ours_airplane.pth" "$OUT_SYMMETRICAL"
#run_score "pvd_ours" "car"      "/home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/car/generated_ours_car.pth"           "$OUT_SYMMETRICAL"
#run_score "pvd_ours" "chair"    "/home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/chair/generated_ours_chair.pth"       "$OUT_SYMMETRICAL"

# ============================================================
# LION - ORIGINAL
# ============================================================

#run_score "lion" "airplane" "/home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/samples_lion_airplane.pth" "$OUT_ORIGINAL"
#run_score "lion" "car"      "/home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/samples_lion_car.pth"      "$OUT_ORIGINAL"
#run_score "lion" "chair"    "/home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/samples_lion_chair.pth"    "$OUT_ORIGINAL"

# LION - MIRRORED OBJECTS
run_score "lion_mirrored" "airplane" "${ABLATION_DIR}/lion_airplane.pth" "$OUT_MIRRORED"
run_score "lion_mirrored" "car"      "${ABLATION_DIR}/lion_car.pth"      "$OUT_MIRRORED"
run_score "lion_mirrored" "chair"    "${ABLATION_DIR}/lion_chair.pth"    "$OUT_MIRRORED"

# LION - SYMMETRICAL / OURS
#run_score "lion_ours" "airplane" "/home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/generated_ours_airplane.pth" "$OUT_SYMMETRICAL"
#run_score "lion_ours" "car"      "/home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/generated_ours_car.pth"      "$OUT_SYMMETRICAL"
#run_score "lion_ours" "chair"    "/home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/generated_ours_chair.pth"    "$OUT_SYMMETRICAL"

# ============================================================
# DiT-3D - ORIGINAL
# ============================================================

#run_score "dit3d" "airplane" "/home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_dit3d_airplane.pth" "$OUT_ORIGINAL"
#run_score "dit3d" "car"      "/home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_dit3d_car.pth"      "$OUT_ORIGINAL"
#run_score "dit3d" "chair"    "/home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_dit3d_chair.pth"    "$OUT_ORIGINAL"

# DiT-3D - MIRRORED OBJECTS
run_score "dit3d_mirrored" "airplane" "${ABLATION_DIR}/dit3d_airplane.pth" "$OUT_MIRRORED"
run_score "dit3d_mirrored" "car"      "${ABLATION_DIR}/dit3d_car.pth"      "$OUT_MIRRORED"
run_score "dit3d_mirrored" "chair"    "${ABLATION_DIR}/dit3d_chair.pth"    "$OUT_MIRRORED"

# DiT-3D - SYMMETRICAL / OURS
#run_score "dit3d_ours" "airplane" "/home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_ours_airplane.pth" "$OUT_SYMMETRICAL"
#run_score "dit3d_ours" "car"      "/home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_ours_car.pth"      "$OUT_SYMMETRICAL"
#run_score "dit3d_ours" "chair"    "/home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_ours_chair.pth"    "$OUT_SYMMETRICAL"

# ============================================================
# XCube - ORIGINAL
# ============================================================

#run_score "xcube" "airplane" "/home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_airplane_xcube_2048.pth" "$OUT_ORIGINAL"
#run_score "xcube" "car"      "/home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_car_xcube_2048.pth"      "$OUT_ORIGINAL"
#run_score "xcube" "chair"    "/home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_chair_xcube_2048.pth"    "$OUT_ORIGINAL"

# XCube - MIRRORED OBJECTS
run_score "xcube_mirrored" "airplane" "${ABLATION_DIR}/xcube_airplane.pth" "$OUT_MIRRORED"
run_score "xcube_mirrored" "car"      "${ABLATION_DIR}/xcube_car.pth"      "$OUT_MIRRORED"
run_score "xcube_mirrored" "chair"    "${ABLATION_DIR}/xcube_chair.pth"    "$OUT_MIRRORED"

# XCube - SYMMETRICAL / OURS
#run_score "xcube_ours" "airplane" "/home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_ours_airplane.pth" "$OUT_SYMMETRICAL"
#run_score "xcube_ours" "car"      "/home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_ours_car.pth"      "$OUT_SYMMETRICAL"
#run_score "xcube_ours" "chair"    "/home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_ours_chair.pth"    "$OUT_SYMMETRICAL"

# ============================================================
# SLIDE3D - ORIGINAL
# ============================================================

#run_score "slide3d" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_airplane_slide_2048.pth" "$OUT_ORIGINAL"
#run_score "slide3d" "car"      "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_car_slide_2048.pth"      "$OUT_ORIGINAL"
#run_score "slide3d" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_chair_slide_2048.pth"    "$OUT_ORIGINAL"

# SLIDE3D - MIRRORED OBJECTS
run_score "slide3d_mirrored" "airplane" "${ABLATION_DIR}/slide3d_airplane.pth" "$OUT_MIRRORED"
run_score "slide3d_mirrored" "car"      "${ABLATION_DIR}/slide3d_car.pth"      "$OUT_MIRRORED"
run_score "slide3d_mirrored" "chair"    "${ABLATION_DIR}/slide3d_chair.pth"    "$OUT_MIRRORED"

# SLIDE3D - SYMMETRICAL / OURS
#run_score "slide3d_ours" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_ours_airplane.pth" "$OUT_SYMMETRICAL"
#run_score "slide3d_ours" "car"      "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_ours_car.pth"      "$OUT_SYMMETRICAL"
#run_score "slide3d_ours" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_ours_chair.pth"    "$OUT_SYMMETRICAL"

# ============================================================
# SPVD-S - ORIGINAL
# ============================================================

#run_score "spvd_s" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_spvd_s_airplane.pth" "$OUT_ORIGINAL"
#run_score "spvd_s" "car"      "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_spvd_s_car.pth"      "$OUT_ORIGINAL"
#run_score "spvd_s" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_spvd_s_chair.pth"    "$OUT_ORIGINAL"

# SPVD-S - MIRRORED OBJECTS
run_score "spvd_s_mirrored" "airplane" "${ABLATION_DIR}/spvd_s_airplane.pth" "$OUT_MIRRORED"
run_score "spvd_s_mirrored" "car"      "${ABLATION_DIR}/spvd_s_car.pth"      "$OUT_MIRRORED"
run_score "spvd_s_mirrored" "chair"    "${ABLATION_DIR}/spvd_s_chair.pth"    "$OUT_MIRRORED"

# SPVD-S - SYMMETRICAL / OURS
#run_score "spvd_s_ours" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_ours_airplane.pth" "$OUT_SYMMETRICAL"
#run_score "spvd_s_ours" "car"      "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_ours_car.pth"      "$OUT_SYMMETRICAL"
#run_score "spvd_s_ours" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_ours_chair.pth"    "$OUT_SYMMETRICAL"

# ============================================================
# SPVD-L - ORIGINAL
# ============================================================

#run_score "spvd_l" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_spvd_l_airplane.pth" "$OUT_ORIGINAL"
#run_score "spvd_l" "car"      "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_spvd_l_car.pth"      "$OUT_ORIGINAL"
#run_score "spvd_l" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_spvd_l_chair.pth"    "$OUT_ORIGINAL"

# SPVD-L - MIRRORED OBJECTS
run_score "spvd_l_mirrored" "airplane" "${ABLATION_DIR}/spvd_l_airplane.pth" "$OUT_MIRRORED"
run_score "spvd_l_mirrored" "car"      "${ABLATION_DIR}/spvd_l_car.pth"      "$OUT_MIRRORED"
run_score "spvd_l_mirrored" "chair"    "${ABLATION_DIR}/spvd_l_chair.pth"    "$OUT_MIRRORED"

# SPVD-L - SYMMETRICAL / OURS
#run_score "spvd_l_ours" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_ours_airplane.pth" "$OUT_SYMMETRICAL"
#run_score "spvd_l_ours" "car"      "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_ours_car.pth"      "$OUT_SYMMETRICAL"
#run_score "spvd_l_ours" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_ours_chair.pth"    "$OUT_SYMMETRICAL"

echo "All scores were computed successfully."

"""# DPM
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_dpm_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_dpm_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_dpm_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_ours_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_ours_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DPM/generated_pth/generated_ours_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm

# PVD
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/airplane/generated_pvd_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/car/generated_pvd_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/chair/generated_pvd_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/airplane/generated_ours_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/car/generated_ours_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/generated_pth/chair/generated_ours_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm

# LION
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/samples_lion_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/samples_lion_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/samples_lion_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/generated_ours_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/generated_ours_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/generated_pth/generated_ours_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm

# DiT-3D
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_dit3d_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_dit3d_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_dit3d_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_ours_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_ours_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_DiT3D/generated_pth/generated_ours_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm

# XCube
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_airplane_xcube_2048.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_car_xcube_2048.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_chair_xcube_2048.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_ours_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_ours_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/generated_pth/generated_ours_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm

# SLIDE 3D
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_airplane_slide_2048.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_car_slide_2048.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_chair_slide_2048.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_ours_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_ours_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_pth/generated_ours_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm

# SPVD-S
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_spvd_s_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_spvd_s_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_spvd_s_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_ours_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_ours_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_S/generated_pth/generated_ours_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm

# SPVD-L
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_spvd_l_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_spvd_l_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_spvd_l_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_original_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_ours_airplane.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_ours_car.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm
python ComputeScores.py \
  --sample_pth /home/ncaytuir/data/Datasets/Resultados_SPVD_L/generated_pth/generated_ours_chair.pth \
  --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth \
  --out_pth results_symmetrical_models.json \
  --req_norm"""

######################### OLDs
#PVD
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/airplane/ckpt_6199/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/car/ckpt_3299/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/chair/ckpt_1199/samples_pvd.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

#LION
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Airplane/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Car/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Chair/generated_pth/samples_lion.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

#XCube
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_airplane_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_car_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/generated_chair_xcube_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

#SLIDE 3D
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_airplane_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_car_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/generated_chair_slide_2048.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

######### OVER HALF DATASET

#PVD
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/airplane/ckpt_6199/samples_ours.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/car/ckpt_3299/samples_ours.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_PVD/Over_Half_Objects/chair/ckpt_1199/samples_ours.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

#LION
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Airplane/generated_pth/samples_ours_ckpt7999.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Car/generated_pth/samples_ours_ckpt7999.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_LION/Over_half/Chair/generated_pth/samples_ours_ckpt7999.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

#XCube
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/samples_ours_airplane.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/samples_ours_car.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_XCube/gen_pth/samples_ours_chair.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True

#SLIDE 3D
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/samples_our_airplane.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_airplane.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/samples_our_car.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_car.pth --req_norm True
#python ComputeScores.py --sample_pth /home/ncaytuir/data/Datasets/Resultados_SLIDE/gen_pth/samples_our_chair.pth --reference_pth /home/ncaytuir/data/Datasets/Reference_PTH_to_Compute_Metrics/reference_pth_chair.pth --req_norm True