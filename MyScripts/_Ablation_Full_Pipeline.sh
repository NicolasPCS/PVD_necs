#!/bin/bash

OUT_DIR="/home/ncaytuir/data-local/PVD_necs/MyScripts/AblationResults"
SCRIPT="MyScripts/_Ablation_Full_Pipeline.py"

run_ablation () {
    MODEL=$1
    CLASS=$2
    INPUT_PATH=$3

    OUTPUT_PATH="${OUT_DIR}/${MODEL}_${CLASS}.pth"

    echo "========================================"
    echo "Running: ${MODEL} - ${CLASS}"
    echo "Input : ${INPUT_PATH}"
    echo "Output: ${OUTPUT_PATH}"
    echo "========================================"

    python "$SCRIPT" \
        "$INPUT_PATH" \
        "$OUTPUT_PATH"
}

# ============================================================
# DPM
# ============================================================

run_ablation "dpm" "airplane" "/home/ncaytuir/data/Datasets/Resultados_DPM/original_airplane_1776364329/original"
run_ablation "dpm" "car"      "/home/ncaytuir/data/Datasets/Resultados_DPM/original_car_1776364397/original"
run_ablation "dpm" "chair"    "/home/ncaytuir/data/Datasets/Resultados_DPM/original_chair_1776364617/original"

# ============================================================
# PVD
# ============================================================

run_ablation "pvd" "airplane" "/home/ncaytuir/data/Datasets/Resultados_PVD/NewExp_ArchitectureAssessing/original_ckpts/airplane/original"
run_ablation "pvd" "car"      "/home/ncaytuir/data/Datasets/Resultados_PVD/NewExp_ArchitectureAssessing/original_ckpts/car/original"
run_ablation "pvd" "chair"    "/home/ncaytuir/data/Datasets/Resultados_PVD/NewExp_ArchitectureAssessing/original_ckpts/chair/original"

# ============================================================
# LION
# ============================================================

run_ablation "lion" "airplane" "/home/ncaytuir/data/Datasets/Resultados_LION/Airplane/pcs"
run_ablation "lion" "car"      "/home/ncaytuir/data/Datasets/Resultados_LION/Car/pcs"
run_ablation "lion" "chair"    "/home/ncaytuir/data/Datasets/Resultados_LION/Chair/pcs"

# ============================================================
# DiT-3D
# ============================================================

run_ablation "dit3d" "airplane" "/home/ncaytuir/data/Datasets/Resultados_DiT3D/original_airplane/original"
run_ablation "dit3d" "car"      "/home/ncaytuir/data/Datasets/Resultados_DiT3D/original_car/original"
run_ablation "dit3d" "chair"    "/home/ncaytuir/data/Datasets/Resultados_DiT3D/original_chair/original"

# ============================================================
# XCube
# ============================================================

run_ablation "xcube" "airplane" "/home/ncaytuir/data/Datasets/Resultados_XCube/Airplane/completes"
run_ablation "xcube" "car"      "/home/ncaytuir/data/Datasets/Resultados_XCube/Car/Completes"
run_ablation "xcube" "chair"    "/home/ncaytuir/data/Datasets/Resultados_XCube/Chair/Completes"

# ============================================================
# SLIDE3D
# ============================================================

run_ablation "slide3d" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_point_cloud_and_mesh/airplane/centroid/completes"
run_ablation "slide3d" "car"      "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_point_cloud_and_mesh/car/centroid/completes"
run_ablation "slide3d" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SLIDE/generated_point_cloud_and_mesh/chair/centroid/completes"

# ============================================================
# SPVD-S
# ============================================================

run_ablation "spvd_s" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/original_airplane/original"
run_ablation "spvd_s" "car"      "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/original_car/original"
run_ablation "spvd_s" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SPVD_S/original_chair/original"

# ============================================================
# SPVD-L
# ============================================================

run_ablation "spvd_l" "airplane" "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/original_airplane/original"
run_ablation "spvd_l" "car"      "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/original_car/original"
run_ablation "spvd_l" "chair"    "/home/ncaytuir/data/Datasets/Resultados_SPVD_L/original_chair/original"

echo "All ablation files were generated successfully."

echo "Computing scores"

CUDA_VISIBLE_DEVICES=1 bash ComputeScores.bash