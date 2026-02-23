#!/bin/bash
# ==============================================================
# HCP Emotion Task - FULL ANALYSIS PIPELINE (REVISED)
# ==============================================================
#
# Revision changes addressing peer review:
#   MC1: Equalized thresholds (FDR q<0.05 for both GLM and LME)
#   MC2: Acquisition parameters documented in config.py
#   MC3: Extended references (in manuscript, not code)
#   MC4: Added LME implementation validation step
#   MC5: Analysis space clarified (MNI volumetric)
#   MC6: ICC limitations documented
#   mC1: Smoothing fixed to 5mm FWHM consistently
#   mC2: ROI selection justified in code docstrings
#   mC3: GitHub URL must be updated before submission
#   mC5: Figure 4 now includes all 3 panels
#   mC7: Memory per LME process documented in config
#
# Estimated time:
#   - Validation:               ~5 min
#   - ROI + Stability + Bench:  ~30 min
#   - Full voxelwise LME:      3-6 hours
#
# Usage:
#   ./run_full_analysis_v2.sh           # Full pipeline
#   ./run_full_analysis_v2.sh --quick   # ROI only (fast)
#   ./run_full_analysis_v2.sh --resume  # Resume voxelwise
#

set -e

# ---------------------------------------------------------------
# Thread control: prevent numpy/MKL/OpenBLAS/R from spawning
# internal threads within each parallel worker. Each worker should
# use exactly 1 core; parallelism comes from joblib, not from
# nested threading. This DRAMATICALLY improves throughput on
# multi-core machines (12 vCPU / 64 GB target).
# ---------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# R also respects OMP_NUM_THREADS for its BLAS calls

QUICK_MODE=false
RESUME=""

for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --resume) RESUME="--resume" ;;
    esac
done

echo "=========================================================="
echo "HCP EMOTION TASK - GLM vs LME ANALYSIS (REVISED)"
echo "Complete Pipeline for Publication"
echo "=========================================================="
echo ""

# Check data
if [ ! -d "/data/hcp_emotion/raw" ]; then
    echo "ERROR: Data not found at /data/hcp_emotion/raw"
    exit 1
fi

N_SUBJECTS=$(ls -d /data/hcp_emotion/raw/*/ 2>/dev/null | wc -l)
echo "Found $N_SUBJECTS subjects"
echo ""

# ===========================================================
# STEP 0: Validate LME implementation (MC1)
# Now performs formal comparison: statsmodels vs R lme4 (pymer4)
# ===========================================================
echo "-----------------------------------------------------------"
echo "STEP 0: LME Implementation Validation (MC1)"
echo "  Formal comparison: statsmodels vs R lme4 (via pymer4)"
echo "-----------------------------------------------------------"
python validation/validate_lme_implementation.py --n-sims 100
echo ""

# ===========================================================
# STEP 1: First-level GLM (with motion QC)
# ===========================================================
echo "-----------------------------------------------------------"
echo "STEP 1: First-level GLM (with motion QC)"
echo "-----------------------------------------------------------"
python main_first_level.py --n-jobs 10

# ===========================================================
# STEP 2: Group-level GLM (now with FDR correction — MC1)
# ===========================================================
echo ""
echo "-----------------------------------------------------------"
echo "STEP 2: Group-level GLM (FDR + uncorrected thresholds)"
echo "-----------------------------------------------------------"
python main_group_glm.py

# ===========================================================
# STEP 3: ROI Analysis
# ===========================================================
echo ""
echo "-----------------------------------------------------------"
echo "STEP 3: ROI Analysis (GLM vs LME)"
echo "-----------------------------------------------------------"
python main_roi_analysis_v2.py

# ===========================================================
# STEP 4: Stability Analysis
# ===========================================================
echo ""
echo "-----------------------------------------------------------"
echo "STEP 4: Stability Analysis (Bootstrap, Split-Half, LOO)"
echo "-----------------------------------------------------------"
python analysis_stability_v2.py --n-bootstrap 1000

# ===========================================================
# STEP 5: Computational Benchmark
# ===========================================================
echo ""
echo "-----------------------------------------------------------"
echo "STEP 5: Computational Benchmark"
echo "-----------------------------------------------------------"
python analysis_benchmark.py

if [ "$QUICK_MODE" = true ]; then
    echo ""
    echo "Quick mode: Skipping voxelwise LME"
    echo ""
else
    # ===========================================================
    # STEP 6: Voxelwise LME (full model: beta ~ 1 + run)
    # ===========================================================
    echo ""
    echo "-----------------------------------------------------------"
    echo "STEP 6a: Voxelwise LME — full model (beta ~ 1 + run)"
    echo "-----------------------------------------------------------"
    python main_lme_voxelwise_v2.py --chunk-size 10000 --n-jobs 8 $RESUME

    # ===========================================================
    # STEP 6b: Voxelwise LME — intermediate model without run effect
    # (Major Concern 2: isolate hierarchical modeling from run effect)
    # ===========================================================
    echo ""
    echo "-----------------------------------------------------------"
    echo "STEP 6b: Voxelwise LME — intercept-only (Major Concern 2)"
    echo "  Model: beta ~ 1 + (1|subject) — no run fixed effect"
    echo "  This isolates the effect of hierarchical variance"
    echo "  decomposition from the run covariate."
    echo "-----------------------------------------------------------"
    python main_lme_voxelwise_v2.py --chunk-size 10000 --n-jobs 8 --no-run-effect $RESUME

    # ===========================================================
    # STEP 7: Equalized Voxelwise Comparison (MC1 + MC2)
    # Now includes intermediate LME-no-run comparison
    # ===========================================================
    echo ""
    echo "-----------------------------------------------------------"
    echo "STEP 7: GLM vs LME Comparison (equalized thresholds)"
    echo "  Includes: GLM vs LME-full vs LME-no-run"
    echo "-----------------------------------------------------------"
    python compare_methods.py
fi

# ===========================================================
# STEP 8: Generate Paper Figures
# ===========================================================
echo ""
echo "-----------------------------------------------------------"
echo "STEP 8: Generate Publication Figures"
echo "-----------------------------------------------------------"
python generate_paper_figures_v2.py

# ===========================================================
# STEP 9: Sensitivity & Diagnostic Analyses (MC2, MC4, mn3, mn5)
# ===========================================================
echo ""
echo "-----------------------------------------------------------"
echo "STEP 9: Sensitivity Analyses (NEW)"
echo "  MC2: FD as run-level covariate"
echo "  MC4: Residual autocorrelation (ACF)"
echo "  mn3: Bootstrap CIs for ICC"
echo "  mn5: FD distribution reporting"
echo "-----------------------------------------------------------"
python analysis_sensitivity.py --n-bootstrap 2000

# ===========================================================
# STEP 10: Extended Simulations (FPR, ICC bootstrap, random slope)
# ===========================================================
echo ""
echo "-----------------------------------------------------------"
echo "STEP 10: Simulation Analyses (FPR, ICC, Random Slope)"
echo "-----------------------------------------------------------"
python analysis_simulations.py --n-sims 1000 --n-bootstrap 500

# ===========================================================
# DONE!
# ===========================================================
echo ""
echo "=========================================================="
echo "ANALYSIS COMPLETE!"
echo "=========================================================="
echo ""
echo "Results:    /data/hcp_emotion/derivatives/"
echo "Figures:    /data/hcp_emotion/figures/"
echo "Validation: /data/hcp_emotion/derivatives/group_level/validation/"
echo ""
echo "Key files for paper:"
echo "  - figures/figure1_brain_maps.pdf     (now with ICC map)"
echo "  - figures/figure2_roi_comparison.pdf"
echo "  - figures/figure3_stability.pdf"
echo "  - figures/figure4_computational.pdf   (3 panels)"
echo "  - figures/simulation_results.pdf       (FPR, ICC, random slope)"
echo "  - figures/mc4_residual_acf.pdf         (NEW: residual ACF — MC4)"
echo "  - figures/comparison_stats.csv        (equalized thresholds)"
echo "  - figures/summary_table.tex"
echo "  - group_level/validation/statsmodels_vs_lme4_comparison.csv  (NEW: MC1)"
echo "  - group_level/validation/parameter_recovery.csv"
echo "  - group_level/sensitivity/mc2_fd_sensitivity.csv     (NEW: MC2)"
echo "  - group_level/sensitivity/mc4_residual_acf.csv       (NEW: MC4)"
echo "  - group_level/sensitivity/mn3_icc_bootstrap_ci.csv   (NEW: mn3)"
echo "  - group_level/sensitivity/fd_distribution_summary.csv (NEW: mn5)"
echo "  - group_level/simulations/fpr_results.csv"
echo "  - group_level/simulations/icc_bootstrap_results.csv"
echo "  - group_level/simulations/random_slope_results.csv"
echo ""
echo "Unthresholded maps for supplementary / NeuroVault:"
echo "  - group_level/glm_zmap.nii.gz"
echo "  - lme_voxelwise/lme_zmap.nii.gz"
echo "  - lme_voxelwise/lme_beta.nii.gz"
echo "  - lme_voxelwise/lme_icc.nii.gz"
echo "  - lme_voxelwise/lme_norun_zmap.nii.gz  (intermediate model — MC2)"
echo ""
