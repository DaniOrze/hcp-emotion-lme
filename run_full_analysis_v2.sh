#!/bin/bash
# ==============================================
# HCP Emotion Task - FULL ANALYSIS PIPELINE
# ==============================================
#
# Complete pipeline including gold-standard analyses
# for publication-ready results.
#
# Estimated time:
#   - ROI + Stability + Benchmark: ~30 min
#   - Full voxelwise LME: 3-6 hours
#
# Usage:
#   ./run_full_analysis_v2.sh           # Full pipeline
#   ./run_full_analysis_v2.sh --quick   # ROI only (fast)
#   ./run_full_analysis_v2.sh --resume  # Resume voxelwise
#

set -e

QUICK_MODE=false
RESUME=""

for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --resume) RESUME="--resume" ;;
    esac
done

echo "=================================================="
echo "HCP EMOTION TASK - GLM vs LME ANALYSIS"
echo "Complete Pipeline for Publication"
echo "=================================================="
echo ""

# Check data
if [ ! -d "/data/hcp_emotion/raw" ]; then
    echo "ERROR: Data not found at /data/hcp_emotion/raw"
    exit 1
fi

N_SUBJECTS=$(ls -d /data/hcp_emotion/raw/*/ 2>/dev/null | wc -l)
echo "📊 Found $N_SUBJECTS subjects"
echo ""

# ===========================================
# STEP 1: First-level GLM
# ===========================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: First-level GLM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main_first_level.py --n-jobs 6

# ===========================================
# STEP 2: Group-level GLM
# ===========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Group-level GLM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main_group_glm.py

# ===========================================
# STEP 3: ROI Analysis
# ===========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: ROI Analysis (GLM vs LME)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main_roi_analysis_v2.py

# ===========================================
# STEP 4: Stability Analysis
# ===========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Stability Analysis (Bootstrap, Split-Half)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python analysis_stability_v2.py --n-bootstrap 1000

# ===========================================
# STEP 5: Computational Benchmark
# ===========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Computational Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python analysis_benchmark.py

if [ "$QUICK_MODE" = true ]; then
    echo ""
    echo "⚡ Quick mode: Skipping voxelwise LME"
    echo ""
else
    # ===========================================
    # STEP 6: Voxelwise LME
    # ===========================================
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STEP 6: Voxelwise LME (this will take hours!)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python main_lme_voxelwise_v2.py --chunk-size 5000 --n-jobs 4 $RESUME

    # ===========================================
    # STEP 7: Voxelwise Comparison
    # ===========================================
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STEP 7: GLM vs LME Comparison"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python compare_methods.py
fi

# ===========================================
# STEP 8: Generate Paper Figures
# ===========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 8: Generate Publication Figures"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python generate_paper_figures.py

# ===========================================
# DONE!
# ===========================================
echo ""
echo "=================================================="
echo "✅ ANALYSIS COMPLETE!"
echo "=================================================="
echo ""
echo "📁 Results: /data/hcp_emotion/derivatives/"
echo "📊 Figures: /data/hcp_emotion/figures/"
echo ""
echo "Key files for paper:"
echo "  - figures/figure1_brain_maps.pdf"
echo "  - figures/figure2_roi_comparison.pdf"
echo "  - figures/figure3_stability.pdf"
echo "  - figures/figure4_computational.pdf"
echo "  - figures/summary_table.tex"
echo ""
