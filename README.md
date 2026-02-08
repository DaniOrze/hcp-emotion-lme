# Beyond the GLM: Linear Mixed-Effects Modeling of fMRI Data

**Comprehensive comparison of GLM vs LME approaches for analyzing hierarchical fMRI data from the Human Connectome Project.**

📄 **Paper:** Orzechowski, D. et al. (2026). *Beyond the General Linear Model: Linear Mixed-Effects Modeling of fMRI Data from the Human Connectome Project.* Human Brain Mapping.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Key Findings

- ✅ **100% convergence** across 215,992 brain voxels
- 📊 **High concordance** between methods (r = 0.77)
- 🧠 **Substantial within-subject variability** (mean ICC = 0.17)
- ⚡ **20× computational overhead** for LME (7.7 hours for whole-brain)
- 🔄 **Excellent reliability** (split-half r > 0.98 for both methods)

---

## Quick Start
```bash
# Clone repository
git clone https://github.com/your-username/hcp-emotion-lme.git
cd hcp-emotion-lme

# Install dependencies
pip install -r requirements.txt

# Configure paths
nano paths.py  # Edit to point to your HCP data

# Run analysis
python main_first_level.py      
python main_group_glm.py         
python main_roi_analysis.py     
python main_lme_voxelwise.py    
```

---

## Requirements

- **Python 3.10+**
- **HCP data access** ([register here](https://db.humanconnectome.org))
- **Hardware:** 12+ CPU cores, 64 GB RAM (for voxelwise LME)

### Main dependencies
```bash
nibabel>=5.0.0
nilearn>=0.10.0
statsmodels>=0.14.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
```

---

## Repository Structure
```
├── main_first_level.py          # First-level GLM (per-subject, per-run)
├── main_group_glm.py            # Second-level GLM (group averaging)
├── main_roi_analysis.py         # ROI-based GLM vs LME comparison
├── main_lme_voxelwise.py        # Voxelwise LME (whole-brain)
├── analysis_stability_v2.py     # Bootstrap & split-half reliability
├── analysis_benchmark.py        # Computational benchmarks
├── generate_paper_figures_v2.py # Publication figures
├── config.py                    # Configuration parameters
├── paths.py                     # Directory paths
└── utils/
    ├── io.py                    # Data loading
    └── lme_v2.py                # LME fitting functions
```

---

## Data Setup

1. **Download HCP data:** S1200 Release → Task fMRI 3T (Recommended)
2. **Required files per subject:**
   - `tfMRI_EMOTION_LR_hp0_clean_rclean_tclean.dtseries.nii`
   - `tfMRI_EMOTION_RL_hp0_clean_rclean_tclean.dtseries.nii`
   - `EVs/` (fear.txt, neut.txt, sync.txt)
   - `Movement_Regressors.txt`

3. **Edit `paths.py`:**
```python
BASE_DIR = Path("/path/to/your/data")
RAW_DATA = BASE_DIR / "raw"
```

---

## Output Files
```
derivatives/
├── group_level/
│   ├── glm_zmap.nii.gz              # GLM z-scores
│   ├── roi_results_v2.csv           # ROI comparison
│   └── stability_results_v2.json    # Reliability metrics
├── lme_voxelwise/
│   ├── lme_zmap.nii.gz              # LME z-scores
│   ├── lme_icc.nii.gz               # ICC map
│   └── lme_fdr05.nii.gz             # Thresholded map
└── figures/
    ├── figure1_brain_maps.png       # Activation maps
    ├── figure2_roi_comparison.png   # ROI results
    ├── figure3_stability.png        # Reliability analysis
    └── figure4_computational.png    # Benchmarks
```

---

## Citation
```bibtex
@article{orzechowski2026lme,
  title={Beyond the General Linear Model: Linear Mixed-Effects Modeling of fMRI Data},
  author={Orzechowski, Daniele and [Co-authors]},
  journal={Human Brain Mapping},
  year={2026},
  doi={10.XXXX/journal.XXXXX}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

- **Email:** daniorzechowski@gmail.com

**Acknowledgments:** Data provided by the Human Connectome Project, WU-Minn Consortium (NIH 1U54MH091657).
