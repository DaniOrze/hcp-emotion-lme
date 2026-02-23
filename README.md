# Beyond the General Linear Model: Linear Mixed-Effects Modeling of fMRI Data from the Human Connectome Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-available-blue.svg)](Dockerfile)

Code and analysis pipeline for comparing General Linear Model (GLM) and Linear Mixed-Effects (LME) approaches to fMRI group analysis, using emotion task data from 143 Human Connectome Project subjects (286 runs).

**Manuscript:** Submitted to *Neuroinformatics* (Springer Nature).

## Key Findings

- GLM and LME show high voxelwise agreement (*r* = 0.95), but LME is more conservative (Dice = 0.74 under FDR *q* < 0.05)
- The sensitivity difference is driven by the **run covariate**, not hierarchical variance decomposition (intermediate model: *r* = 0.995 with GLM)
- Whole-brain ICC mapping reveals spatially heterogeneous within-subject variability (mean ICC = 0.22)
- Run effects are pervasive (15/21 ROIs) and persist after controlling for head motion
- LME is ~17× slower but achieves 99.6% convergence

## Requirements

- Python 3.10
- nilearn 0.10.0
- statsmodels 0.14.0
- scipy 1.11.0
- nibabel 5.1.0

## Quick Start

### Using Docker (recommended)

```bash
docker build -t hcp-emotion-lme .
docker run -v /path/to/hcp/data:/data hcp-emotion-lme
```

### Manual Installation

```bash
pip install -r requirements.txt
```

## Data

This project uses data from the [Human Connectome Project](https://db.humanconnectome.org) S1200 release. You must obtain access to HCP data independently through their data use agreement.

## Citation

If you use this code, please cite:

```
Orzechowski, D., & Martins da Costa, R. Beyond the General Linear Model:
Linear Mixed-Effects Modeling of fMRI Data from the Human Connectome Project.
Submitted to Neuroinformatics.
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Daniele Orzechowski — daniorzechowski@gmail.com