import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.masking import compute_brain_mask
from nilearn.image import threshold_img
from scipy import stats

from config import SMOOTH_FWHM, Z_THRESHOLD, CONTRAST_NAME
from paths import FIRST_LEVEL, GROUP_LEVEL, LOGS, get_subject_ids


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"group_glm_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    subject_ids = get_subject_ids()
    
    beta_imgs = []
    valid_subjects = []
    
    for subj in subject_ids:
        beta_path = FIRST_LEVEL / subj / f"{CONTRAST_NAME}.nii.gz"
        if beta_path.exists():
            beta_imgs.append(str(beta_path))
            valid_subjects.append(subj)
    
    logger.info(f"Found {len(beta_imgs)} subjects with beta maps")
    
    if len(beta_imgs) < 10:
        logger.error("Too few subjects!")
        return
    
    design_matrix = pd.DataFrame({
        "intercept": np.ones(len(beta_imgs))
    })
    
    logger.info("Fitting second-level GLM...")
    
    model = SecondLevelModel(smoothing_fwhm=SMOOTH_FWHM)
    model.fit(beta_imgs, design_matrix=design_matrix)
    
    z_map = model.compute_contrast("intercept", output_type="z_score")
    effect_map = model.compute_contrast("intercept", output_type="effect_size")
    
    # Save maps
    z_map.to_filename(GROUP_LEVEL / "glm_zmap.nii.gz")
    effect_map.to_filename(GROUP_LEVEL / "glm_effect.nii.gz")
    
    # Thresholded map
    z_thresh = threshold_img(z_map, threshold=Z_THRESHOLD)
    z_thresh.to_filename(GROUP_LEVEL / f"glm_zmap_thresh{Z_THRESHOLD}.nii.gz")
    
    # Stats summary
    z_data = z_map.get_fdata()
    mask = ~np.isnan(z_data) & (z_data != 0)
    
    logger.info("\n" + "="*50)
    logger.info("GROUP GLM RESULTS")
    logger.info(f"Subjects: {len(beta_imgs)}")
    logger.info(f"Max Z: {np.nanmax(z_data):.2f}")
    logger.info(f"Min Z: {np.nanmin(z_data):.2f}")
    logger.info(f"Mean Z: {np.nanmean(z_data[mask]):.2f}")
    logger.info(f"Voxels > Z={Z_THRESHOLD}: {(z_data > Z_THRESHOLD).sum():,}")
    logger.info("="*50)
    
    # Save summary
    summary = {
        'n_subjects': len(beta_imgs),
        'max_z': float(np.nanmax(z_data)),
        'min_z': float(np.nanmin(z_data)),
        'mean_z': float(np.nanmean(z_data[mask])),
        'n_sig_uncorr': int((z_data > Z_THRESHOLD).sum())
    }
    
    pd.Series(summary).to_csv(GROUP_LEVEL / "glm_summary.csv")
    
    logger.info(f"Results saved to {GROUP_LEVEL}")


if __name__ == "__main__":
    main()
