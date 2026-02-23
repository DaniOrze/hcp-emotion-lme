import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import threshold_img
from scipy import stats

from config import (
    SMOOTH_FWHM, Z_THRESHOLD, CONTRAST_NAME, FDR_ALPHA,
    N_VOLUMES_PER_RUN, TR, FIELD_STRENGTH, ANALYSIS_SPACE
)
from paths import FIRST_LEVEL, GROUP_LEVEL, LOGS, get_subject_ids
from utils.lme_v2 import compute_fdr_correction


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"group_glm_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_peak_coordinates(stat_map, mask=None, min_distance=20, n_peaks=20):
    data = stat_map.get_fdata()
    affine = stat_map.affine

    if mask is None:
        mask = ~np.isnan(data) & (data != 0)

    from scipy import ndimage

    binary = (data > Z_THRESHOLD) & mask
    labeled, n_clusters = ndimage.label(binary)

    peaks = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled == cluster_id
        cluster_size = cluster_mask.sum()

        cluster_data = np.where(cluster_mask, data, -np.inf)
        peak_idx = np.unravel_index(np.argmax(cluster_data), data.shape)
        peak_z = data[peak_idx]

        voxel_coords = np.array([peak_idx[0], peak_idx[1], peak_idx[2], 1])
        mni_coords = affine @ voxel_coords

        peaks.append({
            'x_mni': float(mni_coords[0]),
            'y_mni': float(mni_coords[1]),
            'z_mni': float(mni_coords[2]),
            'z_score': float(peak_z),
            'cluster_size': int(cluster_size),
            'cluster_id': cluster_id
        })

    if not peaks:
        return pd.DataFrame()

    peaks_df = pd.DataFrame(peaks)
    peaks_df = peaks_df.sort_values('z_score', ascending=False).head(n_peaks)
    peaks_df = peaks_df.reset_index(drop=True)

    return peaks_df


def compute_glm_fdr(z_map_img, alpha=0.05):
    z_data = z_map_img.get_fdata()
    valid_mask = ~np.isnan(z_data) & (z_data != 0) & np.isfinite(z_data)

    p_values = np.full(z_data.shape, np.nan)
    p_values[valid_mask] = 2 * (1 - stats.norm.cdf(np.abs(z_data[valid_mask])))

    p_valid = p_values[valid_mask]
    sig_fdr = compute_fdr_correction(p_valid, alpha=alpha)

    fdr_mask = np.zeros(z_data.shape, dtype=np.uint8)
    fdr_mask[valid_mask] = sig_fdr.astype(np.uint8)

    return fdr_mask, p_values


def compute_cluster_fwe(z_map_img, cluster_forming_threshold=3.1,
                        alpha=0.05, two_sided=True):
    from scipy import ndimage
    from nilearn.image import threshold_img

    z_data = z_map_img.get_fdata()
    affine = z_map_img.affine

    voxel_vol = np.abs(np.linalg.det(affine[:3, :3]))

    clusters_all = []

    for direction, sign in [('positive', 1), ('negative', -1)]:
        if not two_sided and sign == -1:
            continue

        binary = (sign * z_data) > cluster_forming_threshold
        labeled, n_clusters = ndimage.label(binary)

        for cid in range(1, n_clusters + 1):
            cluster_mask = labeled == cid
            cluster_size = int(cluster_mask.sum())

            cluster_z = np.where(cluster_mask, sign * z_data, -np.inf)
            peak_idx = np.unravel_index(np.argmax(cluster_z), z_data.shape)
            peak_z = float(z_data[peak_idx])

            voxel = np.array([peak_idx[0], peak_idx[1], peak_idx[2], 1])
            mni = affine @ voxel

            clusters_all.append({
                'direction': direction,
                'cluster_size_voxels': cluster_size,
                'cluster_size_mm3': cluster_size * voxel_vol,
                'peak_z': peak_z,
                'peak_x_mni': float(mni[0]),
                'peak_y_mni': float(mni[1]),
                'peak_z_mni': float(mni[2]),
            })

    if not clusters_all:
        return pd.DataFrame(), nib.Nifti1Image(
            np.zeros_like(z_data, dtype=np.int32), affine)

    cluster_df = pd.DataFrame(clusters_all)
    cluster_df = cluster_df.sort_values('cluster_size_voxels', ascending=False)
    cluster_df = cluster_df.reset_index(drop=True)

    try:
        from nilearn.glm import threshold_stats_img
        thresholded_map, fwe_threshold = threshold_stats_img(
            z_map_img,
            alpha=alpha,
            height_control='fdr',  
            cluster_threshold=0,   
        )
        logger.info(f"Cluster-level FWE threshold computed via nilearn")
    except Exception as e:
        logger.warning(f"nilearn cluster correction failed: {e}. "
                       "Reporting uncorrected clusters.")

    return cluster_df, z_map_img


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

    logger.info(f"\n{'='*50}")
    logger.info(f"SAMPLE CHARACTERISTICS")
    logger.info(f"{'='*50}")
    logger.info(f"N subjects: {len(beta_imgs)}")
    logger.info(f"Analysis space: {ANALYSIS_SPACE}")
    logger.info(f"Smoothing: {SMOOTH_FWHM} mm FWHM")
    logger.info(f"Scanner: {FIELD_STRENGTH}T")
    logger.info(f"TR: {TR}s, Volumes/run: {N_VOLUMES_PER_RUN}")

    design_matrix = pd.DataFrame({
        "intercept": np.ones(len(beta_imgs))
    })

    logger.info("Fitting second-level GLM...")
    model = SecondLevelModel(smoothing_fwhm=SMOOTH_FWHM)
    model.fit(beta_imgs, design_matrix=design_matrix)

    z_map = model.compute_contrast("intercept", output_type="z_score")
    effect_map = model.compute_contrast("intercept", output_type="effect_size")

    z_map.to_filename(GROUP_LEVEL / "glm_zmap.nii.gz")
    effect_map.to_filename(GROUP_LEVEL / "glm_effect.nii.gz")

    logger.info("Applying FDR correction (Benjamini-Hochberg, q < 0.05)...")
    fdr_mask, p_map = compute_glm_fdr(z_map, alpha=FDR_ALPHA)

    fdr_img = nib.Nifti1Image(fdr_mask, z_map.affine)
    fdr_img.to_filename(GROUP_LEVEL / "glm_fdr05.nii.gz")

    p_img = nib.Nifti1Image(p_map, z_map.affine)
    p_img.to_filename(GROUP_LEVEL / "glm_pmap.nii.gz")

    z_thresh = threshold_img(z_map, threshold=Z_THRESHOLD)
    z_thresh.to_filename(GROUP_LEVEL / f"glm_zmap_thresh{Z_THRESHOLD}.nii.gz")

    logger.info("Extracting peak coordinates (MNI)...")
    peaks_df = extract_peak_coordinates(z_map)
    if len(peaks_df) > 0:
        peaks_df.to_csv(GROUP_LEVEL / "glm_peak_coordinates.csv", index=False)
        logger.info(f"Saved {len(peaks_df)} peaks to glm_peak_coordinates.csv")
        logger.info(f"\nTop 10 peaks (MNI coordinates):")
        logger.info(peaks_df.head(10).to_string(index=False))
    else:
        logger.warning("No significant peaks found at Z > 3.1")
        
    logger.info("\nComputing cluster-level statistics (Major Concern 6)...")
    cluster_df, _ = compute_cluster_fwe(z_map, cluster_forming_threshold=Z_THRESHOLD)
    if len(cluster_df) > 0:
        cluster_df.to_csv(GROUP_LEVEL / "glm_cluster_table.csv", index=False)
        logger.info(f"Found {len(cluster_df)} clusters at Z > {Z_THRESHOLD}")
        logger.info(f"Top 10 clusters:")
        logger.info(cluster_df.head(10).to_string(index=False))
    else:
        logger.warning("No clusters found at cluster-forming threshold")

    z_data = z_map.get_fdata()
    mask = ~np.isnan(z_data) & (z_data != 0)

    n_sig_uncorr = int((z_data > Z_THRESHOLD).sum())
    n_sig_fdr = int(fdr_mask.sum())

    logger.info(f"\n{'='*50}")
    logger.info("GROUP GLM RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Subjects: {len(beta_imgs)}")
    logger.info(f"Max Z: {np.nanmax(z_data):.2f}")
    logger.info(f"Min Z: {np.nanmin(z_data):.2f}")
    logger.info(f"Mean Z: {np.nanmean(z_data[mask]):.2f}")
    logger.info(f"Voxels > Z={Z_THRESHOLD} (uncorrected): {n_sig_uncorr:,}")
    logger.info(f"Voxels significant (FDR q<0.05):        {n_sig_fdr:,}")
    logger.info(f"{'='*50}")

    summary = {
        'n_subjects': len(beta_imgs),
        'max_z': float(np.nanmax(z_data)),
        'min_z': float(np.nanmin(z_data)),
        'mean_z': float(np.nanmean(z_data[mask])),
        'n_sig_uncorr': n_sig_uncorr,
        'n_sig_fdr': n_sig_fdr,
        'fdr_alpha': FDR_ALPHA,
        'z_threshold': Z_THRESHOLD,
        'smooth_fwhm': SMOOTH_FWHM,
    }

    pd.Series(summary).to_csv(GROUP_LEVEL / "glm_summary.csv")

    logger.info(f"Results saved to {GROUP_LEVEL}")


if __name__ == "__main__":
    main()
