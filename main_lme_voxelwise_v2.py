import argparse
import logging
from datetime import datetime
from pathlib import Path
import pickle

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.masking import apply_mask
from joblib import Parallel, delayed
from tqdm import tqdm

from config import (
    LME_N_JOBS, CHUNK_SIZE, LME_METHOD, LME_REML, LME_MAX_ITER
)
from paths import FIRST_LEVEL, LME_VOXELWISE, LOGS, get_subject_ids
from utils.lme_v2 import fit_lme_chunk, compute_fdr_correction


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"lme_voxelwise_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_all_runs_betas(subject_ids, contrast_name="fear_minus_neut"):
    beta_imgs = []
    run_info = []
    
    runs = ['LR', 'RL']  # HCP emotion task runs
    
    for subj in subject_ids:
        for run in runs:
            # Try both possible naming conventions
            beta_path = FIRST_LEVEL / subj / f"{contrast_name}_{run}.nii.gz"
            
            if not beta_path.exists():
                beta_path = FIRST_LEVEL / subj / f"{contrast_name}.nii.gz"
                if not beta_path.exists():
                    continue
            
            try:
                img = nib.load(str(beta_path))
                beta_imgs.append(img)
                run_info.append({
                    'subject': subj,
                    'run': run,
                    'observation_id': f"{subj}_{run}"
                })
            except Exception as e:
                logger.warning(f"Failed to load {subj} {run}: {e}")
    
    logger.info(f"Loaded {len(beta_imgs)} runs from {len(subject_ids)} subjects")
    
    # Check if we have multiple runs per subject
    subjects_with_data = [r['subject'] for r in run_info]
    unique_subjects = set(subjects_with_data)
    
    logger.info(f"Data from {len(unique_subjects)} unique subjects")
    logger.info(f"Mean runs per subject: {len(run_info) / len(unique_subjects):.2f}")
    
    return beta_imgs, run_info


def create_group_mask(beta_imgs, threshold=0.5):
    logger.info("Computing group mask...")
    
    # Get individual masks
    masks = []
    for img in beta_imgs:
        data = img.get_fdata()
        mask = ~np.isnan(data) & (data != 0) & np.isfinite(data)
        masks.append(mask)
    
    mask_stack = np.stack(masks, axis=-1)
    overlap = mask_stack.sum(axis=-1) / len(masks)
    
    group_mask = overlap >= threshold
    
    from scipy import ndimage
    group_mask = ndimage.binary_opening(group_mask, iterations=1)
    
    logger.info(f"Group mask: {group_mask.sum():,} voxels")
    
    mask_img = nib.Nifti1Image(group_mask.astype(np.uint8), beta_imgs[0].affine)
    return mask_img


def extract_beta_matrix(beta_imgs, mask_img):
    logger.info("Extracting beta matrix...")
    
    beta_matrix = []
    for img in tqdm(beta_imgs, desc="Loading betas"):
        masked = apply_mask(img, mask_img)
        beta_matrix.append(masked)
    
    return np.vstack(beta_matrix)


def run_lme_voxelwise_chunked(beta_matrix, run_info_df, chunk_size, n_jobs,
                               checkpoint_path=None, resume=False):
    n_voxels = beta_matrix.shape[1]
    n_chunks = int(np.ceil(n_voxels / chunk_size))
    
    logger.info(f"Running LME on {n_voxels:,} voxels in {n_chunks} chunks")
    logger.info(f"Data shape: {beta_matrix.shape[0]} runs × {n_voxels:,} voxels")
    
    n_subjects = run_info_df['subject'].nunique()
    n_runs = len(run_info_df)
    logger.info(f"Subjects: {n_subjects}, Total runs: {n_runs}, Ratio: {n_runs/n_subjects:.2f}")
    
    if n_runs / n_subjects < 1.5:
        logger.error("⚠️  Need multiple runs per subject for LME!")
        logger.error("Check if you're loading run-level data correctly")
        raise ValueError("Insufficient within-subject replication")
    
    processed_voxels = set()
    all_results = []
    
    if resume and checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            all_results = checkpoint['results']
            processed_voxels = set(checkpoint['processed_voxels'])
            logger.info(f"Resumed from checkpoint: {len(processed_voxels):,} voxels done")
    
    for chunk_idx in tqdm(range(n_chunks), desc="LME chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_voxels)
        
        chunk_voxels = set(range(start_idx, end_idx))
        if chunk_voxels.issubset(processed_voxels):
            continue
        
        beta_chunk = beta_matrix[:, start_idx:end_idx]
        voxel_indices = list(range(start_idx, end_idx))
        
        n_voxels_chunk = beta_chunk.shape[1]
        voxels_per_job = max(1, n_voxels_chunk // n_jobs)
        
        def process_subchunk(job_idx):
            sub_start = job_idx * voxels_per_job
            sub_end = min(sub_start + voxels_per_job, n_voxels_chunk)
            
            if sub_start >= n_voxels_chunk:
                return []
            
            return fit_lme_chunk(
                beta_chunk[:, sub_start:sub_end],
                run_info_df,
                voxel_indices[sub_start:sub_end],
                method=LME_METHOD,
                reml=LME_REML,
                max_iter=LME_MAX_ITER
            )
        
        # Run parallel
        chunk_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_subchunk)(i) for i in range(n_jobs + 1)
        )
        
        for sublist in chunk_results:
            all_results.extend(sublist)
        
        processed_voxels.update(chunk_voxels)
        
        # Checkpoint every 5 chunks
        if checkpoint_path and (chunk_idx + 1) % 5 == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'results': all_results,
                    'processed_voxels': list(processed_voxels)
                }, f)
            logger.info(f"Checkpoint saved: {len(processed_voxels):,} voxels")
    
    return pd.DataFrame(all_results)


def create_result_maps(results_df, mask_img, output_dir):
    logger.info("Creating result maps...")
    
    mask_data = mask_img.get_fdata().astype(bool)
    n_voxels_mask = mask_data.sum()
    affine = mask_img.affine
    
    n_total = len(results_df)
    n_converged = results_df['converged'].sum()
    
    valid_z = np.isfinite(results_df['z_score'])
    valid_p = np.isfinite(results_df['p_value'])
    valid_se = np.isfinite(results_df['std_err'])
    valid_beta = np.isfinite(results_df['fixed_effect'])
    valid_icc = np.isfinite(results_df['icc'])
    
    logger.info(f"\n{'='*50}")
    logger.info(f"LME CONVERGENCE DIAGNOSTICS")
    logger.info(f"{'='*50}")
    logger.info(f"Total voxels:        {n_total:>10,}")
    logger.info(f"Converged:           {n_converged:>10,} ({100*n_converged/n_total:>5.1f}%)")
    logger.info(f"Valid fixed effect:  {valid_beta.sum():>10,} ({100*valid_beta.sum()/n_total:>5.1f}%)")
    logger.info(f"Valid SE:            {valid_se.sum():>10,} ({100*valid_se.sum()/n_total:>5.1f}%)")
    logger.info(f"Valid Z-scores:      {valid_z.sum():>10,} ({100*valid_z.sum()/n_total:>5.1f}%)")
    logger.info(f"Valid p-values:      {valid_p.sum():>10,} ({100*valid_p.sum()/n_total:>5.1f}%)")
    logger.info(f"Valid ICC:           {valid_icc.sum():>10,} ({100*valid_icc.sum()/n_total:>5.1f}%)")
    
    fully_valid = (
        results_df['converged'] & 
        valid_z & valid_p & valid_se & valid_beta
    )
    valid_results = results_df[fully_valid].copy()
    
    n_valid = len(valid_results)
    logger.info(f"\nFully valid voxels:  {n_valid:>10,} ({100*n_valid/n_total:>5.1f}%)")
    
    # Warning if too many failures
    if n_valid < n_total * 0.5:
        logger.error("⚠️  WARNING: Less than 50% valid voxels - results may be unreliable!")
    elif n_valid < n_total * 0.8:
        logger.warning("⚠️  Low validity rate - check data quality")
    
    # Diagnostics on valid voxels
    if n_valid > 0:
        z_values = valid_results['z_score'].values
        icc_values = valid_results['icc'].values
        
        logger.info(f"\nZ-score distribution (valid voxels):")
        logger.info(f"  Mean:   {np.mean(z_values):>8.2f}")
        logger.info(f"  Median: {np.median(z_values):>8.2f}")
        logger.info(f"  SD:     {np.std(z_values):>8.2f}")
        logger.info(f"  Min:    {np.min(z_values):>8.2f}")
        logger.info(f"  Max:    {np.max(z_values):>8.2f}")
        
        logger.info(f"\nICC distribution (valid voxels):")
        logger.info(f"  Mean:   {np.mean(icc_values):>8.3f}")
        logger.info(f"  Median: {np.median(icc_values):>8.3f}")
        logger.info(f"  Range:  [{np.min(icc_values):.3f}, {np.max(icc_values):.3f}]")
        
        # Sanity checks
        if abs(np.mean(z_values)) > 5:
            logger.warning(f"⚠️  Unusual mean Z-score: {np.mean(z_values):.2f}")
        if np.median(icc_values) < 0.1:
            logger.warning(f"⚠️  Very low ICC - little within-subject correlation")
    
    logger.info(f"{'='*50}\n")
    
    z_map = np.full(mask_data.shape, np.nan)
    p_map = np.full(mask_data.shape, np.nan)
    beta_map = np.full(mask_data.shape, np.nan)
    icc_map = np.full(mask_data.shape, np.nan)
    
    voxel_coords = np.argwhere(mask_data)
    
    for _, row in valid_results.iterrows():
        idx = int(row['voxel_idx'])
        if idx < len(voxel_coords):
            x, y, z = voxel_coords[idx]
            z_map[x, y, z] = row['z_score']
            p_map[x, y, z] = row['p_value']
            beta_map[x, y, z] = row['fixed_effect']
            icc_map[x, y, z] = row['icc']
    
    nib.save(nib.Nifti1Image(z_map, affine), output_dir / "lme_zmap.nii.gz")
    nib.save(nib.Nifti1Image(p_map, affine), output_dir / "lme_pmap.nii.gz")
    nib.save(nib.Nifti1Image(beta_map, affine), output_dir / "lme_beta.nii.gz")
    nib.save(nib.Nifti1Image(icc_map, affine), output_dir / "lme_icc.nii.gz")
    
    logger.info("Saved: lme_zmap.nii.gz, lme_pmap.nii.gz, lme_beta.nii.gz, lme_icc.nii.gz")
    
    p_values = p_map[mask_data]
    sig_fdr = compute_fdr_correction(p_values, alpha=0.05)
    
    fdr_map = np.zeros(mask_data.shape, dtype=np.uint8)
    fdr_map[mask_data] = sig_fdr.astype(np.uint8)
    nib.save(nib.Nifti1Image(fdr_map, affine), output_dir / "lme_fdr05.nii.gz")
    
    logger.info("Saved: lme_fdr05.nii.gz")
    
    # Summary stats
    n_sig = sig_fdr.sum()
    sig_percent = 100 * n_sig / n_valid if n_valid > 0 else 0
    
    logger.info(f"\nSignificant voxels (FDR q<0.05): {n_sig:,} ({sig_percent:.1f}% of valid)")
    
    return {
        'n_voxels_total': n_total,
        'n_voxels_valid': n_valid,
        'n_sig_fdr': n_sig,
        'sig_percent': sig_percent,
        'mean_z': np.nanmean(z_map[mask_data]) if n_valid > 0 else np.nan,
        'max_z': np.nanmax(z_map[mask_data]) if n_valid > 0 else np.nan,
        'min_z': np.nanmin(z_map[mask_data]) if n_valid > 0 else np.nan,
        'mean_icc': np.nanmean(icc_map[mask_data]) if n_valid > 0 else np.nan,
        'convergence_rate': 100 * n_converged / n_total
    }


def main():
    parser = argparse.ArgumentParser(description="Voxelwise LME for HCP Emotion")
    parser.add_argument("--chunk-size", type=int, default=2000,
                       help="Voxels per chunk (lower = less memory)")
    parser.add_argument("--n-jobs", type=int, default=LME_N_JOBS)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    LME_VOXELWISE.mkdir(parents=True, exist_ok=True)
    
    subject_ids = get_subject_ids()
    beta_imgs, run_info = load_all_runs_betas(subject_ids)
    
    if len(beta_imgs) < 20:
        logger.error("Too few runs for LME analysis!")
        return
    
    run_info_df = pd.DataFrame(run_info)
    
    mask_img = create_group_mask(beta_imgs, threshold=0.5)
    mask_path = LME_VOXELWISE / "group_mask.nii.gz"
    nib.save(mask_img, str(mask_path))
    
    beta_matrix = extract_beta_matrix(beta_imgs, mask_img)
    logger.info(f"Beta matrix shape: {beta_matrix.shape}")
    logger.info(f"  Runs: {beta_matrix.shape[0]}")
    logger.info(f"  Voxels: {beta_matrix.shape[1]:,}")
    
    checkpoint_path = LME_VOXELWISE / "checkpoint.pkl"
    
    results_df = run_lme_voxelwise_chunked(
        beta_matrix,
        run_info_df,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
        checkpoint_path=checkpoint_path,
        resume=args.resume
    )
    
    results_df.to_csv(LME_VOXELWISE / "lme_results.csv", index=False)
    logger.info(f"Saved results CSV with {len(results_df):,} rows")
    
    summary = create_result_maps(results_df, mask_img, LME_VOXELWISE)
    
    logger.info("\n" + "="*50)
    logger.info("LME ANALYSIS COMPLETE")
    logger.info(f"Total voxels:        {summary['n_voxels_total']:>10,}")
    logger.info(f"Valid voxels:        {summary['n_voxels_valid']:>10,} ({summary['convergence_rate']:.1f}%)")
    logger.info(f"Significant (FDR):   {summary['n_sig_fdr']:>10,} ({summary['sig_percent']:.1f}%)")
    logger.info(f"Z-score range:       [{summary['min_z']:>6.2f}, {summary['max_z']:>6.2f}]")
    logger.info(f"Mean Z-score:        {summary['mean_z']:>10.2f}")
    logger.info(f"Mean ICC:            {summary['mean_icc']:>10.3f}")
    logger.info("="*50)


if __name__ == "__main__":
    main()