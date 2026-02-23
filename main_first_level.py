import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from joblib import Parallel, delayed

from config import (
    TR, HRF_MODEL, HIGH_PASS, NOISE_MODEL, N_JOBS, CONTRAST_NAME,
    DRIFT_MODEL, FD_THRESHOLD, FD_PERCENT_THRESHOLD,
    FIELD_STRENGTH, TE, VOXEL_SIZE, SEQUENCE, MULTIBAND_FACTOR,
    FLIP_ANGLE, N_SLICES, N_VOLUMES_PER_RUN, ANALYSIS_SPACE,
    SMOOTH_FWHM
)
from paths import RAW_DATA, FIRST_LEVEL, LOGS, get_subject_ids
from utils.io import load_hcp_emotion_events, load_confounds
from utils.motion_qc import compute_framewise_displacement, check_motion_exclusion


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"first_level_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RUNS = ["LR", "RL"]


def process_subject(subj_id: str, force: bool = False) -> dict:
    subj_out = FIRST_LEVEL / subj_id
    
    all_exist = all(
        (subj_out / f"{CONTRAST_NAME}_{run}.nii.gz").exists() 
        for run in RUNS
    ) and (subj_out / f"{CONTRAST_NAME}.nii.gz").exists()
    
    if all_exist and not force:
        logger.info(f"  {subj_id} already processed, skipping")
        return {"subject": subj_id, "status": "skipped", "runs_processed": 0}
    
    logger.info(f"  Processing {subj_id}")
    
    subj_dir = RAW_DATA / subj_id / "MNINonLinear" / "Results"
    subj_out.mkdir(parents=True, exist_ok=True)
    
    runs_processed = 0
    run_betas = {}
    motion_stats = {}
    
    for run in RUNS:
        try:
            run_dir = subj_dir / f"tfMRI_EMOTION_{run}"
            beta_path = subj_out / f"{CONTRAST_NAME}_{run}.nii.gz"
            
            nii_files = list(run_dir.glob("tfMRI_EMOTION_*clean*.nii.gz"))
            if len(nii_files) == 0:
                logger.warning(f"No fMRI file for {subj_id} {run}")
                continue
            
            fmri_path = nii_files[0]
            
            ev_dir = run_dir / "EVs"
            events_df = load_hcp_emotion_events(ev_dir)
            
            conf_path = run_dir / "Movement_Regressors.txt"
            confounds_df = load_confounds(conf_path) if conf_path.exists() else None
            
            if confounds_df is not None:
                fd_values = compute_framewise_displacement(confounds_df)
                exclude, fd_stats = check_motion_exclusion(
                    fd_values, 
                    fd_threshold=FD_THRESHOLD,
                    fd_percent_threshold=FD_PERCENT_THRESHOLD
                )
                motion_stats[run] = fd_stats
                
                if exclude:
                    logger.warning(
                        f"  {subj_id} {run} EXCLUDED: "
                        f"mean FD={fd_stats['mean_fd']:.3f}mm, "
                        f"{fd_stats['percent_above']:.1f}% frames > {FD_THRESHOLD}mm"
                    )
                    continue
                else:
                    logger.info(
                        f"  {subj_id} {run} motion OK: "
                        f"mean FD={fd_stats['mean_fd']:.3f}mm"
                    )
            
            model = FirstLevelModel(
                t_r=TR,
                hrf_model=HRF_MODEL,
                drift_model=DRIFT_MODEL,
                high_pass=HIGH_PASS,
                noise_model=NOISE_MODEL,
                standardize=False,
                minimize_memory=True
            )
            
            model.fit(fmri_path, events=events_df, confounds=confounds_df)
            
            fear_beta = model.compute_contrast("fear", output_type="effect_size")
            fear_beta.to_filename(subj_out / f"fear_{run}.nii.gz")
            
            neut_beta = model.compute_contrast("neut", output_type="effect_size")
            neut_beta.to_filename(subj_out / f"neut_{run}.nii.gz")
            
            contrast_beta = model.compute_contrast("fear - neut", output_type="effect_size")
            contrast_beta.to_filename(beta_path)
            
            run_betas[run] = contrast_beta
            
            z_map = model.compute_contrast("fear - neut", output_type="z_score")
            z_map.to_filename(subj_out / f"{CONTRAST_NAME}_{run}_zmap.nii.gz")
            
            runs_processed += 1
            logger.info(f"  {subj_id} {run} done")
            
        except Exception as e:
            logger.error(f"  {subj_id} {run} failed: {e}")
    
    if len(run_betas) >= 1:
        from nilearn.image import mean_img
        
        if len(run_betas) == 2:
            avg_beta = mean_img(list(run_betas.values()))
        else:
            avg_beta = list(run_betas.values())[0]
        
        avg_beta.to_filename(subj_out / f"{CONTRAST_NAME}.nii.gz")
        logger.info(f"  Saved average beta from {len(run_betas)} runs")
    
    if motion_stats:
        pd.DataFrame(motion_stats).T.to_csv(subj_out / "motion_qc.csv")
    
    status = "success" if runs_processed == len(RUNS) else f"partial ({runs_processed}/{len(RUNS)})"
    logger.info(f"  {subj_id} completed: {runs_processed} runs")
    
    return {"subject": subj_id, "status": status, "runs_processed": runs_processed}


def main():
    parser = argparse.ArgumentParser(description="First-level GLM for HCP Emotion")
    parser.add_argument("--subjects", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    args = parser.parse_args()
    
    subjects = get_subject_ids()
    
    if args.subjects:
        subjects = subjects[:args.subjects]
    
    logger.info(f"Processing {len(subjects)} subjects with {args.n_jobs} jobs")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ACQUISITION & PREPROCESSING PARAMETERS")
    logger.info(f"{'='*60}")
    logger.info(f"Scanner: {FIELD_STRENGTH}T {SEQUENCE}")
    logger.info(f"TR={TR}s, TE={TE}ms, Flip={FLIP_ANGLE} deg")
    logger.info(f"Voxel: {VOXEL_SIZE[0]}x{VOXEL_SIZE[1]}x{VOXEL_SIZE[2]} mm")
    logger.info(f"Slices: {N_SLICES}, Multiband: {MULTIBAND_FACTOR}")
    logger.info(f"Volumes/run: {N_VOLUMES_PER_RUN}")
    logger.info(f"Space: {ANALYSIS_SPACE}")
    logger.info(f"HRF: {HRF_MODEL}, Noise: {NOISE_MODEL}, Drift: {DRIFT_MODEL}")
    logger.info(f"High-pass: {HIGH_PASS:.4f} Hz ({1/HIGH_PASS:.0f}s cutoff)")
    logger.info(f"Motion QC: FD>{FD_THRESHOLD}mm, exclude if >{FD_PERCENT_THRESHOLD*100:.0f}% frames")
    logger.info(f"{'='*60}\n")
    
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_subject)(subj, force=args.force)
        for subj in subjects
    )
    
    df = pd.DataFrame(results)
    success = (df["status"] == "success").sum()
    partial = df["status"].str.startswith("partial").sum()
    skipped = (df["status"] == "skipped").sum()
    
    total_runs = df["runs_processed"].sum()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY: {success} success, {partial} partial, {skipped} skipped")
    logger.info(f"Total runs processed: {total_runs}")
    logger.info(f"{'='*50}")
    
    df.to_csv(FIRST_LEVEL / "processing_log.csv", index=False)


if __name__ == "__main__":
    main()