#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from config import CONTRAST_NAME, N_JOBS
from paths import FIRST_LEVEL, GROUP_LEVEL, FIGURES, LOGS, get_subject_ids


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"stability_v2_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


LME_FORMULA = "beta ~ 1 + run"


def load_roi_betas():
    beta_path = GROUP_LEVEL / 'roi_betas_v2.csv'
    if not beta_path.exists():
        beta_path = GROUP_LEVEL / 'roi_betas.csv'
    
    if beta_path.exists():
        df = pd.read_csv(beta_path)
        if 'run' not in df.columns:
            raise ValueError("roi_betas CSV must have a 'run' column. Re-run main_roi_analysis_v2.py first.")
        logger.info(f"Loaded betas from {beta_path}")
        return df
    else:
        raise FileNotFoundError("Run main_roi_analysis_v2.py first!")


def _fit_lme_safe(df, formula=LME_FORMULA, method="powell", maxiter=100):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            use_formula = formula
            if df['run'].nunique() < 2:
                use_formula = "beta ~ 1"
            
            df = df.copy()
            df['run'] = pd.Categorical(df['run'])
            
            result = MixedLM.from_formula(
                use_formula, groups="subject", data=df
            ).fit(reml=True, method=method, maxiter=maxiter)
            
            return result.fe_params['Intercept'], result.tvalues['Intercept']
    except:
        return np.nan, np.nan


def bootstrap_glm(betas, n_bootstrap=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    
    boot_means = []
    boot_t = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(betas, size=len(betas), replace=True)
        boot_means.append(np.mean(sample))
        t, _ = stats.ttest_1samp(sample, 0)
        boot_t.append(t)
    
    return {
        'mean': np.mean(betas),
        'boot_mean': np.mean(boot_means),
        'ci_lower': np.percentile(boot_means, 2.5),
        'ci_upper': np.percentile(boot_means, 97.5),
        'boot_se': np.std(boot_means),
        't_mean': np.mean(boot_t),
        't_ci_lower': np.percentile(boot_t, 2.5),
        't_ci_upper': np.percentile(boot_t, 97.5),
    }


def bootstrap_lme(df, n_bootstrap=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    subjects = df['subject'].unique()
    
    boot_intercepts = []
    boot_t = []
    
    for _ in range(n_bootstrap):
        boot_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        
        frames = []
        for i, s in enumerate(boot_subjects):
            subj_data = df[df['subject'] == s].copy()
            subj_data['subject_boot'] = f"s{i}"
            frames.append(subj_data)
        
        boot_df = pd.concat(frames, ignore_index=True)
        boot_df['run'] = pd.Categorical(boot_df['run'])
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                use_formula = LME_FORMULA.replace("subject", "subject_boot")
                if boot_df['run'].nunique() < 2:
                    model = MixedLM.from_formula(
                        "beta ~ 1", groups="subject_boot", data=boot_df
                    )
                else:
                    model = MixedLM.from_formula(
                        "beta ~ 1 + run", groups="subject_boot", data=boot_df
                    )
                
                result = model.fit(reml=True, method="powell", maxiter=100)
                boot_intercepts.append(result.fe_params['Intercept'])
                boot_t.append(result.tvalues['Intercept'])
        except:
            continue
    
    if len(boot_intercepts) < n_bootstrap * 0.5:
        logger.warning(f"Only {len(boot_intercepts)} successful bootstrap iterations")
    
    if len(boot_intercepts) == 0:
        return {
            'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'boot_se': np.nan, 't_mean': np.nan, 
            't_ci_lower': np.nan, 't_ci_upper': np.nan,
        }
    
    return {
        'mean': np.mean(boot_intercepts),
        'ci_lower': np.percentile(boot_intercepts, 2.5),
        'ci_upper': np.percentile(boot_intercepts, 97.5),
        'boot_se': np.std(boot_intercepts),
        't_mean': np.mean(boot_t),
        't_ci_lower': np.percentile(boot_t, 2.5),
        't_ci_upper': np.percentile(boot_t, 97.5),
    }


def split_half_reliability(df, n_splits=100, random_state=42):
    rng = np.random.RandomState(random_state)
    subjects = df['subject'].unique()
    n_half = len(subjects) // 2
    rois = df['roi'].unique()
    
    correlations_glm = []
    correlations_lme = []
    
    for _ in range(n_splits):
        rng.shuffle(subjects)
        half1 = subjects[:n_half]
        half2 = subjects[n_half:2*n_half]
        
        df1 = df[df['subject'].isin(half1)]
        df2 = df[df['subject'].isin(half2)]
        
        effects1_glm, effects2_glm = [], []
        effects1_lme, effects2_lme = [], []
        
        for roi in rois:
            roi_df1 = df1[df1['roi'] == roi]
            roi_df2 = df2[df2['roi'] == roi]
            
            effects1_glm.append(roi_df1.groupby('subject')['beta'].mean().mean())
            effects2_glm.append(roi_df2.groupby('subject')['beta'].mean().mean())
            
            intercept1, _ = _fit_lme_safe(roi_df1)
            intercept2, _ = _fit_lme_safe(roi_df2)
            effects1_lme.append(intercept1)
            effects2_lme.append(intercept2)
        
        r_glm, _ = stats.pearsonr(effects1_glm, effects2_glm)
        correlations_glm.append(r_glm)
        
        e1 = np.array(effects1_lme)
        e2 = np.array(effects2_lme)
        valid = np.isfinite(e1) & np.isfinite(e2)
        if valid.sum() > 2:
            r_lme, _ = stats.pearsonr(e1[valid], e2[valid])
            correlations_lme.append(r_lme)
    
    def spearman_brown(r):
        return 2 * r / (1 + r)
    
    return {
        'glm': {
            'r_half': np.mean(correlations_glm),
            'r_full': spearman_brown(np.mean(correlations_glm)),
            'r_std': np.std(correlations_glm)
        },
        'lme': {
            'r_half': np.mean(correlations_lme) if correlations_lme else np.nan,
            'r_full': spearman_brown(np.mean(correlations_lme)) if correlations_lme else np.nan,
            'r_std': np.std(correlations_lme) if correlations_lme else np.nan
        }
    }

def leave_one_out_sensitivity(df, roi_name):
    roi_df = df[df['roi'] == roi_name]
    subjects = roi_df['subject'].unique()
    
    loo_glm = []
    loo_lme = []
    
    for leave_out in subjects:
        subset = roi_df[roi_df['subject'] != leave_out]
        
        subj_means = subset.groupby('subject')['beta'].mean().values
        t, _ = stats.ttest_1samp(subj_means, 0)
        loo_glm.append(t)
        
        _, t_lme = _fit_lme_safe(subset)
        loo_lme.append(t_lme)
    
    loo_lme_arr = np.array(loo_lme, dtype=float)
    
    return {
        'glm': {
            'mean': np.mean(loo_glm),
            'std': np.std(loo_glm),
            'range': np.ptp(loo_glm),
            'cv': np.std(loo_glm) / np.abs(np.mean(loo_glm)) if np.mean(loo_glm) != 0 else np.nan
        },
        'lme': {
            'mean': np.nanmean(loo_lme_arr),
            'std': np.nanstd(loo_lme_arr),
            'range': np.nanmax(loo_lme_arr) - np.nanmin(loo_lme_arr) if np.any(np.isfinite(loo_lme_arr)) else np.nan,
            'cv': np.nanstd(loo_lme_arr) / np.abs(np.nanmean(loo_lme_arr)) if np.nanmean(loo_lme_arr) != 0 else np.nan
        }
    }


def create_stability_figure(bootstrap_results, split_half, loo_results, output_dir):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    rois = list(bootstrap_results['glm'].keys())
    x = np.arange(len(rois))
    width = 0.35
    
    roi_labels = [r.replace('cortical_', 'c:').replace('subcortical_', 's:') for r in rois]
    
    glm_means = [bootstrap_results['glm'][r]['mean'] for r in rois]
    glm_ci_low = [bootstrap_results['glm'][r]['ci_lower'] for r in rois]
    glm_ci_high = [bootstrap_results['glm'][r]['ci_upper'] for r in rois]
    
    lme_means = [bootstrap_results['lme'][r]['mean'] for r in rois]
    lme_ci_low = [bootstrap_results['lme'][r]['ci_lower'] for r in rois]
    lme_ci_high = [bootstrap_results['lme'][r]['ci_upper'] for r in rois]
    
    ax.bar(x - width/2, glm_means, width, label='GLM', 
           yerr=[np.array(glm_means)-np.array(glm_ci_low), 
                 np.array(glm_ci_high)-np.array(glm_means)], capsize=3)
    ax.bar(x + width/2, lme_means, width, label='LME',
           yerr=[np.array(lme_means)-np.array(lme_ci_low),
                 np.array(lme_ci_high)-np.array(lme_means)], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Effect (β)')
    ax.set_title('Bootstrap 95% CIs')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--')
    
    ax = axes[0, 1]
    methods = ['GLM', 'LME']
    r_halves = [split_half['glm']['r_half'], split_half['lme']['r_half']]
    r_fulls = [split_half['glm']['r_full'], split_half['lme']['r_full']]
    
    x = np.arange(len(methods))
    ax.bar(x - width/2, r_halves, width, label='r (half)')
    ax.bar(x + width/2, r_fulls, width, label='r (Spearman-Brown)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Correlation')
    ax.set_title('Split-Half Reliability')
    ax.legend()
    ax.set_ylim(0, 1)
    
    ax = axes[1, 0]
    rois_loo = list(loo_results.keys())
    roi_labels_loo = [r.replace('cortical_', 'c:').replace('subcortical_', 's:') for r in rois_loo]
    glm_cv = [loo_results[r]['glm']['cv'] for r in rois_loo]
    lme_cv = [loo_results[r]['lme']['cv'] for r in rois_loo]
    
    x = np.arange(len(rois_loo))
    ax.bar(x - width/2, glm_cv, width, label='GLM')
    ax.bar(x + width/2, lme_cv, width, label='LME')
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels_loo, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('CV (lower = more stable)')
    ax.set_title('Leave-One-Out Stability')
    ax.legend()
    
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = """
    STABILITY ANALYSIS SUMMARY (v2)
    LME model: beta ~ 1 + run
    
    Split-Half Reliability (Spearman-Brown):
      GLM: r = {:.3f}
      LME: r = {:.3f}
    
    Interpretation:
      r > 0.9: Excellent
      r > 0.8: Good
      r > 0.7: Acceptable
      r < 0.7: Poor
    
    Leave-One-Out CV (mean across ROIs):
      GLM: {:.4f}
      LME: {:.4f}
      
    Lower CV = more robust to individual subjects
    """.format(
        split_half['glm']['r_full'],
        split_half['lme']['r_full'],
        np.nanmean(glm_cv),
        np.nanmean(lme_cv)
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_analysis_v2.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'stability_analysis_v2.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figure saved to {output_dir / 'stability_analysis_v2.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--n-splits', type=int, default=100)
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("STABILITY ANALYSIS (v2 — beta ~ 1 + run)")
    logger.info("="*50)
    
    df = load_roi_betas()
    rois = df['roi'].unique()
    logger.info(f"Loaded {len(df)} observations, {len(rois)} ROIs")
    logger.info(f"Runs per subject: {df.groupby('subject')['run'].nunique().mean():.1f}")
    
    logger.info(f"\nRunning bootstrap ({args.n_bootstrap} iterations)...")
    bootstrap_results = {'glm': {}, 'lme': {}}
    
    for roi in tqdm(rois, desc="Bootstrap"):
        roi_df = df[df['roi'] == roi]
        
        roi_subj_means = roi_df.groupby('subject')['beta'].mean().values
        bootstrap_results['glm'][roi] = bootstrap_glm(roi_subj_means, args.n_bootstrap)
        
        bootstrap_results['lme'][roi] = bootstrap_lme(roi_df, args.n_bootstrap)
    
    logger.info(f"\nRunning split-half reliability ({args.n_splits} splits)...")
    split_half = split_half_reliability(df, args.n_splits)
    
    logger.info("\nRunning leave-one-out analysis...")
    loo_results = {}
    for roi in tqdm(rois, desc="LOO"):
        loo_results[roi] = leave_one_out_sensitivity(df, roi)
    
    create_stability_figure(bootstrap_results, split_half, loo_results, FIGURES)
    
    results = {
        'bootstrap': bootstrap_results,
        'split_half': split_half,
        'loo': loo_results
    }
    
    import json
    with open(GROUP_LEVEL / 'stability_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info("\n" + "="*50)
    logger.info("STABILITY SUMMARY (v2)")
    logger.info("="*50)
    logger.info(f"\nSplit-Half Reliability (Spearman-Brown corrected):")
    logger.info(f"  GLM: r = {split_half['glm']['r_full']:.3f}")
    logger.info(f"  LME: r = {split_half['lme']['r_full']:.3f}")
    
    logger.info(f"\nMean LOO Coefficient of Variation:")
    glm_cv = np.nanmean([loo_results[r]['glm']['cv'] for r in rois])
    lme_cv = np.nanmean([loo_results[r]['lme']['cv'] for r in rois])
    logger.info(f"  GLM: CV = {glm_cv:.4f}")
    logger.info(f"  LME: CV = {lme_cv:.4f}")


if __name__ == "__main__":
    main()