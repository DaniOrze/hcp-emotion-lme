import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

from config import CONTRAST_NAME
from paths import FIRST_LEVEL, GROUP_LEVEL, FIGURES, LOGS, get_subject_ids


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"roi_analysis_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_atlases():
    logger.info("Loading Harvard-Oxford atlases...")
    
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    
    return ho_cort, ho_sub


def get_emotion_rois():
    cortical_rois = {
        'Frontal_Pole': 1,
        'Insular_Cortex': 2,
        'Superior_Frontal_Gyrus': 3,
        'Middle_Frontal_Gyrus': 4,
        'Inferior_Frontal_Gyrus_pars_triangularis': 5,
        'Inferior_Frontal_Gyrus_pars_opercularis': 6,
        'Frontal_Medial_Cortex': 13,
        'Paracingulate_Gyrus': 16,
        'Cingulate_Gyrus_anterior': 17,
        'Frontal_Orbital_Cortex': 20,
    }
    
    subcortical_rois = {
        'Left_Cerebral_Cortex': 1,
        'Left_Thalamus': 4,
        'Left_Caudate': 5,
        'Left_Putamen': 6,
        'Left_Pallidum': 7,
        'Brainstem': 8,
        'Left_Hippocampus': 9,
        'Left_Amygdala': 10,
        'Right_Thalamus': 15,
        'Right_Hippocampus': 19,
        'Right_Amygdala': 20,
    }
    
    return cortical_rois, subcortical_rois


def extract_roi_betas(subject_ids, contrast_name, atlas_img, roi_dict, atlas_name):
    rows = []
    atlas_data = atlas_img.get_fdata()
    
    logger.info(f"Processing {atlas_name} ROIs...")
    logger.info(f"Atlas shape: {atlas_data.shape}")
    logger.info(f"Atlas unique values: {len(np.unique(atlas_data))}")
    
    runs = ['LR', 'RL']
    
    for subj in subject_ids:
        for run in runs:
            beta_path = FIRST_LEVEL / subj / f"{contrast_name}_{run}.nii.gz"
            
            if not beta_path.exists():
                continue
            
            try:
                beta_img = nib.load(str(beta_path))
                beta_data = beta_img.get_fdata()
                
                if beta_data.shape != atlas_data.shape:
                    logger.warning(f"Shape mismatch for {subj} {run}")
                    continue
                
                for roi_name, roi_idx in roi_dict.items():
                    roi_mask = atlas_data == roi_idx
                    n_voxels = roi_mask.sum()
                    
                    if n_voxels > 0:
                        mean_beta = beta_data[roi_mask].mean()
                        rows.append({
                            'subject': subj,
                            'run': run,
                            'roi': f"{atlas_name}_{roi_name}",
                            'beta': float(mean_beta),
                            'n_voxels': int(n_voxels)
                        })
            except Exception as e:
                logger.warning(f"Error processing {subj} {run}: {e}")
    
    return pd.DataFrame(rows)


def run_roi_glm(df, roi_name):
    roi_data = df[df['roi'] == roi_name]
    
    if len(roi_data) == 0:
        return None
    
    subject_means = roi_data.groupby('subject')['beta'].mean().values
    
    if len(subject_means) < 3:
        return None
    
    t_stat, p_val = stats.ttest_1samp(subject_means, 0)
    total_var = roi_data['beta'].var()
    cohens_d = float(np.mean(subject_means) / np.sqrt(total_var)) if total_var > 0 else 0.0
    
    return {
        'roi': roi_name,
        'method': 'GLM',
        'mean_beta': float(np.mean(subject_means)),
        'std_beta': float(np.std(subject_means)),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'n_subjects': len(subject_means),
        'cohens_d': cohens_d,
        'converged': True
    }


def run_roi_lme_ultra_robust(df, roi_name, max_attempts=5):
    roi_data = df[df['roi'] == roi_name].copy()
    
    if len(roi_data) == 0:
        logger.warning(f"No data for {roi_name}")
        return None
    
    if len(roi_data) < 3:
        logger.warning(f"Insufficient data for {roi_name}: n={len(roi_data)}")
        return None
    
    roi_data['run'] = pd.Categorical(roi_data['run'])
    
    n_runs = roi_data['run'].nunique()
    use_run_effect = n_runs >= 2
    
    betas = roi_data['beta'].values
    between_subj_means = roi_data.groupby('subject')['beta'].mean()
    between_subj_var = between_subj_means.var()
    total_var = betas.var()
    
    if between_subj_var < 1e-10 or total_var < 1e-10:
        logger.warning(f"Near-zero variance for {roi_name}, LME may be unstable")
    
    methods_to_try = [
        ('lbfgs', {}),
        ('bfgs', {}),
        ('powell', {}),
        ('lbfgs', {'maxiter': 500}),
        ('cg', {})
    ]
    
    best_result = None
    best_score = -np.inf
    
    formula = "beta ~ 1 + run" if use_run_effect else "beta ~ 1"
    
    for attempt, (method, fit_kwargs) in enumerate(methods_to_try):
        if attempt >= max_attempts:
            break
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                model = MixedLM.from_formula(
                    formula,
                    groups="subject",
                    data=roi_data
                )
                
                result = model.fit(reml=True, method=method, **fit_kwargs)
                
                if not result.converged:
                    logger.debug(f"LME did not converge for {roi_name} with method {method}")
                    continue
                
                try:
                    mean_beta = float(result.fe_params['Intercept'])
                    std_beta = float(result.bse['Intercept'])
                    t_val = float(result.tvalues['Intercept'])
                    p_val = float(result.pvalues['Intercept'])
                except (KeyError, IndexError, ValueError) as e:
                    logger.debug(f"Error extracting values for {roi_name}: {e}")
                    continue
                
                if not all(np.isfinite([mean_beta, std_beta, t_val, p_val])):
                    logger.debug(f"Non-finite values for {roi_name}")
                    continue
                
                if abs(t_val) > 500:
                    logger.debug(f"Suspiciously large t-value for {roi_name}: {t_val}")
                    continue
                
                if p_val == 0.0:
                    p_val = np.finfo(float).tiny
                
                try:
                    var_subject = float(result.cov_re.iloc[0, 0])
                    var_residual = float(result.scale)
                except Exception as e:
                    logger.debug(f"Error computing variance for {roi_name}: {e}")
                    continue
                
                if var_subject < 0:
                    logger.debug(f"Negative between-subject variance for {roi_name}")
                    continue
                
                if var_residual <= 0:
                    logger.debug(f"Non-positive residual variance for {roi_name}")
                    continue
                
                if var_subject > 10000 or var_residual > 10000:
                    logger.debug(f"Unreasonably large variance for {roi_name}")
                    continue
                
                total_variance = var_subject + var_residual
                if total_variance <= 0:
                    continue
                
                icc = var_subject / total_variance
                
                if not (0 <= icc <= 1):
                    logger.debug(f"Invalid ICC for {roi_name}: {icc}")
                    continue
                
                cohens_d = mean_beta / np.sqrt(total_variance)
                
                if not np.isfinite(cohens_d):
                    continue
                
                if abs(cohens_d) > 10:
                    logger.debug(f"Unreasonably large effect size for {roi_name}: {cohens_d}")
                    continue
                
                run_effect = None
                run_effect_p = None
                if use_run_effect:
                    try:
                        run_keys = [k for k in result.fe_params.index if k.startswith('run')]
                        if run_keys:
                            run_key = run_keys[0]
                            run_effect = float(result.fe_params[run_key])
                            run_effect_p = float(result.pvalues[run_key])
                    except Exception:
                        pass
                
                try:
                    score = float(result.llf)
                except:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        'roi': roi_name,
                        'method': 'LME',
                        'mean_beta': mean_beta,
                        'std_beta': std_beta,
                        't_stat': t_val,
                        'p_value': p_val,
                        'n_subjects': roi_data['subject'].nunique(),
                        'cohens_d': cohens_d,
                        'var_subject': var_subject,
                        'var_residual': var_residual,
                        'icc': icc,
                        'run_effect': run_effect,
                        'run_effect_p': run_effect_p,
                        'converged': True,
                        'method_used': method,
                        'log_likelihood': score
                    }
                    
        except Exception as e:
            logger.debug(f"LME attempt {attempt+1} failed for {roi_name} with {method}: {e}")
            continue
    
    if best_result is None:
        logger.warning(f"All {max_attempts} LME attempts failed for {roi_name}")
        return None
    
    logger.debug(f"LME succeeded for {roi_name} using method: {best_result['method_used']}")
    return best_result


def apply_fdr_correction(results_df):
    from statsmodels.stats.multitest import multipletests
    
    for method in ['GLM', 'LME']:
        mask = results_df['method'] == method
        p_vals = results_df.loc[mask, 'p_value'].values
        
        if len(p_vals) > 0:
            _, p_fdr, _, _ = multipletests(p_vals, method='fdr_bh')
            results_df.loc[mask, 'p_fdr'] = p_fdr
    
    return results_df


def create_roi_figure(results_df, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    results_df['roi_short'] = results_df['roi'].str.replace('cortical_', '').str.replace('subcortical_', '')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    pivot = results_df.pivot(index='roi_short', columns='method', values='cohens_d')
    pivot = pivot.sort_values('GLM')
    pivot.plot(kind='barh', ax=ax, alpha=0.8)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Effect Sizes by ROI", fontsize=14, fontweight='bold')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(title='Method', frameon=True)
    ax.grid(axis='x', alpha=0.3)
    
    ax = axes[0, 1]
    for method, marker, color in [('GLM', 'o', 'steelblue'), ('LME', 's', 'darkorange')]:
        method_data = results_df[results_df['method'] == method].copy()
        method_data['p_value_clipped'] = method_data['p_value'].clip(lower=1e-300)
        method_data['neg_log_p'] = -np.log10(method_data['p_value_clipped'])
        
        ax.scatter(method_data['roi_short'], method_data['neg_log_p'], 
                  label=method, marker=marker, s=100, alpha=0.7, color=color)
    
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05', linewidth=1.5)
    ax.axhline(-np.log10(0.01), color='darkred', linestyle='--', label='p=0.01', linewidth=1.5)
    ax.set_ylabel("-log10(p-value)", fontsize=12)
    ax.set_title("Statistical Significance by ROI", fontsize=14, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax = axes[1, 0]
    lme_results = results_df[results_df['method'] == 'LME'].copy()
    if 'icc' in lme_results.columns and len(lme_results) > 0:
        lme_results = lme_results.sort_values('icc', ascending=True)
        colors = ['darkgreen' if x > 0.75 else 'orange' if x > 0.5 else 'indianred' for x in lme_results['icc']]
        ax.barh(lme_results['roi_short'], lme_results['icc'], color=colors, alpha=0.7)
        ax.set_xlabel("Intraclass Correlation (ICC)", fontsize=12)
        ax.set_title("Between-Subject Reliability (LME)", fontsize=14, fontweight='bold')
        ax.axvline(0.5, color='orange', linestyle='--', linewidth=1.5, label='Moderate (0.5)', alpha=0.7)
        ax.axvline(0.75, color='darkgreen', linestyle='--', linewidth=1.5, label='Good (0.75)', alpha=0.7)
        ax.legend(frameon=True)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
    
    ax = axes[1, 1]
    glm = results_df[results_df['method'] == 'GLM'].set_index('roi_short')
    lme = results_df[results_df['method'] == 'LME'].set_index('roi_short')
    common_rois = glm.index.intersection(lme.index)
    
    if len(common_rois) > 0:
        ax.scatter(glm.loc[common_rois, 'cohens_d'], 
                  lme.loc[common_rois, 'cohens_d'], 
                  s=120, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
        
        for roi in common_rois:
            ax.annotate(roi, 
                       (glm.loc[roi, 'cohens_d'], lme.loc[roi, 'cohens_d']), 
                       fontsize=8, alpha=0.8, xytext=(3, 3), textcoords='offset points')
        
        all_values = list(glm.loc[common_rois, 'cohens_d']) + list(lme.loc[common_rois, 'cohens_d'])
        lims = [min(all_values) - 0.1, max(all_values) + 0.1]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Identity', linewidth=2)
        
        from scipy.stats import pearsonr, spearmanr
        r_pearson, p_pearson = pearsonr(glm.loc[common_rois, 'cohens_d'], 
                                        lme.loc[common_rois, 'cohens_d'])
        r_spearman, p_spearman = spearmanr(glm.loc[common_rois, 'cohens_d'], 
                                           lme.loc[common_rois, 'cohens_d'])
        
        textstr = f'Pearson r = {r_pearson:.3f} (p = {p_pearson:.3e})\nSpearman ρ = {r_spearman:.3f}'
        ax.text(0.05, 0.95, textstr, 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10)
        
        ax.set_xlabel("GLM Cohen's d", fontsize=12)
        ax.set_ylabel("LME Cohen's d", fontsize=12)
        ax.set_title("Method Comparison", fontsize=14, fontweight='bold')
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roi_comparison_v2.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'roi_comparison_v2.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figure saved to {output_dir / 'roi_comparison_v2.png'}")


def main():
    logger.info("="*50)
    logger.info("ROI ANALYSIS: GLM vs LME (ULTRA-ROBUST)")
    logger.info("="*50)
    
    ho_cort, ho_sub = load_atlases()
    
    cortical_rois, subcortical_rois = get_emotion_rois()
    
    # Get subjects
    subject_ids = get_subject_ids()
    logger.info(f"Found {len(subject_ids)} subjects")
    
    logger.info("Extracting ROI beta values...")
    
    cortical_df = extract_roi_betas(
        subject_ids, 
        CONTRAST_NAME,
        ho_cort['maps'],
        cortical_rois,
        'cortical'
    )
    
    subcortical_df = extract_roi_betas(
        subject_ids,
        CONTRAST_NAME,
        ho_sub['maps'],
        subcortical_rois,
        'subcortical'
    )
    
    beta_df = pd.concat([cortical_df, subcortical_df], ignore_index=True)
    
    logger.info(f"Extracted {len(beta_df)} ROI observations from {beta_df['roi'].nunique()} ROIs")
    logger.info(f"ROIs found: {sorted(beta_df['roi'].unique())}")
    
    if len(beta_df) == 0:
        logger.error("No ROI data extracted! Check that:")
        logger.error("1. Beta maps exist in FIRST_LEVEL directory")
        logger.error("2. Beta maps have the same resolution as the atlas (2mm)")
        logger.error("3. ROI indices are correct for the Harvard-Oxford atlas")
        return
    
    results = []
    unique_rois = sorted(beta_df['roi'].unique())
    
    glm_failures = []
    lme_failures = []
    
    for roi in unique_rois:
        n_subjects = len(beta_df[beta_df['roi'] == roi])
        logger.info(f"Analyzing {roi} (n={n_subjects})...")
        
        # GLM
        glm_result = run_roi_glm(beta_df, roi)
        if glm_result:
            results.append(glm_result)
        else:
            glm_failures.append(roi)
        
        # LME (ultra-robust)
        lme_result = run_roi_lme_ultra_robust(beta_df, roi, max_attempts=5)
        if lme_result:
            results.append(lme_result)
        else:
            lme_failures.append(roi)
            logger.warning(f"LME failed for {roi} - using GLM result only")
    
    if len(results) == 0:
        logger.error("No results generated!")
        return
    
    results_df = pd.DataFrame(results)
    
    # Report failures
    if glm_failures:
        logger.warning(f"\nGLM failed for {len(glm_failures)} ROIs: {glm_failures}")
    if lme_failures:
        logger.warning(f"\nLME failed for {len(lme_failures)} ROIs: {lme_failures}")
    
    results_df = apply_fdr_correction(results_df)
    
    results_df.to_csv(GROUP_LEVEL / 'roi_results_v2.csv', index=False)
    beta_df.to_csv(GROUP_LEVEL / 'roi_betas_v2.csv', index=False)
    
    create_roi_figure(results_df, FIGURES)
    
    logger.info("\n" + "="*50)
    logger.info("ROI ANALYSIS SUMMARY")
    logger.info("="*50)
    
    print("\nGLM Results:")
    glm_cols = ['roi', 'mean_beta', 't_stat', 'p_value', 'p_fdr', 'cohens_d']
    glm_results = results_df[results_df['method'] == 'GLM'][glm_cols]
    print(glm_results.to_string(index=False))
    
    print("\nLME Results:")
    lme_cols = ['roi', 'mean_beta', 't_stat', 'p_value', 'p_fdr', 'cohens_d', 'icc', 'run_effect', 'run_effect_p']
    lme_results = results_df[results_df['method'] == 'LME']
    if len(lme_results) > 0:
        print(lme_results[[c for c in lme_cols if c in lme_results.columns]].to_string(index=False))
    
    if 'run_effect' in results_df.columns:
        lme_with_run = results_df[(results_df['method'] == 'LME') & (results_df['run_effect'].notna())]
        if len(lme_with_run) > 0:
            n_run_sig = (lme_with_run['run_effect_p'] < 0.05).sum()
            print(f"\n  Run effect (LR vs RL):")
            print(f"    ROIs with significant run effect (p<0.05): {n_run_sig}/{len(lme_with_run)}")
            print(f"    Mean |run effect|: {lme_with_run['run_effect'].abs().mean():.6f}")
            print(f"    Max |run effect|:  {lme_with_run['run_effect'].abs().max():.6f}")
    
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    for method in ['GLM', 'LME']:
        method_data = results_df[results_df['method'] == method]
        if len(method_data) > 0:
            n_total = len(unique_rois)
            n_analyzed = len(method_data)
            n_sig_p = (method_data['p_value'] < 0.05).sum()
            n_sig_fdr = (method_data['p_fdr'] < 0.05).sum()
            mean_effect = method_data['cohens_d'].abs().mean()
            
            print(f"\n{method}:")
            print(f"  ROIs analyzed: {n_analyzed}/{n_total}")
            print(f"  Significant (p < 0.05): {n_sig_p}/{n_analyzed}")
            print(f"  Significant (FDR < 0.05): {n_sig_fdr}/{n_analyzed}")
            print(f"  Mean |effect size|: {mean_effect:.3f}")
            
            if method == 'LME' and 'icc' in method_data.columns:
                mean_icc = method_data['icc'].mean()
                print(f"  Mean ICC: {mean_icc:.3f}")
    
    if len(results_df[results_df['method'] == 'GLM']) > 0 and len(results_df[results_df['method'] == 'LME']) > 0:
        glm_sig = set(results_df[(results_df['method'] == 'GLM') & (results_df['p_fdr'] < 0.05)]['roi'])
        lme_sig = set(results_df[(results_df['method'] == 'LME') & (results_df['p_fdr'] < 0.05)]['roi'])
        
        agreement = len(glm_sig & lme_sig)
        glm_only = len(glm_sig - lme_sig)
        lme_only = len(lme_sig - glm_sig)
        
        print(f"\nAgreement (FDR < 0.05):")
        print(f"  Both methods: {agreement}")
        print(f"  GLM only: {glm_only}")
        print(f"  LME only: {lme_only}")
    
    logger.info(f"\nResults saved to {GROUP_LEVEL / 'roi_results_v2.csv'}")


if __name__ == "__main__":
    main()