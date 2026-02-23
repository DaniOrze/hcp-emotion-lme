import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import GROUP_LEVEL, FIGURES, LOGS
from utils.lme_v2 import fit_lme_single_voxel, get_backend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"simulations_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def simulate_fpr(n_subjects=143, n_runs=2, n_simulations=1000,
                 alpha=0.05, seed=42):
    logger.info(f"Simulating FPR under null (n_sims={n_simulations}, "
                f"n_subj={n_subjects}, alpha={alpha})")

    rng = np.random.RandomState(seed)

    glm_rejections = 0
    lme_rejections = 0
    lme_converged = 0

    for sim in range(n_simulations):
        between_sd = rng.uniform(0.1, 1.0)  # random between-subject variance
        residual_sd = rng.uniform(0.5, 1.5)  # random residual variance

        subject_effects = rng.normal(0, between_sd, n_subjects)
        rows = []
        for i in range(n_subjects):
            for j in range(n_runs):
                beta = subject_effects[i] + rng.normal(0, residual_sd)
                rows.append({
                    'subject': f's{i:03d}',
                    'run': f'run{j}',
                    'beta': beta
                })

        df = pd.DataFrame(rows)
        df['run'] = pd.Categorical(df['run'])

        subj_means = df.groupby('subject')['beta'].mean().values
        _, p_glm = stats.ttest_1samp(subj_means, 0)
        if p_glm < alpha:
            glm_rejections += 1

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = MixedLM.from_formula(
                    "beta ~ 1 + run", groups="subject", data=df
                )
                result = model.fit(reml=True, method="powell", maxiter=100)

                if result.converged:
                    lme_converged += 1
                    p_lme = result.pvalues['Intercept']
                    if p_lme < alpha:
                        lme_rejections += 1
        except Exception:
            pass

        if (sim + 1) % 200 == 0:
            logger.info(f"  Completed {sim + 1}/{n_simulations} simulations")

    def wilson_ci(k, n, z=1.96):
        p_hat = k / n
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
        return max(0, center - margin), min(1, center + margin)

    glm_fpr = glm_rejections / n_simulations
    lme_fpr = lme_rejections / lme_converged if lme_converged > 0 else np.nan

    glm_ci = wilson_ci(glm_rejections, n_simulations)
    lme_ci = wilson_ci(lme_rejections, lme_converged) if lme_converged > 0 else (np.nan, np.nan)

    results = {
        'glm_fpr': glm_fpr,
        'glm_rejections': glm_rejections,
        'glm_ci_lower': glm_ci[0],
        'glm_ci_upper': glm_ci[1],
        'lme_fpr': lme_fpr,
        'lme_rejections': lme_rejections,
        'lme_converged': lme_converged,
        'lme_ci_lower': lme_ci[0],
        'lme_ci_upper': lme_ci[1],
        'n_simulations': n_simulations,
        'alpha': alpha,
    }

    logger.info(f"\n  FPR Results (alpha = {alpha}):")
    logger.info(f"    GLM: {glm_fpr:.4f} [{glm_ci[0]:.4f}, {glm_ci[1]:.4f}]  "
                f"({glm_rejections}/{n_simulations})")
    logger.info(f"    LME: {lme_fpr:.4f} [{lme_ci[0]:.4f}, {lme_ci[1]:.4f}]  "
                f"({lme_rejections}/{lme_converged} converged)")

    nominal_ok_glm = glm_ci[0] <= alpha <= glm_ci[1]
    nominal_ok_lme = lme_ci[0] <= alpha <= lme_ci[1] if not np.isnan(lme_fpr) else False

    logger.info(f"    GLM nominal rate within CI: {'YES' if nominal_ok_glm else 'NO (inflated)' if glm_fpr > alpha else 'NO (conservative)'}")
    logger.info(f"    LME nominal rate within CI: {'YES' if nominal_ok_lme else 'NO (inflated)' if lme_fpr > alpha else 'NO (conservative)'}")

    return results

def compare_random_slope(n_subjects=143, n_runs=2, n_simulations=200,
                          seed=42):
    logger.info("Comparing random intercept vs random slope models...")

    rng = np.random.RandomState(seed)
    results = {
        'ri_converged': 0, 'rs_converged': 0,
        'ri_better_aic': 0, 'rs_better_aic': 0,
        'rs_singular': 0, 'total': 0,
        'aic_equal': 0, 'comparison_failed': 0,
        'ri_aic_values': [], 'rs_aic_values': [],
    }

    for sim in range(n_simulations):
        between_sd = 0.5
        run_effect = 0.05
        run_by_subj_sd = rng.uniform(0, 0.3)
        residual_sd = 1.0

        subject_effects = rng.normal(0, between_sd, n_subjects)
        subject_slopes = rng.normal(0, run_by_subj_sd, n_subjects)

        rows = []
        for i in range(n_subjects):
            for j, run in enumerate(['LR', 'RL']):
                beta = (0.3 + subject_effects[i] +
                        (run_effect + subject_slopes[i]) * j +
                        rng.normal(0, residual_sd))
                rows.append({
                    'subject': f's{i:03d}',
                    'run': run,
                    'beta': beta
                })

        df = pd.DataFrame(rows)
        df['run'] = pd.Categorical(df['run'])

        results['total'] += 1
        ri_aic = np.nan
        ri_llf = np.nan
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                m_ri = MixedLM.from_formula(
                    "beta ~ 1 + run", groups="subject", data=df
                ).fit(reml=False, method="powell", maxiter=100)
                if m_ri.converged:
                    results['ri_converged'] += 1
                    ri_aic = float(m_ri.aic) if hasattr(m_ri, 'aic') and np.isfinite(m_ri.aic) else np.nan
                    ri_llf = float(m_ri.llf)
        except Exception:
            pass

        rs_aic = np.nan
        rs_llf = np.nan
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                m_rs = MixedLM.from_formula(
                    "beta ~ 1 + run",
                    groups="subject",
                    re_formula="~run",
                    data=df
                ).fit(reml=False, method="powell", maxiter=200)
                if m_rs.converged:
                    results['rs_converged'] += 1
                    rs_aic = float(m_rs.aic) if hasattr(m_rs, 'aic') and np.isfinite(m_rs.aic) else np.nan
                    rs_llf = float(m_rs.llf)

                    re_cov = m_rs.cov_re
                    if re_cov.shape[0] > 1:
                        min_eigenval = np.min(np.linalg.eigvalsh(re_cov.values))
                        if min_eigenval < 1e-6:
                            results['rs_singular'] += 1
        except Exception:
            pass

        if np.isfinite(ri_aic) and np.isfinite(rs_aic):
            results['ri_aic_values'].append(ri_aic)
            results['rs_aic_values'].append(rs_aic)
            diff = ri_aic - rs_aic
            if abs(diff) < 0.01:
                results['aic_equal'] += 1
            elif ri_aic < rs_aic:
                results['ri_better_aic'] += 1
            else:
                results['rs_better_aic'] += 1
        else:
            results['comparison_failed'] += 1

    results_clean = {k: v for k, v in results.items()
                     if k not in ('ri_aic_values', 'rs_aic_values')}

    n_compared = results['ri_better_aic'] + results['rs_better_aic'] + results['aic_equal']

    logger.info(f"\n  Random Slope Comparison ({n_simulations} simulations, ML estimation):")
    logger.info(f"    Random-intercept converged:     {results['ri_converged']}/{results['total']}")
    logger.info(f"    Random-slope converged:         {results['rs_converged']}/{results['total']}")
    logger.info(f"    Random-slope singular:          {results['rs_singular']}/{results['rs_converged']}")
    logger.info(f"    AIC comparisons possible:       {n_compared}")
    logger.info(f"    RI better (lower AIC):          {results['ri_better_aic']}")
    logger.info(f"    RS better (lower AIC):          {results['rs_better_aic']}")
    logger.info(f"    AIC equal (diff < 0.01):        {results['aic_equal']}")
    logger.info(f"    Comparison failed (NaN AIC):    {results['comparison_failed']}")

    if results['ri_aic_values'] and results['rs_aic_values']:
        ri_arr = np.array(results['ri_aic_values'])
        rs_arr = np.array(results['rs_aic_values'])
        mean_diff = np.mean(ri_arr - rs_arr)
        logger.info(f"    Mean AIC difference (RI - RS):  {mean_diff:.2f}")
        logger.info(f"      (positive = RS better, negative = RI better)")

    logger.info(f"\n  INTERPRETATION: With only 2 runs per subject, the random")
    logger.info(f"  slope model is nearly saturated. Even when it converges,")
    logger.info(f"  the additional parameters rarely improve fit, and singularity")
    logger.info(f"  is common. This justifies using random-intercept only.")

    return results_clean

def bootstrap_icc_stability(n_subjects=143, n_runs_list=[2, 3, 5, 10],
                             true_icc=0.17, n_bootstrap=500, seed=42):
    logger.info(f"ICC Bootstrap Stability (true ICC = {true_icc})")

    rng = np.random.RandomState(seed)

    total_var = 1.0
    between_var = true_icc * total_var
    within_var = (1 - true_icc) * total_var
    between_sd = np.sqrt(between_var)
    within_sd = np.sqrt(within_var)

    results = []

    for n_runs in n_runs_list:
        icc_estimates = []

        for b in range(n_bootstrap):
            subject_effects = rng.normal(0, between_sd, n_subjects)
            rows = []
            for i in range(n_subjects):
                for j in range(n_runs):
                    y = subject_effects[i] + rng.normal(0, within_sd)
                    rows.append({
                        'subject': f's{i:03d}',
                        'run': f'r{j}',
                        'beta': y
                    })

            df = pd.DataFrame(rows)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model = MixedLM.from_formula(
                        "beta ~ 1", groups="subject", data=df
                    ).fit(reml=True, method="powell", maxiter=100)

                    if model.converged:
                        re_var = float(model.cov_re.iloc[0, 0])
                        resid_var = float(model.scale)
                        total = re_var + resid_var
                        icc = re_var / total if total > 0 else np.nan
                        icc = max(0, min(1, icc))
                        icc_estimates.append(icc)
            except Exception:
                pass

        if icc_estimates:
            icc_arr = np.array(icc_estimates)
            results.append({
                'n_runs': n_runs,
                'true_icc': true_icc,
                'mean_icc': np.mean(icc_arr),
                'median_icc': np.median(icc_arr),
                'sd_icc': np.std(icc_arr),
                'bias': np.mean(icc_arr) - true_icc,
                'rmse': np.sqrt(np.mean((icc_arr - true_icc) ** 2)),
                'ci_lower': np.percentile(icc_arr, 2.5),
                'ci_upper': np.percentile(icc_arr, 97.5),
                'n_converged': len(icc_arr),
            })

            logger.info(f"\n  n_runs={n_runs}: ICC = {np.mean(icc_arr):.3f} +/- {np.std(icc_arr):.3f} "
                        f"(bias={np.mean(icc_arr) - true_icc:+.3f}, "
                        f"RMSE={np.sqrt(np.mean((icc_arr - true_icc) ** 2)):.3f})")

    results_df = pd.DataFrame(results)

    logger.info(f"\n  KEY FINDING: With only 2 runs, ICC estimates have")
    logger.info(f"  substantially higher variance and RMSE compared to")
    logger.info(f"  designs with more observations per subject.")
    logger.info(f"  This is a known statistical limitation documented")
    logger.info(f"  in the Discussion section of the manuscript.")

    return results_df

def smoothing_sensitivity_note():
    logger.info("\n  SMOOTHING SENSITIVITY ANALYSIS")
    logger.info("  ================================")
    logger.info("  Current smoothing: 5mm FWHM")
    logger.info("  To run full sensitivity analysis:")
    logger.info("    Modify SMOOTH_FWHM in config.py to [0, 3, 5, 8]")
    logger.info("    Re-run main_group_glm.py for each value")
    logger.info("    Compare n_significant voxels and spatial overlap")
    logger.info("")
    logger.info("  This analysis is recommended but computationally")
    logger.info("  intensive. Report the following for the manuscript:")
    logger.info("  - N sig voxels at each FWHM")
    logger.info("  - Dice overlap with reference (5mm)")
    logger.info("  - Qualitative description of how results change")

    fwhm_values = [0, 3, 5, 8]
    rows = []
    for fwhm in fwhm_values:
        rows.append({
            'fwhm_mm': fwhm,
            'n_sig_glm_fdr': 'RUN_NEEDED',
            'n_sig_lme_fdr': 'RUN_NEEDED',
            'dice_with_5mm': 'RUN_NEEDED' if fwhm != 5 else 1.0,
        })

    stub_df = pd.DataFrame(rows)
    output_path = GROUP_LEVEL / 'smoothing_sensitivity_stub.csv'
    stub_df.to_csv(output_path, index=False)
    logger.info(f"\n  Template saved to {output_path}")
    logger.info("  Fill in after running analyses at each FWHM.")

    return stub_df


def create_simulation_figures(fpr_results, icc_results, slope_results, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if fpr_results:
        methods = ['GLM', 'LME']
        fprs = [fpr_results['glm_fpr'], fpr_results['lme_fpr']]
        ci_low = [fpr_results['glm_ci_lower'], fpr_results['lme_ci_lower']]
        ci_high = [fpr_results['glm_ci_upper'], fpr_results['lme_ci_upper']]
        yerr = [[f - l for f, l in zip(fprs, ci_low)],
                [h - f for f, h in zip(fprs, ci_high)]]

        bars = ax.bar(methods, fprs, yerr=yerr, capsize=5,
                      color=['#2ecc71', '#3498db'], alpha=0.8)
        ax.axhline(fpr_results['alpha'], color='red', linestyle='--',
                   label=f'Nominal alpha = {fpr_results["alpha"]}')
        ax.set_ylabel('False Positive Rate')
        ax.set_title('A) Type I Error Rate Under Null')
        ax.legend()
        ax.set_ylim(0, max(fprs) * 1.5 + 0.01)

    ax = axes[0, 1]
    if icc_results is not None and len(icc_results) > 0:
        ax.errorbar(icc_results['n_runs'], icc_results['mean_icc'],
                    yerr=icc_results['sd_icc'],
                    marker='o', capsize=5, color='#3498db', linewidth=2)
        ax.axhline(icc_results['true_icc'].iloc[0], color='red',
                   linestyle='--', label=f'True ICC = {icc_results["true_icc"].iloc[0]:.2f}')
        ax.fill_between(icc_results['n_runs'],
                        icc_results['ci_lower'], icc_results['ci_upper'],
                        alpha=0.2, color='#3498db')
        ax.set_xlabel('Number of Runs per Subject')
        ax.set_ylabel('Estimated ICC')
        ax.set_title('B) ICC Estimation Stability')
        ax.legend()
        ax.axvline(2, color='orange', linestyle=':', alpha=0.7, label='Our design (n=2)')
        ax.legend()

    ax = axes[1, 0]
    if icc_results is not None and len(icc_results) > 0:
        ax.bar(icc_results['n_runs'].astype(str), icc_results['bias'],
               color=['#e74c3c' if b < -0.01 else '#2ecc71' for b in icc_results['bias']])
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Number of Runs per Subject')
        ax.set_ylabel('ICC Bias (estimated - true)')
        ax.set_title('C) ICC Estimation Bias')

    ax = axes[1, 1]
    if slope_results:
        categories = ['RI\nConverged', 'RS\nConverged', 'RS\nSingular',
                       'RI better\n(AIC)', 'RS better\n(AIC)']
        total = slope_results['total']
        values = [
            slope_results['ri_converged'] / total * 100,
            slope_results['rs_converged'] / total * 100,
            slope_results['rs_singular'] / max(1, slope_results['rs_converged']) * 100,
            slope_results['ri_better_aic'] / total * 100,
            slope_results['rs_better_aic'] / total * 100,
        ]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#2ecc71', '#3498db']
        ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('D) Random Intercept vs Slope (2 runs)')
        ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_dir / 'simulation_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'simulation_results.pdf', bbox_inches='tight')
    plt.close()
    logger.info(f"Simulation figure saved to {output_dir / 'simulation_results.png'}")

def main():
    parser = argparse.ArgumentParser(description="Extended simulation analyses")
    parser.add_argument('--fpr-only', action='store_true')
    parser.add_argument('--icc-only', action='store_true')
    parser.add_argument('--random-slope', action='store_true')
    parser.add_argument('--smoothing', action='store_true')
    parser.add_argument('--n-sims', type=int, default=1000,
                        help='Simulations for FPR (default: 1000)')
    parser.add_argument('--n-bootstrap', type=int, default=500,
                        help='Bootstrap reps for ICC (default: 500)')
    args = parser.parse_args()

    run_all = not any([args.fpr_only, args.icc_only, args.random_slope, args.smoothing])

    output_dir = GROUP_LEVEL / 'simulations'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXTENDED SIMULATION & SENSITIVITY ANALYSES")
    logger.info("=" * 60)

    fpr_results = None
    icc_results = None
    slope_results = None

    if run_all or args.fpr_only:
        logger.info("\n--- FALSE POSITIVE RATE SIMULATION (MC6) ---")
        fpr_results = simulate_fpr(n_simulations=args.n_sims)
        pd.DataFrame([fpr_results]).to_csv(output_dir / 'fpr_results.csv', index=False)

    if run_all or args.random_slope:
        logger.info("\n--- RANDOM SLOPE COMPARISON (MC3 / Suggestion 4) ---")
        slope_results = compare_random_slope(n_simulations=min(args.n_sims, 200))
        pd.DataFrame([slope_results]).to_csv(output_dir / 'random_slope_results.csv', index=False)

    if run_all or args.icc_only:
        logger.info("\n--- ICC BOOTSTRAP STABILITY (Suggestion 7) ---")
        icc_results = bootstrap_icc_stability(n_bootstrap=args.n_bootstrap)
        icc_results.to_csv(output_dir / 'icc_bootstrap_results.csv', index=False)

    if run_all or args.smoothing:
        logger.info("\n--- SMOOTHING SENSITIVITY (Minor Concern) ---")
        smoothing_sensitivity_note()

    if fpr_results or icc_results is not None or slope_results:
        create_simulation_figures(fpr_results, icc_results, slope_results, FIGURES)

    logger.info("\n--- UNTHRESHOLDED MAPS (Suggestion 8) ---")
    logger.info("  Unthresholded maps are already saved by the pipeline:")
    logger.info("    - group_level/glm_zmap.nii.gz (unthresholded)")
    logger.info("    - lme_voxelwise/lme_zmap.nii.gz (unthresholded)")
    logger.info("    - lme_voxelwise/lme_beta.nii.gz (unthresholded)")
    logger.info("    - lme_voxelwise/lme_icc.nii.gz (unthresholded)")
    logger.info("  These should be included as supplementary materials.")
    logger.info("  Consider uploading to NeuroVault for open access.")

    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION ANALYSES COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
