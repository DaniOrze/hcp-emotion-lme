import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import GROUP_LEVEL

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_validation_data(n_subjects=50, n_runs=2, true_intercept=0.3,
                              true_run_effect=0.05, between_subj_sd=0.5,
                              residual_sd=1.0, seed=42):
    rng = np.random.RandomState(seed)
    subject_effects = rng.normal(0, between_subj_sd, n_subjects)
    runs = ['LR', 'RL']

    rows = []
    for i in range(n_subjects):
        for j, run in enumerate(runs[:n_runs]):
            beta = (true_intercept +
                    subject_effects[i] +
                    (true_run_effect if run == 'RL' else 0) +
                    rng.normal(0, residual_sd))
            rows.append({'subject': f'sub_{i:03d}', 'run': run, 'beta': beta})

    df = pd.DataFrame(rows)
    df['run'] = pd.Categorical(df['run'])

    true_icc = between_subj_sd**2 / (between_subj_sd**2 + residual_sd**2)
    ground_truth = {
        'intercept': true_intercept,
        'run_effect': true_run_effect,
        'between_subj_var': between_subj_sd**2,
        'residual_var': residual_sd**2,
        'icc': true_icc,
    }
    return df, ground_truth

def fit_statsmodels_lme(df):
    from statsmodels.regression.mixed_linear_model import MixedLM
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = MixedLM.from_formula("beta ~ 1 + run", groups="subject", data=df)
        result = model.fit(reml=True, method="powell", maxiter=200)

    intercept = result.fe_params['Intercept']
    se_intercept = result.bse['Intercept']
    p_intercept = result.pvalues['Intercept']

    run_keys = [k for k in result.fe_params.index if k.startswith('run')]
    run_effect = result.fe_params[run_keys[0]] if run_keys else np.nan
    p_run = result.pvalues[run_keys[0]] if run_keys else np.nan

    re_var = float(result.cov_re.iloc[0, 0])
    resid_var = float(result.scale)
    total_var = re_var + resid_var
    icc = re_var / total_var if total_var > 0 else np.nan

    return {
        'intercept': float(intercept),
        'se_intercept': float(se_intercept),
        'p_intercept': float(p_intercept),
        'run_effect': float(run_effect),
        'p_run': float(p_run),
        'between_subj_var': float(re_var),
        'residual_var': float(resid_var),
        'icc': float(icc),
        'converged': result.converged,
    }


def fit_pymer4_lme(df):
    from pymer4.models import Lmer

    model = Lmer("beta ~ 1 + run + (1|subject)", data=df)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(REML=True, summarize=False)

    coefs = model.coefs
    intercept = float(coefs.loc['Intercept', 'Estimate'])
    se_intercept = float(coefs.loc['Intercept', 'SE'])
    p_intercept = float(coefs.loc['Intercept', 'P-val'])

    run_keys = [k for k in coefs.index if k.startswith('run')]
    run_effect = float(coefs.loc[run_keys[0], 'Estimate']) if run_keys else np.nan
    p_run = float(coefs.loc[run_keys[0], 'P-val']) if run_keys else np.nan

    ranef_var = model.ranef_var
    re_var = float(ranef_var[ranef_var['grp'] == 'subject']['vcov'].values[0])
    resid_var = float(ranef_var[ranef_var['grp'] == 'Residual']['vcov'].values[0])
    total_var = re_var + resid_var
    icc = re_var / total_var if total_var > 0 else np.nan

    return {
        'intercept': float(intercept),
        'se_intercept': float(se_intercept),
        'p_intercept': float(p_intercept),
        'run_effect': float(run_effect),
        'p_run': float(p_run),
        'between_subj_var': float(re_var),
        'residual_var': float(resid_var),
        'icc': float(icc),
        'converged': True,
    }

def concordance_correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return np.nan
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    cov_xy = np.cov(x, y)[0, 1]
    ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)**2)
    return ccc

def run_validation(use_real_data=False, n_simulations=100):
    logger.info("=" * 60)
    logger.info("LME IMPLEMENTATION VALIDATION")
    logger.info("Formal comparison: statsmodels vs R lme4 (via pymer4)")
    logger.info("=" * 60)

    try:
        from pymer4.models import Lmer
        has_pymer4 = True
        logger.info("pymer4 (R lme4) is AVAILABLE — formal validation enabled")
    except ImportError:
        has_pymer4 = False
        logger.warning("pymer4 NOT available — only parameter recovery possible")

    logger.info("\n--- TEST 1: Parameter Recovery (100 simulations) ---")

    scenarios = [
        {'name': 'Moderate effect', 'true_intercept': 0.3, 'between_subj_sd': 0.5, 'residual_sd': 1.0},
        {'name': 'Large effect', 'true_intercept': 1.0, 'between_subj_sd': 0.3, 'residual_sd': 0.8},
        {'name': 'Small effect', 'true_intercept': 0.1, 'between_subj_sd': 0.8, 'residual_sd': 1.2},
        {'name': 'High ICC', 'true_intercept': 0.5, 'between_subj_sd': 1.5, 'residual_sd': 0.5},
        {'name': 'Low ICC', 'true_intercept': 0.5, 'between_subj_sd': 0.1, 'residual_sd': 1.5},
    ]

    recovery_results = []
    for scenario in scenarios:
        intercepts_sm = []
        iccs_sm = []
        for sim in range(n_simulations):
            df, gt = generate_validation_data(
                n_subjects=143, n_runs=2, seed=sim,
                true_intercept=scenario['true_intercept'],
                between_subj_sd=scenario['between_subj_sd'],
                residual_sd=scenario['residual_sd'],
            )
            res = fit_statsmodels_lme(df)
            intercepts_sm.append(res['intercept'])
            iccs_sm.append(res['icc'])

        true_icc = scenario['between_subj_sd']**2 / (
            scenario['between_subj_sd']**2 + scenario['residual_sd']**2
        )

        row = {
            'scenario': scenario['name'],
            'true_intercept': scenario['true_intercept'],
            'est_intercept_mean': np.mean(intercepts_sm),
            'bias_intercept': np.mean(intercepts_sm) - scenario['true_intercept'],
            'true_icc': true_icc,
            'est_icc_mean': np.mean(iccs_sm),
            'bias_icc': np.mean(iccs_sm) - true_icc,
            'coverage_95': np.mean([
                abs(i - scenario['true_intercept']) < 1.96 * np.std(intercepts_sm)
                for i in intercepts_sm
            ]),
        }
        recovery_results.append(row)

        logger.info(f"\n  Scenario: {scenario['name']}")
        logger.info(f"    Intercept: true={scenario['true_intercept']:.3f}, "
                     f"est={np.mean(intercepts_sm):.3f}, "
                     f"bias={row['bias_intercept']:.4f}")
        logger.info(f"    ICC: true={true_icc:.3f}, "
                     f"est={np.mean(iccs_sm):.3f}, "
                     f"bias={row['bias_icc']:.4f}")

    recovery_df = pd.DataFrame(recovery_results)

    comparison_df = None
    if has_pymer4:
        logger.info("\n--- TEST 2: Formal comparison statsmodels vs R lme4 ---")
        logger.info("  Fitting 15 synthetic ROIs with both backends...")

        comparison_rows = []
        n_rois = 15
        for roi_idx in range(n_rois):
            between_sd = 0.1 + roi_idx * 0.15
            resid_sd = 1.0
            true_int = 0.1 + roi_idx * 0.1

            df, gt = generate_validation_data(
                n_subjects=143, n_runs=2, seed=roi_idx * 10,
                true_intercept=true_int,
                between_subj_sd=between_sd,
                residual_sd=resid_sd,
            )

            sm_res = fit_statsmodels_lme(df)
            try:
                r_res = fit_pymer4_lme(df)
            except Exception as e:
                logger.warning(f"  ROI {roi_idx}: pymer4 failed: {e}")
                continue

            row = {
                'roi': f'synth_roi_{roi_idx:02d}',
                'true_intercept': true_int,
                'true_icc': gt['icc'],
                # statsmodels
                'sm_intercept': sm_res['intercept'],
                'sm_se': sm_res['se_intercept'],
                'sm_p': sm_res['p_intercept'],
                'sm_icc': sm_res['icc'],
                'sm_run_effect': sm_res['run_effect'],
                'sm_var_between': sm_res['between_subj_var'],
                'sm_var_resid': sm_res['residual_var'],
                # lme4
                'r_intercept': r_res['intercept'],
                'r_se': r_res['se_intercept'],
                'r_p': r_res['p_intercept'],
                'r_icc': r_res['icc'],
                'r_run_effect': r_res['run_effect'],
                'r_var_between': r_res['between_subj_var'],
                'r_var_resid': r_res['residual_var'],
                # Differences
                'diff_intercept': abs(sm_res['intercept'] - r_res['intercept']),
                'diff_se': abs(sm_res['se_intercept'] - r_res['se_intercept']),
                'diff_icc': abs(sm_res['icc'] - r_res['icc']),
                'diff_run': abs(sm_res['run_effect'] - r_res['run_effect']),
            }
            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)

        if len(comparison_df) > 0:
            logger.info(f"\n  {'Parameter':<20} {'Max |Diff|':>12} {'CCC':>10} {'PASS':>6}")
            logger.info(f"  {'-'*52}")

            for param_pair, label in [
                (('sm_intercept', 'r_intercept'), 'Intercept'),
                (('sm_se', 'r_se'), 'SE(Intercept)'),
                (('sm_icc', 'r_icc'), 'ICC'),
                (('sm_run_effect', 'r_run_effect'), 'Run effect'),
                (('sm_var_between', 'r_var_between'), 'Var(between)'),
                (('sm_var_resid', 'r_var_resid'), 'Var(residual)'),
            ]:
                sm_vals = comparison_df[param_pair[0]].values
                r_vals = comparison_df[param_pair[1]].values
                max_diff = np.nanmax(np.abs(sm_vals - r_vals))
                ccc = concordance_correlation(sm_vals, r_vals)
                passed = max_diff < 0.01 and ccc > 0.999
                logger.info(f"  {label:<20} {max_diff:>12.6f} {ccc:>10.6f} "
                             f"{'YES' if passed else 'NO':>6}")

            max_intercept_diff = comparison_df['diff_intercept'].max()
            max_icc_diff = comparison_df['diff_icc'].max()
            ccc_intercept = concordance_correlation(
                comparison_df['sm_intercept'], comparison_df['r_intercept'])

            if max_intercept_diff < 0.01 and max_icc_diff < 0.02:
                logger.info("\n  PASS: statsmodels and R lme4 produce equivalent results")
                logger.info(f"    Max intercept diff: {max_intercept_diff:.6f}")
                logger.info(f"    Max ICC diff: {max_icc_diff:.6f}")
                logger.info(f"    CCC(intercept): {ccc_intercept:.6f}")
            else:
                logger.warning(f"\n  ATTENTION: Max intercept diff={max_intercept_diff:.6f}, "
                               f"ICC diff={max_icc_diff:.6f}")

    roi_betas_path = GROUP_LEVEL / 'roi_betas_v2.csv'
    if use_real_data and roi_betas_path.exists() and has_pymer4:
        logger.info("\n--- TEST 3: Formal comparison on REAL ROI data ---")
        roi_df = pd.read_csv(roi_betas_path)
        rois = roi_df['roi'].unique()[:15]  # up to 15 ROIs

        real_comparison_rows = []
        for roi in rois:
            roi_data = roi_df[roi_df['roi'] == roi].copy()
            roi_data['run'] = pd.Categorical(roi_data['run'])

            try:
                sm_res = fit_statsmodels_lme(roi_data)
                r_res = fit_pymer4_lme(roi_data)
            except Exception as e:
                logger.warning(f"  {roi}: comparison failed: {e}")
                continue

            real_comparison_rows.append({
                'roi': roi,
                'sm_intercept': sm_res['intercept'],
                'r_intercept': r_res['intercept'],
                'diff_intercept': abs(sm_res['intercept'] - r_res['intercept']),
                'sm_icc': sm_res['icc'],
                'r_icc': r_res['icc'],
                'diff_icc': abs(sm_res['icc'] - r_res['icc']),
                'sm_p': sm_res['p_intercept'],
                'r_p': r_res['p_intercept'],
            })

            logger.info(f"  {roi}: intercept diff={abs(sm_res['intercept'] - r_res['intercept']):.6f}, "
                         f"ICC diff={abs(sm_res['icc'] - r_res['icc']):.6f}")

        if real_comparison_rows:
            real_df = pd.DataFrame(real_comparison_rows)
            output_dir = GROUP_LEVEL / 'validation'
            output_dir.mkdir(parents=True, exist_ok=True)
            real_df.to_csv(output_dir / 'real_roi_comparison.csv', index=False)
            logger.info(f"\n  Max intercept diff (real data): {real_df['diff_intercept'].max():.6f}")
            logger.info(f"  Max ICC diff (real data): {real_df['diff_icc'].max():.6f}")

    output_dir = GROUP_LEVEL / 'validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    recovery_df.to_csv(output_dir / 'parameter_recovery.csv', index=False)
    if comparison_df is not None:
        comparison_df.to_csv(output_dir / 'statsmodels_vs_lme4_comparison.csv', index=False)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)

    return recovery_df, comparison_df


def main():
    parser = argparse.ArgumentParser(description="Validate LME implementation")
    parser.add_argument('--real-data', action='store_true',
                        help='Also compare on real ROI data')
    parser.add_argument('--n-sims', type=int, default=100,
                        help='Number of simulations for parameter recovery')
    args = parser.parse_args()

    run_validation(use_real_data=args.real_data, n_simulations=args.n_sims)


if __name__ == "__main__":
    main()
