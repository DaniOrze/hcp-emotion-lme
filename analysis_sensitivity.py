import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONTRAST_NAME, FD_THRESHOLD
from paths import FIRST_LEVEL, GROUP_LEVEL, FIGURES, LOGS, get_subject_ids
from utils.lme_v2 import fit_lme_single_voxel, get_backend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"sensitivity_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_fd_per_run():
    subject_ids = get_subject_ids()
    rows = []
    for subj in subject_ids:
        qc_path = FIRST_LEVEL / subj / "motion_qc.csv"
        if qc_path.exists():
            try:
                qc_df = pd.read_csv(qc_path, index_col=0)
                for run in qc_df.columns:
                    row = qc_df[run].to_dict()
                    row['subject'] = subj
                    row['run'] = run
                    rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to read motion QC for {subj}: {e}")

    if not rows:
        logger.warning("No motion QC data found. Run main_first_level.py first.")
        return pd.DataFrame()

    fd_df = pd.DataFrame(rows)
    logger.info(f"Loaded FD data for {fd_df['subject'].nunique()} subjects, "
                f"{len(fd_df)} subject-run pairs")
    return fd_df


def run_fd_sensitivity(beta_df, fd_df, rois=None):
    logger.info("\n" + "=" * 60)
    logger.info("MC2: FD SENSITIVITY ANALYSIS")
    logger.info("Model A: beta ~ 1 + run + (1|subject)")
    logger.info("Model B: beta ~ 1 + run + mean_fd + (1|subject)")
    logger.info("=" * 60)

    merged = beta_df.merge(fd_df[['subject', 'run', 'mean_fd']],
                           on=['subject', 'run'], how='inner')

    if len(merged) == 0:
        logger.error("No overlap between beta and FD data!")
        return pd.DataFrame()

    logger.info(f"Merged data: {len(merged)} observations, "
                f"{merged['subject'].nunique()} subjects")

    if rois is None:
        rois = sorted(merged['roi'].unique())

    results = []
    for roi in rois:
        roi_data = merged[merged['roi'] == roi].copy()
        if len(roi_data) < 20:
            continue

        roi_data['run'] = pd.Categorical(roi_data['run'])
        betas = roi_data['beta'].values

        run_info_no_fd = roi_data[['subject', 'run']].copy()
        run_info_with_fd = roi_data[['subject', 'run', 'mean_fd']].copy()

        res_a = fit_lme_single_voxel(betas, run_info_no_fd,
                                      include_run_effect=True,
                                      fd_covariate=False)

        res_b = fit_lme_single_voxel(betas, run_info_with_fd,
                                      include_run_effect=True,
                                      fd_covariate=True)

        row = {
            'roi': roi,
            'n_obs': len(roi_data),
            # Model A (no FD)
            'intercept_nofd': res_a.get('fixed_effect', np.nan),
            'intercept_p_nofd': res_a.get('p_value', np.nan),
            'icc_nofd': res_a.get('icc', np.nan),
            # Model B (with FD)
            'intercept_fd': res_b.get('fixed_effect', np.nan),
            'intercept_p_fd': res_b.get('p_value', np.nan),
            'icc_fd': res_b.get('icc', np.nan),
            # Changes
            'intercept_change': (res_b.get('fixed_effect', np.nan) -
                                  res_a.get('fixed_effect', np.nan)),
            'icc_change': (res_b.get('icc', np.nan) -
                           res_a.get('icc', np.nan)),
            'converged_a': res_a.get('converged', False),
            'converged_b': res_b.get('converged', False),
            'backend': res_a.get('backend', 'unknown'),
        }
        results.append(row)

        logger.info(f"  {roi}: intercept {row['intercept_nofd']:.4f} -> "
                     f"{row['intercept_fd']:.4f} "
                     f"(delta={row['intercept_change']:.4f}), "
                     f"ICC {row['icc_nofd']:.3f} -> {row['icc_fd']:.3f}")

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        logger.info(f"\n  SUMMARY:")
        logger.info(f"  Mean |intercept change|: "
                     f"{results_df['intercept_change'].abs().mean():.4f}")
        logger.info(f"  Mean |ICC change|: "
                     f"{results_df['icc_change'].abs().mean():.4f}")

        # Highlight amygdala specifically (MC2 concern)
        amyg_rows = results_df[results_df['roi'].str.contains('Amygdala', case=False)]
        if len(amyg_rows) > 0:
            logger.info(f"\n  AMYGDALA (specifically flagged in MC2):")
            for _, row in amyg_rows.iterrows():
                logger.info(f"    {row['roi']}: intercept change={row['intercept_change']:.4f}, "
                             f"ICC change={row['icc_change']:.4f}")

    return results_df

def compute_residual_acf(beta_df, rois=None, max_lag=10):
    logger.info("\n" + "=" * 60)
    logger.info("MC4: RESIDUAL AUTOCORRELATION DIAGNOSTICS")
    logger.info("=" * 60)

    if rois is None:
        rois = sorted(beta_df['roi'].unique())

    acf_results = []
    residual_store = {}

    for roi in rois:
        roi_data = beta_df[beta_df['roi'] == roi].copy()
        if len(roi_data) < 20:
            continue

        roi_data['run'] = pd.Categorical(roi_data['run'])
        betas = roi_data['beta'].values
        run_info = roi_data[['subject', 'run']].copy()

        res = fit_lme_single_voxel(betas, run_info, include_run_effect=True)

        if not res.get('converged', False) or 'residuals' not in res:
            logger.warning(f"  {roi}: no residuals available")
            continue

        residuals = res['residuals']
        residual_store[roi] = residuals

        n = len(residuals)
        mean_r = np.mean(residuals)
        var_r = np.var(residuals)

        if var_r < 1e-10:
            continue

        acf_values = []
        for lag in range(min(max_lag + 1, n)):
            if lag == 0:
                acf_values.append(1.0)
            else:
                acf = np.mean((residuals[:n-lag] - mean_r) *
                              (residuals[lag:] - mean_r)) / var_r
                acf_values.append(float(acf))

        if len(acf_values) > 1:
            lb_stat = n * (n + 2) * (acf_values[1]**2 / (n - 1))
            lb_p = 1 - stats.chi2.cdf(lb_stat, df=1)
        else:
            lb_stat = np.nan
            lb_p = np.nan

        row = {
            'roi': roi,
            'n_residuals': n,
            'acf_lag1': acf_values[1] if len(acf_values) > 1 else np.nan,
            'acf_lag2': acf_values[2] if len(acf_values) > 2 else np.nan,
            'ljung_box_stat': lb_stat,
            'ljung_box_p': lb_p,
            'residual_mean': float(np.mean(residuals)),
            'residual_sd': float(np.std(residuals)),
        }
        acf_results.append(row)

        sig_str = " *SIGNIFICANT*" if lb_p < 0.05 else ""
        logger.info(f"  {roi}: ACF(1)={row['acf_lag1']:.4f}, "
                     f"LB p={lb_p:.4f}{sig_str}")

    acf_df = pd.DataFrame(acf_results)

    if len(acf_df) > 0:
        n_sig = (acf_df['ljung_box_p'] < 0.05).sum()
        logger.info(f"\n  SUMMARY: {n_sig}/{len(acf_df)} ROIs show significant "
                     f"autocorrelation at lag 1 (p < 0.05)")
        logger.info(f"  Mean |ACF(1)|: {acf_df['acf_lag1'].abs().mean():.4f}")

    return acf_df, residual_store


def create_acf_figure(acf_df, residual_store, output_dir):
    import matplotlib.pyplot as plt

    rois_to_plot = list(residual_store.keys())[:6]

    if len(rois_to_plot) == 0:
        logger.warning("No residuals available for ACF plot")
        return

    n_plots = len(rois_to_plot)
    fig, axes = plt.subplots(2, min(3, n_plots), figsize=(15, 8))
    if n_plots == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for idx, roi in enumerate(rois_to_plot):
        if idx >= len(axes):
            break
        ax = axes[idx]
        residuals = residual_store[roi]
        n = len(residuals)
        max_lag = min(15, n - 1)

        mean_r = np.mean(residuals)
        var_r = np.var(residuals)
        acf_vals = []
        for lag in range(max_lag + 1):
            if lag == 0:
                acf_vals.append(1.0)
            else:
                acf = np.mean((residuals[:n-lag] - mean_r) *
                              (residuals[lag:] - mean_r)) / var_r
                acf_vals.append(acf)

        lags = list(range(max_lag + 1))
        ax.bar(lags, acf_vals, color='steelblue', alpha=0.7)
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, color='red', linestyle='--', alpha=0.5)
        ax.axhline(-ci, color='red', linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linewidth=0.5)

        short_name = roi.replace('cortical_', 'c:').replace('subcortical_', 's:')
        ax.set_title(f'{short_name}', fontsize=9)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_ylim(-0.5, 1.05)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('MC4: Autocorrelation of LME Residuals\n'
                 '(red dashed = 95% CI under null of no autocorrelation)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'mc4_residual_acf.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'mc4_residual_acf.pdf', bbox_inches='tight')
    plt.close()
    logger.info(f"ACF figure saved to {output_dir / 'mc4_residual_acf.png'}")

def bootstrap_icc_ci(beta_df, n_bootstrap=2000, seed=42):
    logger.info("\n" + "=" * 60)
    logger.info("mn3: BOOTSTRAP 95% CIs FOR ICC")
    logger.info("=" * 60)

    rng = np.random.RandomState(seed)
    rois = sorted(beta_df['roi'].unique())
    results = []

    for roi in rois:
        roi_data = beta_df[beta_df['roi'] == roi].copy()
        subjects = roi_data['subject'].unique()

        if len(subjects) < 10:
            continue

        boot_iccs = []
        for _ in range(n_bootstrap):
            boot_subj = rng.choice(subjects, size=len(subjects), replace=True)
            frames = []
            for i, s in enumerate(boot_subj):
                subj_data = roi_data[roi_data['subject'] == s].copy()
                subj_data['subject'] = f"boot_{i:04d}"
                frames.append(subj_data)

            boot_df = pd.concat(frames, ignore_index=True)
            boot_df['run'] = pd.Categorical(boot_df['run'])
            betas = boot_df['beta'].values
            run_info = boot_df[['subject', 'run']].copy()

            res = fit_lme_single_voxel(betas, run_info, include_run_effect=True)
            if res.get('converged', False) and np.isfinite(res.get('icc', np.nan)):
                boot_iccs.append(res['icc'])

        if len(boot_iccs) < n_bootstrap * 0.5:
            logger.warning(f"  {roi}: only {len(boot_iccs)}/{n_bootstrap} "
                           f"successful bootstraps")

        if len(boot_iccs) >= 10:
            point_est = np.mean(boot_iccs)
            ci_lower = np.percentile(boot_iccs, 2.5)
            ci_upper = np.percentile(boot_iccs, 97.5)
        else:
            point_est = ci_lower = ci_upper = np.nan

        results.append({
            'roi': roi,
            'icc_point': point_est,
            'icc_ci_lower': ci_lower,
            'icc_ci_upper': ci_upper,
            'icc_boot_se': np.std(boot_iccs) if len(boot_iccs) > 1 else np.nan,
            'n_successful_boots': len(boot_iccs),
        })

        logger.info(f"  {roi}: ICC = {point_est:.3f} "
                     f"[{ci_lower:.3f}, {ci_upper:.3f}]")

    return pd.DataFrame(results)

def report_fd_distribution(fd_df):
    logger.info("\n" + "=" * 60)
    logger.info("mn5: FRAMEWISE DISPLACEMENT DISTRIBUTION")
    logger.info("=" * 60)

    if len(fd_df) == 0:
        logger.warning("No FD data available")
        return {}

    logger.info(f"  N subject-run pairs: {len(fd_df)}")
    logger.info(f"  N subjects: {fd_df['subject'].nunique()}")

    mean_fd = fd_df['mean_fd'].values
    logger.info(f"\n  Mean FD across runs:")
    logger.info(f"    Mean:      {np.mean(mean_fd):.4f} mm")
    logger.info(f"    Median:    {np.median(mean_fd):.4f} mm")
    logger.info(f"    SD:        {np.std(mean_fd):.4f} mm")
    logger.info(f"    Range:     [{np.min(mean_fd):.4f}, {np.max(mean_fd):.4f}] mm")
    logger.info(f"    P25:       {np.percentile(mean_fd, 25):.4f} mm")
    logger.info(f"    P75:       {np.percentile(mean_fd, 75):.4f} mm")
    logger.info(f"    P95:       {np.percentile(mean_fd, 95):.4f} mm")

    if 'percent_above' in fd_df.columns:
        pct_above = fd_df['percent_above'].values
        logger.info(f"\n  Percent frames > {FD_THRESHOLD}mm:")
        logger.info(f"    Mean:      {np.mean(pct_above):.2f}%")
        logger.info(f"    Median:    {np.median(pct_above):.2f}%")
        logger.info(f"    P95:       {np.percentile(pct_above, 95):.2f}%")

    subj_mean_fd = fd_df.groupby('subject')['mean_fd'].mean()
    logger.info(f"\n  Per-subject mean FD (averaged across runs):")
    logger.info(f"    Mean:      {subj_mean_fd.mean():.4f} mm")
    logger.info(f"    SD:        {subj_mean_fd.std():.4f} mm")
    logger.info(f"    P95:       {np.percentile(subj_mean_fd, 95):.4f} mm")

    summary = {
        'n_subjects': int(fd_df['subject'].nunique()),
        'n_runs': int(len(fd_df)),
        'mean_fd_mean': float(np.mean(mean_fd)),
        'mean_fd_median': float(np.median(mean_fd)),
        'mean_fd_sd': float(np.std(mean_fd)),
        'mean_fd_p95': float(np.percentile(mean_fd, 95)),
        'mean_fd_range': [float(np.min(mean_fd)), float(np.max(mean_fd))],
    }

    return summary

def main():
    parser = argparse.ArgumentParser(description="Sensitivity analyses")
    parser.add_argument('--fd-only', action='store_true')
    parser.add_argument('--acf-only', action='store_true')
    parser.add_argument('--icc-ci-only', action='store_true')
    parser.add_argument('--fd-report-only', action='store_true')
    parser.add_argument('--n-bootstrap', type=int, default=2000)
    args = parser.parse_args()

    run_all = not any([args.fd_only, args.acf_only,
                       args.icc_ci_only, args.fd_report_only])

    output_dir = GROUP_LEVEL / 'sensitivity'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SENSITIVITY & DIAGNOSTIC ANALYSES")
    logger.info(f"LME backend: {get_backend()}")
    logger.info("=" * 60)

    beta_path = GROUP_LEVEL / 'roi_betas_v2.csv'
    if not beta_path.exists():
        beta_path = GROUP_LEVEL / 'roi_betas.csv'

    beta_df = None
    if beta_path.exists():
        beta_df = pd.read_csv(beta_path)
        logger.info(f"Loaded {len(beta_df)} ROI beta observations")
    else:
        logger.warning("ROI betas not found — run main_roi_analysis_v2.py first")

    fd_df = load_fd_per_run()

    if run_all or args.fd_report_only:
        if len(fd_df) > 0:
            fd_summary = report_fd_distribution(fd_df)
            pd.DataFrame([fd_summary]).to_csv(
                output_dir / 'fd_distribution_summary.csv', index=False)
            fd_df.to_csv(output_dir / 'fd_per_run.csv', index=False)

    if (run_all or args.fd_only) and beta_df is not None and len(fd_df) > 0:
        fd_results = run_fd_sensitivity(beta_df, fd_df)
        if len(fd_results) > 0:
            fd_results.to_csv(output_dir / 'mc2_fd_sensitivity.csv', index=False)

    if (run_all or args.acf_only) and beta_df is not None:
        acf_df, residual_store = compute_residual_acf(beta_df)
        if len(acf_df) > 0:
            acf_df.to_csv(output_dir / 'mc4_residual_acf.csv', index=False)
            create_acf_figure(acf_df, residual_store, FIGURES)

    if (run_all or args.icc_ci_only) and beta_df is not None:
        icc_ci_df = bootstrap_icc_ci(beta_df, n_bootstrap=args.n_bootstrap)
        if len(icc_ci_df) > 0:
            icc_ci_df.to_csv(output_dir / 'mn3_icc_bootstrap_ci.csv', index=False)

    logger.info("\n" + "=" * 60)
    logger.info("SENSITIVITY ANALYSES COMPLETE")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
