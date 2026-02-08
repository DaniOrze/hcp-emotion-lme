import time
import logging
from datetime import datetime
import tracemalloc
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

from paths import GROUP_LEVEL, FIGURES, LOGS

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / f"benchmark_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_subjects, n_voxels):
    """Generate synthetic fMRI-like data for benchmarking."""
    # Random effect (between-subject variance)
    subject_effects = np.random.normal(0, 0.5, n_subjects)
    
    # Fixed effect (true activation)
    true_beta = 0.3
    
    # Generate data
    data = []
    for i, subj in enumerate(range(n_subjects)):
        betas = true_beta + subject_effects[i] + np.random.normal(0, 1, n_voxels)
        for v, beta in enumerate(betas):
            data.append({'subject': f'sub_{subj:03d}', 'voxel': v, 'beta': beta})
    
    return pd.DataFrame(data)


def benchmark_glm(betas):
    """Benchmark GLM (one-sample t-test)."""
    tracemalloc.start()
    start_time = time.perf_counter()
    
    t_stat, p_val = stats.ttest_1samp(betas, 0)
    
    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'time_sec': elapsed,
        'memory_mb': peak / 1024 / 1024,
        't_stat': t_stat,
        'p_value': p_val,
        'converged': True
    }


def benchmark_lme(df, max_attempts=3):
    """
    Benchmark LME with robust error handling.
    """
    tracemalloc.start()
    start_time = time.perf_counter()
    
    converged = False
    result = None
    
    # Try different optimization methods
    methods = ['lbfgs', 'bfgs', 'cg']
    
    for attempt, method in enumerate(methods):
        if attempt >= max_attempts:
            break
            
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = MixedLM.from_formula("beta ~ 1", groups="subject", data=df)
                result = model.fit(reml=True, method=method, maxiter=100)
                converged = True
                break
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug(f"LME fit failed with method '{method}': {e}")
            continue
    
    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    if not converged or result is None:
        logger.warning("LME failed to converge, returning NaN")
        return {
            'time_sec': elapsed,
            'memory_mb': peak / 1024 / 1024,
            't_stat': np.nan,
            'p_value': np.nan,
            'converged': False
        }
    
    return {
        'time_sec': elapsed,
        'memory_mb': peak / 1024 / 1024,
        't_stat': result.tvalues['Intercept'],
        'p_value': result.pvalues['Intercept'],
        'converged': True
    }


def run_scaling_benchmark(subject_counts, voxel_counts, n_repeats=3):
    results = []
    
    # Scale with subjects (fixed voxels)
    logger.info("Benchmarking scaling with subjects...")
    n_voxels_fixed = 1000
    
    for n_subj in subject_counts:
        logger.info(f"  N subjects = {n_subj}")
        
        for rep in range(n_repeats):
            df = generate_synthetic_data(n_subj, n_voxels_fixed)
            
            # GLM: just use mean across subjects
            betas = df.groupby('subject')['beta'].mean().values
            glm_result = benchmark_glm(betas)
            glm_result.update({
                'method': 'GLM',
                'n_subjects': n_subj,
                'n_voxels': n_voxels_fixed,
                'replicate': rep
            })
            results.append(glm_result)
            
            # LME: single voxel
            single_voxel = df[df['voxel'] == 0]
            lme_result = benchmark_lme(single_voxel)
            lme_result.update({
                'method': 'LME',
                'n_subjects': n_subj,
                'n_voxels': n_voxels_fixed,
                'replicate': rep
            })
            results.append(lme_result)
    
    # Scale with voxels (fixed subjects)
    logger.info("\nBenchmarking scaling with voxels...")
    n_subj_fixed = 100
    
    for n_vox in voxel_counts:
        logger.info(f"  N voxels = {n_vox}")
        
        df = generate_synthetic_data(n_subj_fixed, n_vox)
        
        # Time to process all voxels with GLM
        start_glm = time.perf_counter()
        for v in range(n_vox):
            vox_betas = df[df['voxel'] == v].groupby('subject')['beta'].first().values
            _ = stats.ttest_1samp(vox_betas, 0)
        glm_total = time.perf_counter() - start_glm
        
        results.append({
            'method': 'GLM',
            'n_subjects': n_subj_fixed,
            'n_voxels': n_vox,
            'time_sec': glm_total,
            'time_per_voxel_ms': glm_total / n_vox * 1000,
            'replicate': 0,
            'converged': True
        })
        
        # Time to process subset of voxels with LME (with error handling)
        start_lme = time.perf_counter()
        n_lme_voxels = min(n_vox, 50)
        lme_success = 0
        
        for v in range(n_lme_voxels):
            vox_df = df[df['voxel'] == v]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model = MixedLM.from_formula("beta ~ 1", groups="subject", data=vox_df)
                    _ = model.fit(reml=True, method="lbfgs", maxiter=50)
                    lme_success += 1
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.debug(f"LME failed for voxel {v}: {e}")
                continue
        
        lme_total = time.perf_counter() - start_lme
        
        # Only extrapolate if we had successful fits
        if lme_success > 0:
            lme_per_voxel = lme_total / lme_success
            lme_extrapolated = lme_per_voxel * n_vox
            convergence_rate = lme_success / n_lme_voxels
        else:
            lme_extrapolated = np.nan
            lme_per_voxel = np.nan
            convergence_rate = 0.0
        
        results.append({
            'method': 'LME',
            'n_subjects': n_subj_fixed,
            'n_voxels': n_vox,
            'time_sec': lme_extrapolated,
            'time_per_voxel_ms': lme_per_voxel * 1000 if not np.isnan(lme_per_voxel) else np.nan,
            'replicate': 0,
            'converged': True,
            'convergence_rate': convergence_rate
        })
        
        if lme_success < n_lme_voxels:
            logger.warning(f"LME convergence rate: {convergence_rate:.1%} ({lme_success}/{n_lme_voxels})")
    
    return pd.DataFrame(results)


def create_benchmark_figure(results_df, output_dir):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Time vs N subjects
    ax = axes[0, 0]
    for method in ['GLM', 'LME']:
        data = results_df[(results_df['method'] == method) & 
                          (results_df['n_voxels'] == 1000) &
                          (results_df['converged'] == True)]
        if len(data) > 0:
            grouped = data.groupby('n_subjects')['time_sec'].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                       label=method, marker='o', capsize=3)
    ax.set_xlabel('Number of Subjects')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time vs Sample Size')
    ax.legend()
    ax.set_yscale('log')
    
    # 2. Time per voxel
    ax = axes[0, 1]
    voxel_data = results_df[(results_df['n_subjects'] == 100) & 
                            (results_df['converged'] == True)]
    for method in ['GLM', 'LME']:
        data = voxel_data[voxel_data['method'] == method]
        if len(data) > 0 and not data['time_per_voxel_ms'].isna().all():
            ax.plot(data['n_voxels'], data['time_per_voxel_ms'], 
                   label=method, marker='o')
    ax.set_xlabel('Number of Voxels')
    ax.set_ylabel('Time per Voxel (ms)')
    ax.set_title('Per-Voxel Processing Time')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 3. Projected full-brain time
    ax = axes[1, 0]
    n_voxels_brain = 150000  # Typical brain mask
    n_subjects_typical = 143  # HCP sample
    
    # Estimate from data (filter out NaN values)
    glm_data = results_df[(results_df['method'] == 'GLM') & 
                          (results_df['n_voxels'] > 100) &
                          (results_df['converged'] == True)]
    lme_data = results_df[(results_df['method'] == 'LME') & 
                          (results_df['n_voxels'] > 100) &
                          (results_df['converged'] == True)]
    
    if len(glm_data) > 0:
        glm_per_vox = glm_data['time_per_voxel_ms'].dropna().mean()
        glm_total_hours = glm_per_vox * n_voxels_brain / 1000 / 3600
    else:
        glm_per_vox = np.nan
        glm_total_hours = np.nan
    
    if len(lme_data) > 0:
        lme_per_vox = lme_data['time_per_voxel_ms'].dropna().mean()
        lme_total_hours = lme_per_vox * n_voxels_brain / 1000 / 3600
    else:
        lme_per_vox = np.nan
        lme_total_hours = np.nan
    
    if not np.isnan(glm_total_hours) and not np.isnan(lme_total_hours):
        bars = ax.bar(['GLM', 'LME'], [glm_total_hours, lme_total_hours])
        ax.set_ylabel('Estimated Time (hours)')
        ax.set_title(f'Full-Brain Analysis Time\n({n_voxels_brain:,} voxels, {n_subjects_typical} subjects)')
        
        for bar, val in zip(bars, [glm_total_hours, lme_total_hours]):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2, height + height*0.05,
                       f'{val:.1f}h', ha='center')
    
    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    if not np.isnan(glm_per_vox) and not np.isnan(lme_per_vox):
        speedup_ratio = lme_per_vox / glm_per_vox
        summary = f"""
    COMPUTATIONAL BENCHMARK SUMMARY
    
    Configuration:
      • Full-brain voxels: {n_voxels_brain:,}
      • Subjects: {n_subjects_typical}
    
    Time per voxel:
      • GLM: {glm_per_vox:.3f} ms
      • LME: {lme_per_vox:.3f} ms
      • Ratio: {speedup_ratio:.0f}x slower
    
    Estimated total time:
      • GLM: {glm_total_hours:.2f} hours ({glm_total_hours*60:.0f} min)
      • LME: {lme_total_hours:.2f} hours
    
    Recommendations:
      • ROI analysis: Use LME (seconds)
      • Voxelwise: Use chunking + parallel
      • Consider n_jobs = {os.cpu_count() or 4} cores
        """
    else:
        summary = """
    COMPUTATIONAL BENCHMARK SUMMARY
    
    Warning: Insufficient data for complete summary.
    Some LME models failed to converge.
    
    Check logs for convergence issues.
        """
    
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'benchmark_results.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Figure saved to {output_dir / 'benchmark_results.png'}")
    
    return {
        'glm_per_voxel_ms': glm_per_vox,
        'lme_per_voxel_ms': lme_per_vox,
        'glm_total_hours': glm_total_hours,
        'lme_total_hours': lme_total_hours,
        'speedup_ratio': lme_per_vox / glm_per_vox if not np.isnan(lme_per_vox) and not np.isnan(glm_per_vox) else np.nan
    }


def main():
    logger.info("="*50)
    logger.info("COMPUTATIONAL BENCHMARK")
    logger.info("="*50)
    
    logger.info(f"\nSystem info:")
    logger.info(f"  CPUs: {os.cpu_count()}")
    
    if HAS_PSUTIL:
        logger.info(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    else:
        logger.info(f"  RAM: (install psutil for info)")
    
    # Run benchmarks
    subject_counts = [20, 50, 100, 150, 200]
    voxel_counts = [100, 500, 1000, 5000, 10000]
    
    results_df = run_scaling_benchmark(subject_counts, voxel_counts)
    
    # Save raw results
    results_df.to_csv(GROUP_LEVEL / 'benchmark_results.csv', index=False)
    logger.info(f"\nRaw results saved to {GROUP_LEVEL / 'benchmark_results.csv'}")
    
    # Create figure and get summary
    summary = create_benchmark_figure(results_df, FIGURES)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*50)
    
    if not np.isnan(summary['glm_per_voxel_ms']):
        logger.info(f"\nTime per voxel:")
        logger.info(f"  GLM: {summary['glm_per_voxel_ms']:.3f} ms")
        
        if not np.isnan(summary['lme_per_voxel_ms']):
            logger.info(f"  LME: {summary['lme_per_voxel_ms']:.3f} ms")
            logger.info(f"  LME is {summary['speedup_ratio']:.0f}x slower")
        else:
            logger.warning(f"  LME: Failed to converge reliably")
        
        logger.info(f"\nEstimated full-brain time (150k voxels):")
        logger.info(f"  GLM: {summary['glm_total_hours']:.2f} hours")
        
        if not np.isnan(summary['lme_total_hours']):
            logger.info(f"  LME: {summary['lme_total_hours']:.2f} hours")
    else:
        logger.error("Benchmark failed - insufficient valid results")
    
    # Report convergence rates
    lme_results = results_df[results_df['method'] == 'LME']
    if 'convergence_rate' in lme_results.columns:
        avg_convergence = lme_results['convergence_rate'].dropna().mean()
        if not np.isnan(avg_convergence):
            logger.info(f"\nLME average convergence rate: {avg_convergence:.1%}")


if __name__ == "__main__":
    main()