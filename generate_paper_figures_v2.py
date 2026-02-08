import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import nibabel as nib
from nilearn import plotting

from paths import GROUP_LEVEL, LME_VOXELWISE, FIGURES


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

COLORS = {
    'GLM': '#2ecc71',   # Green
    'LME': '#3498db',   # Blue
}


def _load_with_fallback(primary, fallback, label="file"):
    if primary.exists():
        logger.info(f"Loading {label} from {primary.name}")
        return primary
    elif fallback.exists():
        logger.warning(f"{primary.name} not found, falling back to {fallback.name}")
        return fallback
    else:
        return None


def figure1_methods_overview():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    glm_zmap = GROUP_LEVEL / "glm_zmap.nii.gz"
    if glm_zmap.exists():
        plotting.plot_stat_map(
            str(glm_zmap),
            threshold=3.1,
            display_mode='z',
            cut_coords=[-20, 0, 20, 40],
            title='A) GLM Group Analysis (Z > 3.1)',
            axes=ax1,
            colorbar=True,
            annotate=False
        )
    else:
        ax1.text(0.5, 0.5, 'GLM z-map not found\nRun main_group_glm.py', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('A) GLM Group Analysis')
    
    ax2 = fig.add_subplot(gs[0, 1])
    lme_zmap = LME_VOXELWISE / "lme_zmap.nii.gz"
    if lme_zmap.exists():
        plotting.plot_stat_map(
            str(lme_zmap),
            threshold=3.1,
            display_mode='z',
            cut_coords=[-20, 0, 20, 40],
            title='B) LME Analysis (Z > 3.1)',
            axes=ax2,
            colorbar=True,
            annotate=False
        )
    else:
        ax2.text(0.5, 0.5, 'LME z-map not found\nRun main_lme_voxelwise.py',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('B) LME Analysis')
    
    ax3 = fig.add_subplot(gs[1, 0])
    if glm_zmap.exists() and lme_zmap.exists():
        glm_data = nib.load(str(glm_zmap)).get_fdata().flatten()
        lme_data = nib.load(str(lme_zmap)).get_fdata().flatten()
        valid = ~np.isnan(glm_data) & ~np.isnan(lme_data) & (glm_data != 0) & (lme_data != 0)
        
        n_plot = min(30000, valid.sum())
        idx = np.random.choice(np.where(valid)[0], n_plot, replace=False)
        
        ax3.scatter(glm_data[idx], lme_data[idx], alpha=0.1, s=1, c='gray')
        ax3.plot([-8, 8], [-8, 8], 'r--', linewidth=2, label='Identity')
        ax3.set_xlabel('GLM Z-score')
        ax3.set_ylabel('LME Z-score')
        ax3.set_title('C) Voxelwise Z-score Comparison')
        
        from scipy import stats
        r, p = stats.pearsonr(glm_data[valid], lme_data[valid])
        ax3.text(0.05, 0.95, f'r = {r:.3f}', transform=ax3.transAxes, fontsize=11)
        ax3.set_xlim(-8, 8)
        ax3.set_ylim(-8, 8)
    else:
        ax3.text(0.5, 0.5, 'Run both GLM and LME first', ha='center', va='center')
        ax3.set_title('C) Voxelwise Comparison')
    
    ax4 = fig.add_subplot(gs[1, 1])
    if glm_zmap.exists() and lme_zmap.exists():
        glm_data = nib.load(str(glm_zmap)).get_fdata()
        lme_data = nib.load(str(lme_zmap)).get_fdata()
        
        glm_valid = glm_data[(~np.isnan(glm_data)) & (glm_data != 0)]
        lme_valid = lme_data[(~np.isnan(lme_data)) & (lme_data != 0)]
        
        ax4.hist(glm_valid, bins=100, alpha=0.6, label='GLM', color=COLORS['GLM'], density=True)
        ax4.hist(lme_valid, bins=100, alpha=0.6, label='LME', color=COLORS['LME'], density=True)
        ax4.axvline(3.1, color='red', linestyle='--', label='p < 0.001')
        ax4.axvline(-3.1, color='red', linestyle='--')
        ax4.set_xlabel('Z-score')
        ax4.set_ylabel('Density')
        ax4.set_title('D) Distribution of Z-scores')
        ax4.legend()
        ax4.set_xlim(-6, 6)
    else:
        ax4.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax4.set_title('D) Z-score Distribution')
    
    plt.savefig(FIGURES / 'figure1_brain_maps.png')
    plt.savefig(FIGURES / 'figure1_brain_maps.pdf')
    plt.close()
    logger.info("Saved Figure 1: Brain maps")


def figure2_roi_comparison():
    roi_path = _load_with_fallback(
        GROUP_LEVEL / 'roi_results_v2.csv',
        GROUP_LEVEL / 'roi_results.csv',
        "ROI results"
    )
    
    if roi_path is None:
        logger.warning("ROI results not found. Run main_roi_analysis_v2.py first.")
        return
    
    df = pd.read_csv(roi_path)
    
    df['roi_short'] = df['roi'].str.replace('cortical_', 'c:').str.replace('subcortical_', 's:')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    ax = axes[0, 0]
    pivot = df.pivot(index='roi_short', columns='method', values='cohens_d')
    pivot = pivot.sort_values('GLM')
    x = np.arange(len(pivot.index))
    width = 0.35
    
    ax.barh(x - width/2, pivot['GLM'], width, label='GLM', color=COLORS['GLM'])
    ax.barh(x + width/2, pivot['LME'], width, label='LME', color=COLORS['LME'])
    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Cohen's d")
    ax.set_title("A) Effect Sizes by ROI")
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.legend()
    
    ax = axes[0, 1]
    if 'p_fdr' in df.columns:
        pivot_p = df.pivot(index='roi_short', columns='method', values='p_fdr')
        pivot_p = pivot_p.reindex(pivot.index)  # same order
        pivot_p_log = -np.log10(pivot_p.clip(lower=1e-300))
        
        ax.barh(x - width/2, pivot_p_log['GLM'], width, label='GLM', color=COLORS['GLM'])
        ax.barh(x + width/2, pivot_p_log['LME'], width, label='LME', color=COLORS['LME'])
        ax.axvline(-np.log10(0.05), color='red', linestyle='--', label='q = 0.05')
        ax.set_xlabel('-log10(p FDR)')
    
    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("B) Statistical Significance")
    ax.legend()
    
    ax = axes[1, 0]
    lme_df = df[df['method'] == 'LME'].copy()
    if 'icc' in lme_df.columns:
        lme_df = lme_df.sort_values('icc', ascending=True)
        colors = ['#e74c3c' if v < 0.5 else '#f39c12' if v < 0.75 else '#27ae60' 
                  for v in lme_df['icc']]
        ax.barh(lme_df['roi_short'], lme_df['icc'], color=colors)
        ax.axvline(0.5, color='orange', linestyle='--', alpha=0.7)
        ax.axvline(0.75, color='green', linestyle='--', alpha=0.7)
        ax.set_xlabel('ICC')
        ax.set_title('C) Intraclass Correlation Coefficient')
        ax.set_xlim(0, 1)
        
        legend_elements = [
            Patch(facecolor='#27ae60', label='Good (>0.75)'),
            Patch(facecolor='#f39c12', label='Moderate (0.5-0.75)'),
            Patch(facecolor='#e74c3c', label='Poor (<0.5)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    
    ax = axes[1, 1]
    glm_df = df[df['method'] == 'GLM'].set_index('roi_short')
    lme_df2 = df[df['method'] == 'LME'].set_index('roi_short')
    common = glm_df.index.intersection(lme_df2.index)
    
    ax.scatter(glm_df.loc[common, 'cohens_d'], lme_df2.loc[common, 'cohens_d'], 
              s=100, c=COLORS['LME'], edgecolors='black')
    
    for roi in common:
        ax.annotate(roi, 
                   (glm_df.loc[roi, 'cohens_d'] + 0.02, 
                    lme_df2.loc[roi, 'cohens_d'] + 0.02),
                   fontsize=7)
    
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]) - 0.1,
            max(ax.get_xlim()[1], ax.get_ylim()[1]) + 0.1]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Identity')
    ax.set_xlabel("GLM Cohen's d")
    ax.set_ylabel("LME Cohen's d")
    ax.set_title("D) Effect Size Agreement")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES / 'figure2_roi_comparison.png')
    plt.savefig(FIGURES / 'figure2_roi_comparison.pdf')
    plt.close()
    logger.info("Saved Figure 2: ROI comparison")


def figure3_stability():
    stab_path = _load_with_fallback(
        GROUP_LEVEL / 'stability_results_v2.json',
        GROUP_LEVEL / 'stability_results.json',
        "Stability results"
    )
    
    if stab_path is None:
        logger.warning("Stability results not found. Run analysis_stability_v2.py first.")
        return
    
    with open(stab_path) as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    width = 0.35
    
    ax = axes[0]
    methods = ['GLM', 'LME']
    r_values = [results['split_half']['glm']['r_full'], 
                results['split_half']['lme']['r_full']]
    bars = ax.bar(methods, r_values, color=[COLORS['GLM'], COLORS['LME']])
    ax.axhline(0.7, color='orange', linestyle='--', alpha=0.7, label='Acceptable')
    ax.axhline(0.8, color='green', linestyle='--', alpha=0.7, label='Good')
    ax.axhline(0.9, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent')
    ax.set_ylabel('Reliability (Spearman-Brown)')
    ax.set_title('A) Split-Half Reliability')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    
    for bar, val in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=10)
    
    ax = axes[1]
    rois = list(results['bootstrap']['glm'].keys())[:5]
    roi_labels = [r.replace('cortical_', 'c:').replace('subcortical_', 's:') for r in rois]
    x = np.arange(len(rois))
    
    glm_means = [results['bootstrap']['glm'][r]['mean'] for r in rois]
    glm_ci_low = [results['bootstrap']['glm'][r]['ci_lower'] for r in rois]
    glm_ci_high = [results['bootstrap']['glm'][r]['ci_upper'] for r in rois]
    
    lme_means = [results['bootstrap']['lme'][r]['mean'] for r in rois]
    lme_ci_low = [results['bootstrap']['lme'][r]['ci_lower'] for r in rois]
    lme_ci_high = [results['bootstrap']['lme'][r]['ci_upper'] for r in rois]
    
    ax.bar(x - width/2, glm_means, width, label='GLM', color=COLORS['GLM'],
           yerr=[np.array(glm_means)-np.array(glm_ci_low),
                 np.array(glm_ci_high)-np.array(glm_means)], capsize=3)
    ax.bar(x + width/2, lme_means, width, label='LME', color=COLORS['LME'],
           yerr=[np.array(lme_means)-np.array(lme_ci_low),
                 np.array(lme_ci_high)-np.array(lme_means)], capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Effect (β)')
    ax.set_title('B) Bootstrap 95% Confidence Intervals')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.legend()
    
    ax = axes[2]
    rois_loo = list(results['loo'].keys())
    roi_labels_loo = [r.replace('cortical_', 'c:').replace('subcortical_', 's:') for r in rois_loo]
    glm_cv = [results['loo'][r]['glm']['cv'] for r in rois_loo]
    lme_cv = [results['loo'][r]['lme']['cv'] for r in rois_loo]
    
    x = np.arange(len(rois_loo))
    ax.bar(x - width/2, glm_cv, width, label='GLM', color=COLORS['GLM'])
    ax.bar(x + width/2, lme_cv, width, label='LME', color=COLORS['LME'])
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels_loo, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('C) Leave-One-Out Stability')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES / 'figure3_stability.png')
    plt.savefig(FIGURES / 'figure3_stability.pdf')
    plt.close()
    logger.info("Saved Figure 3: Stability analysis")


def figure4_computational():
    benchmark_file = GROUP_LEVEL / 'benchmark_results.csv'
    
    if not benchmark_file.exists():
        logger.warning("Benchmark results not found. Run analysis_benchmark.py first.")
        return
    
    df = pd.read_csv(benchmark_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # A) Time scaling with subjects
    ax = axes[0]
    for method in ['GLM', 'LME']:
        data = df[(df['method'] == method) & (df['n_voxels'] == 1000)]
        if not data.empty:
            grouped = data.groupby('n_subjects')['time_sec'].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       label=method, marker='o', capsize=3, color=COLORS[method])
    ax.set_xlabel('Number of Subjects')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('A) Computation Time vs Sample Size')
    ax.legend()
    ax.set_yscale('log')
    
    # B) Full-brain projection
    ax = axes[1]
    
    glm_per_vox = df[(df['method'] == 'GLM') & (df['time_per_voxel_ms'].notna())]['time_per_voxel_ms'].mean()
    lme_per_vox = df[(df['method'] == 'LME') & (df['time_per_voxel_ms'].notna())]['time_per_voxel_ms'].mean()
    
    if pd.notna(glm_per_vox) and pd.notna(lme_per_vox):
        n_voxels = 150000
        glm_hours = glm_per_vox * n_voxels / 1000 / 3600
        lme_hours = lme_per_vox * n_voxels / 1000 / 3600
        
        bars = ax.bar(['GLM', 'LME'], [glm_hours, lme_hours], 
                     color=[COLORS['GLM'], COLORS['LME']])
        ax.set_ylabel('Estimated Time (hours)')
        ax.set_title(f'B) Projected Full-Brain Time\n({n_voxels:,} voxels)')
        
        for bar, val in zip(bars, [glm_hours, lme_hours]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.1f}h', ha='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(FIGURES / 'figure4_computational.png')
    plt.savefig(FIGURES / 'figure4_computational.pdf')
    plt.close()
    logger.info("Saved Figure 4: Computational benchmark")


def generate_summary_table():
    summary_data = []
    
    roi_path = _load_with_fallback(
        GROUP_LEVEL / 'roi_results_v2.csv',
        GROUP_LEVEL / 'roi_results.csv',
        "ROI results"
    )
    if roi_path:
        roi_df = pd.read_csv(roi_path)
        
        for method in ['GLM', 'LME']:
            method_df = roi_df[roi_df['method'] == method]
            row = {
                'Analysis': f'{method} - ROI',
                'Mean Effect': f"{method_df['mean_beta'].mean():.3f}",
                'Mean Cohen\'s d': f"{method_df['cohens_d'].abs().mean():.3f}",
                'Sig. ROIs (p<0.05)': int((method_df['p_value'] < 0.05).sum()),
                'Sig. ROIs (FDR<0.05)': int((method_df['p_fdr'] < 0.05).sum()) if 'p_fdr' in method_df else 'N/A'
            }
            if method == 'LME' and 'icc' in method_df.columns:
                row['Mean ICC'] = f"{method_df['icc'].mean():.3f}"
            summary_data.append(row)
    
    stab_path = _load_with_fallback(
        GROUP_LEVEL / 'stability_results_v2.json',
        GROUP_LEVEL / 'stability_results.json',
        "Stability results"
    )
    if stab_path:
        with open(stab_path) as f:
            stab = json.load(f)
        
        summary_data.append({
            'Analysis': 'Split-Half Reliability',
            'GLM': f"{stab['split_half']['glm']['r_full']:.3f}",
            'LME': f"{stab['split_half']['lme']['r_full']:.3f}"
        })
        
        # LOO CV
        rois_loo = list(stab['loo'].keys())
        glm_cv = np.nanmean([stab['loo'][r]['glm']['cv'] for r in rois_loo])
        lme_cv = np.nanmean([stab['loo'][r]['lme']['cv'] for r in rois_loo])
        summary_data.append({
            'Analysis': 'LOO CV (mean)',
            'GLM': f"{glm_cv:.4f}",
            'LME': f"{lme_cv:.4f}"
        })
    
    # Voxelwise comparison
    voxel_glm_z = GROUP_LEVEL / "glm_zmap.nii.gz"
    if voxel_glm_z.exists():
        glm_data = nib.load(str(voxel_glm_z)).get_fdata()
        glm_sig = np.nansum(glm_data > 3.1)
        
        lme_zmap = LME_VOXELWISE / "lme_zmap.nii.gz"
        if lme_zmap.exists():
            lme_data = nib.load(str(lme_zmap)).get_fdata()
            lme_sig = np.nansum(lme_data > 3.1)
            
            summary_data.append({
                'Analysis': 'Voxelwise (Z > 3.1)',
                'GLM': f"{int(glm_sig):,}",
                'LME': f"{int(lme_sig):,}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(FIGURES / 'summary_table.csv', index=False)
    
    latex = summary_df.to_latex(index=False, escape=False)
    with open(FIGURES / 'summary_table.tex', 'w') as f:
        f.write(latex)
    
    logger.info("Saved summary table")
    return summary_df


def main():
    logger.info("="*50)
    logger.info("GENERATING PAPER FIGURES (v2)")
    logger.info("="*50)
    
    FIGURES.mkdir(parents=True, exist_ok=True)
    
    figure1_methods_overview()
    figure2_roi_comparison()
    figure3_stability()
    figure4_computational()
    
    summary = generate_summary_table()
    print("\nSummary Table:")
    print(summary.to_string())
    
    logger.info(f"\nAll figures saved to: {FIGURES}")
    logger.info("Files generated:")
    for f in sorted(FIGURES.glob("*.png")):
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()