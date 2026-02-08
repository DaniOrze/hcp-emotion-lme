import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from nilearn import plotting
from pathlib import Path

from paths import GROUP_LEVEL, LME_VOXELWISE, FIGURES


def load_results():
    results = {}
    
    # GLM
    glm_z = GROUP_LEVEL / "glm_zmap.nii.gz"
    if glm_z.exists():
        results['glm_z'] = nib.load(str(glm_z))
    
    # LME
    lme_z = LME_VOXELWISE / "lme_zmap.nii.gz"
    if lme_z.exists():
        results['lme_z'] = nib.load(str(lme_z))
    
    lme_fdr = LME_VOXELWISE / "lme_fdr05.nii.gz"
    if lme_fdr.exists():
        results['lme_fdr'] = nib.load(str(lme_fdr))
    
    return results


def compute_overlap(map1, map2, threshold=3.1):
    d1 = map1.get_fdata() > threshold
    d2 = map2.get_fdata() > threshold
    
    intersection = (d1 & d2).sum()
    union = d1.sum() + d2.sum()
    
    if union == 0:
        return 0
    
    dice = 2 * intersection / union
    return dice


def compute_correlation(map1, map2):
    d1 = map1.get_fdata().flatten()
    d2 = map2.get_fdata().flatten()
    
    # Only valid voxels
    valid = ~np.isnan(d1) & ~np.isnan(d2) & (d1 != 0) & (d2 != 0)
    
    r, p = stats.pearsonr(d1[valid], d2[valid])
    return r, p


def create_comparison_figure(results):
    """Create main comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # GLM z-map
    if 'glm_z' in results:
        plotting.plot_stat_map(
            results['glm_z'],
            title="GLM Group Analysis",
            threshold=3.1,
            display_mode='z',
            cut_coords=5,
            axes=axes[0, 0],
            colorbar=True
        )
    
    # LME z-map
    if 'lme_z' in results:
        plotting.plot_stat_map(
            results['lme_z'],
            title="LME Analysis",
            threshold=3.1,
            display_mode='z',
            cut_coords=5,
            axes=axes[0, 1],
            colorbar=True
        )
    
    if 'glm_z' in results and 'lme_z' in results:
        glm_data = results['glm_z'].get_fdata().flatten()
        lme_data = results['lme_z'].get_fdata().flatten()
        
        valid = ~np.isnan(glm_data) & ~np.isnan(lme_data) & (glm_data != 0) & (lme_data != 0)
        
        # Subsample for plotting
        n_plot = min(50000, valid.sum())
        idx = np.random.choice(np.where(valid)[0], n_plot, replace=False)
        
        axes[1, 0].scatter(glm_data[idx], lme_data[idx], alpha=0.1, s=1)
        axes[1, 0].plot([-10, 10], [-10, 10], 'r--', label='Identity')
        axes[1, 0].set_xlabel('GLM Z-score')
        axes[1, 0].set_ylabel('LME Z-score')
        axes[1, 0].set_title('Voxelwise Comparison')
        axes[1, 0].legend()
        
        r, _ = stats.pearsonr(glm_data[valid], lme_data[valid])
        axes[1, 0].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[1, 0].transAxes)
    
    if 'glm_z' in results and 'lme_z' in results:
        glm_data = results['glm_z'].get_fdata()
        lme_data = results['lme_z'].get_fdata()
        
        glm_valid = glm_data[~np.isnan(glm_data) & (glm_data != 0)]
        lme_valid = lme_data[~np.isnan(lme_data) & (lme_data != 0)]
        
        axes[1, 1].hist(glm_valid, bins=100, alpha=0.5, label='GLM', density=True)
        axes[1, 1].hist(lme_valid, bins=100, alpha=0.5, label='LME', density=True)
        axes[1, 1].axvline(3.1, color='r', linestyle='--', label='p<0.001')
        axes[1, 1].set_xlabel('Z-score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution of Z-scores')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES / 'glm_vs_lme_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / 'glm_vs_lme_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {FIGURES / 'glm_vs_lme_comparison.png'}")


def main():
    results = load_results()
    
    if len(results) < 2:
        print("Need both GLM and LME results to compare!")
        print(f"Found: {list(results.keys())}")
        return
    
    print("\n" + "="*50)
    print("GLM vs LME COMPARISON")
    print("="*50)
    
    if 'glm_z' in results and 'lme_z' in results:
        r, p = compute_correlation(results['glm_z'], results['lme_z'])
        print(f"\nVoxelwise correlation: r = {r:.4f}, p = {p:.2e}")
    
    print("\nDice overlap at different thresholds:")
    for thresh in [2.3, 3.1, 4.0]:
        dice = compute_overlap(results['glm_z'], results['lme_z'], thresh)
        print(f"  Z > {thresh}: Dice = {dice:.3f}")
    
    glm_data = results['glm_z'].get_fdata()
    lme_data = results['lme_z'].get_fdata()
    
    print(f"\nSignificant voxels (Z > 3.1):")
    print(f"  GLM: {(glm_data > 3.1).sum():,}")
    print(f"  LME: {(lme_data > 3.1).sum():,}")
    
    create_comparison_figure(results)
    
    stats_df = pd.DataFrame({
        'method': ['GLM', 'LME'],
        'n_sig_z31': [(glm_data > 3.1).sum(), (lme_data > 3.1).sum()],
        'max_z': [np.nanmax(glm_data), np.nanmax(lme_data)],
        'mean_z': [np.nanmean(glm_data[glm_data != 0]), np.nanmean(lme_data[lme_data != 0])]
    })
    stats_df.to_csv(FIGURES / 'comparison_stats.csv', index=False)
    
    print(f"\nStats saved to {FIGURES / 'comparison_stats.csv'}")


if __name__ == "__main__":
    main()
