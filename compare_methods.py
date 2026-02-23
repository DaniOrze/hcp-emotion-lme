import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from nilearn import plotting
from pathlib import Path

from config import Z_THRESHOLD, FDR_ALPHA, SMOOTH_FWHM
from paths import GROUP_LEVEL, LME_VOXELWISE, FIGURES


def load_results():
    results = {}

    for name, path in [
        ('glm_z', GROUP_LEVEL / "glm_zmap.nii.gz"),
        ('glm_fdr', GROUP_LEVEL / "glm_fdr05.nii.gz"),
        ('glm_p', GROUP_LEVEL / "glm_pmap.nii.gz"),
    ]:
        if path.exists():
            results[name] = nib.load(str(path))

    for name, path in [
        ('lme_z', LME_VOXELWISE / "lme_zmap.nii.gz"),
        ('lme_fdr', LME_VOXELWISE / "lme_fdr05.nii.gz"),
        ('lme_p', LME_VOXELWISE / "lme_pmap.nii.gz"),
        ('lme_icc', LME_VOXELWISE / "lme_icc.nii.gz"),
    ]:
        if path.exists():
            results[name] = nib.load(str(path))

    for name, path in [
        ('lme_norun_z', LME_VOXELWISE / "lme_norun_zmap.nii.gz"),
        ('lme_norun_fdr', LME_VOXELWISE / "lme_norun_fdr05.nii.gz"),
    ]:
        if path.exists():
            results[name] = nib.load(str(path))

    return results


def compute_overlap_dice(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 0
    return 2 * intersection / total


def compute_equalized_comparison(results):
    print("\n" + "=" * 60)
    print("EQUALIZED THRESHOLD COMPARISON (MC1 FIX)")
    print("=" * 60)

    comparison = {}

    if 'glm_fdr' in results and 'lme_fdr' in results:
        glm_fdr = results['glm_fdr'].get_fdata().astype(bool)
        lme_fdr = results['lme_fdr'].get_fdata().astype(bool)

        n_glm_fdr = glm_fdr.sum()
        n_lme_fdr = lme_fdr.sum()
        dice_fdr = compute_overlap_dice(glm_fdr, lme_fdr)

        print(f"\n1. FDR q < {FDR_ALPHA} (PRIMARY — equalized comparison):")
        print(f"   GLM significant voxels:  {n_glm_fdr:>8,}")
        print(f"   LME significant voxels:  {n_lme_fdr:>8,}")
        print(f"   Ratio (GLM/LME):         {n_glm_fdr / n_lme_fdr:.2f}x" if n_lme_fdr > 0 else "   Ratio: N/A")
        print(f"   Dice overlap:            {dice_fdr:.3f}")
        print(f"   Both significant:        {(glm_fdr & lme_fdr).sum():>8,}")
        print(f"   GLM only:                {(glm_fdr & ~lme_fdr).sum():>8,}")
        print(f"   LME only:                {(~glm_fdr & lme_fdr).sum():>8,}")

        comparison['fdr'] = {
            'glm_n_sig': int(n_glm_fdr),
            'lme_n_sig': int(n_lme_fdr),
            'dice': float(dice_fdr),
            'both': int((glm_fdr & lme_fdr).sum()),
            'glm_only': int((glm_fdr & ~lme_fdr).sum()),
            'lme_only': int((~glm_fdr & lme_fdr).sum()),
        }

    if 'glm_z' in results and 'lme_z' in results:
        glm_z = results['glm_z'].get_fdata()
        lme_z = results['lme_z'].get_fdata()

        glm_sig_z = glm_z > Z_THRESHOLD
        lme_sig_z = lme_z > Z_THRESHOLD

        n_glm_z = glm_sig_z.sum()
        n_lme_z = lme_sig_z.sum()
        dice_z = compute_overlap_dice(glm_sig_z, lme_sig_z)

        print(f"\n2. Z > {Z_THRESHOLD} uncorrected (SECONDARY — matched threshold):")
        print(f"   GLM significant voxels:  {n_glm_z:>8,}")
        print(f"   LME significant voxels:  {n_lme_z:>8,}")
        print(f"   Ratio (GLM/LME):         {n_glm_z / n_lme_z:.2f}x" if n_lme_z > 0 else "   Ratio: N/A")
        print(f"   Dice overlap:            {dice_z:.3f}")

        comparison['z_threshold'] = {
            'glm_n_sig': int(n_glm_z),
            'lme_n_sig': int(n_lme_z),
            'dice': float(dice_z),
        }

    if 'glm_z' in results and 'lme_z' in results:
        glm_flat = glm_z.flatten()
        lme_flat = lme_z.flatten()
        valid = (~np.isnan(glm_flat) & ~np.isnan(lme_flat) &
                 (glm_flat != 0) & (lme_flat != 0))

        r, p = stats.pearsonr(glm_flat[valid], lme_flat[valid])
        print(f"\n3. Voxelwise correlation:")
        print(f"   Pearson r = {r:.4f} (p = {p:.2e})")
        print(f"   N valid voxels: {valid.sum():,}")

        comparison['correlation'] = {'r': float(r), 'p': float(p), 'n_valid': int(valid.sum())}

    if 'glm_z' in results and 'lme_z' in results:
        print(f"\n4. Dice overlap at matched Z-thresholds:")
        for thresh in [2.0, 2.3, 3.1, 4.0, 5.0]:
            dice = compute_overlap_dice(glm_z > thresh, lme_z > thresh)
            print(f"   Z > {thresh}: Dice = {dice:.3f}")

    if 'lme_norun_z' in results:
        lme_norun_z = results['lme_norun_z'].get_fdata()
        print(f"\n5. INTERMEDIATE MODEL: LME without run effect (Major Concern 2):")
        print(f"   Model: beta ~ 1 + (1|subject)  [no run fixed effect]")

        if 'lme_norun_fdr' in results:
            lme_norun_fdr = results['lme_norun_fdr'].get_fdata().astype(bool)
            n_lme_norun_fdr = lme_norun_fdr.sum()
            print(f"   LME-no-run sig voxels (FDR q<{FDR_ALPHA}): {n_lme_norun_fdr:>8,}")

            if 'glm_fdr' in results:
                glm_fdr_data = results['glm_fdr'].get_fdata().astype(bool)
                n_glm_fdr = glm_fdr_data.sum()
                dice_norun_glm = compute_overlap_dice(glm_fdr_data, lme_norun_fdr)
                print(f"   GLM sig voxels (FDR q<{FDR_ALPHA}):         {n_glm_fdr:>8,}")
                print(f"   Dice(GLM, LME-no-run):                {dice_norun_glm:.3f}")

            if 'lme_fdr' in results:
                lme_fdr_data = results['lme_fdr'].get_fdata().astype(bool)
                n_lme_fdr = lme_fdr_data.sum()
                dice_norun_full = compute_overlap_dice(lme_fdr_data, lme_norun_fdr)
                print(f"   LME (full) sig voxels (FDR q<{FDR_ALPHA}):  {n_lme_fdr:>8,}")
                print(f"   Dice(LME-full, LME-no-run):           {dice_norun_full:.3f}")

            comparison['lme_norun'] = {
                'n_sig_fdr': int(n_lme_norun_fdr),
            }

        if 'glm_z' in results:
            glm_flat = results['glm_z'].get_fdata().flatten()
            norun_flat = lme_norun_z.flatten()
            valid = (~np.isnan(glm_flat) & ~np.isnan(norun_flat) &
                     (glm_flat != 0) & (norun_flat != 0))
            if valid.sum() > 10:
                r_norun, _ = stats.pearsonr(glm_flat[valid], norun_flat[valid])
                print(f"   Correlation(GLM, LME-no-run):         r = {r_norun:.4f}")

        print(f"\n   INTERPRETATION: Comparing GLM vs LME-no-run isolates the effect")
        print(f"   of hierarchical variance decomposition. Comparing LME-no-run vs")
        print(f"   LME-full isolates the effect of including the run covariate.")

    return comparison


def create_comparison_figure(results):
    fig = plt.figure(figsize=(16, 14))

    ax1 = fig.add_axes([0.02, 0.75, 0.46, 0.22])
    ax2 = fig.add_axes([0.52, 0.75, 0.46, 0.22])

    vmax = 10

    if 'glm_z' in results:
        plotting.plot_stat_map(
            results['glm_z'],
            title='A) GLM (FDR q<0.05)',
            threshold=Z_THRESHOLD,
            display_mode='ortho',
            cut_coords=(0, -20, 10),
            axes=ax1,
            colorbar=True,
            annotate=True,
            vmax=vmax,
        )

    if 'lme_z' in results:
        plotting.plot_stat_map(
            results['lme_z'],
            title='B) LME (FDR q<0.05)',
            threshold=Z_THRESHOLD,
            display_mode='ortho',
            cut_coords=(0, -20, 10),
            axes=ax2,
            colorbar=True,
            annotate=True,
            vmax=vmax,
        )

    ax3 = fig.add_axes([0.02, 0.52, 0.46, 0.20])
    ax4 = fig.add_axes([0.52, 0.52, 0.46, 0.20])

    if 'glm_z' in results:
        plotting.plot_stat_map(
            results['glm_z'],
            threshold=Z_THRESHOLD,
            display_mode='z',
            cut_coords=[-25, -10, 0, 10, 25, 45],
            axes=ax3,
            colorbar=False,
            annotate=True,
            vmax=vmax,
        )
        ax3.set_title('C) GLM axial slices', fontsize=10)

    if 'lme_z' in results:
        plotting.plot_stat_map(
            results['lme_z'],
            threshold=Z_THRESHOLD,
            display_mode='z',
            cut_coords=[-25, -10, 0, 10, 25, 45],
            axes=ax4,
            colorbar=False,
            annotate=True,
            vmax=vmax,
        )
        ax4.set_title('D) LME axial slices', fontsize=10)

    ax5 = fig.add_subplot(2, 2, 3)
    if 'glm_z' in results and 'lme_z' in results:
        glm_data = results['glm_z'].get_fdata().flatten()
        lme_data = results['lme_z'].get_fdata().flatten()
        valid = (~np.isnan(glm_data) & ~np.isnan(lme_data) &
                 (glm_data != 0) & (lme_data != 0))

        n_plot = min(50000, valid.sum())
        idx = np.random.choice(np.where(valid)[0], n_plot, replace=False)

        ax5.scatter(glm_data[idx], lme_data[idx], alpha=0.1, s=1, c='gray')
        ax5.plot([-10, 10], [-10, 10], 'r--', linewidth=2, label='Identity')
        ax5.set_xlabel('GLM Z-score')
        ax5.set_ylabel('LME Z-score')
        ax5.set_title('E) Voxelwise Z-score Comparison')

        r, _ = stats.pearsonr(glm_data[valid], lme_data[valid])
        ax5.text(0.05, 0.95, f'r = {r:.3f}', transform=ax5.transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax5.set_xlim(-10, 10)
        ax5.set_ylim(-10, 10)
        ax5.legend()

    ax6 = fig.add_subplot(2, 2, 4)
    if 'glm_z' in results and 'lme_z' in results:
        glm_data = results['glm_z'].get_fdata()
        lme_data = results['lme_z'].get_fdata()

        glm_valid = glm_data[(~np.isnan(glm_data)) & (glm_data != 0)]
        lme_valid = lme_data[(~np.isnan(lme_data)) & (lme_data != 0)]

        ax6.hist(glm_valid, bins=100, alpha=0.5, label='GLM', color='#2ecc71', density=True)
        ax6.hist(lme_valid, bins=100, alpha=0.5, label='LME', color='#3498db', density=True)
        ax6.axvline(Z_THRESHOLD, color='red', linestyle='--', label=f'Z={Z_THRESHOLD}')
        ax6.axvline(-Z_THRESHOLD, color='red', linestyle='--')
        ax6.set_xlabel('Z-score')
        ax6.set_ylabel('Density')
        ax6.set_title('F) Distribution of Z-scores')
        ax6.legend()
        ax6.set_xlim(-8, 8)

        if 'glm_fdr' in results and 'lme_fdr' in results:
            n_glm_fdr = results['glm_fdr'].get_fdata().astype(bool).sum()
            n_lme_fdr = results['lme_fdr'].get_fdata().astype(bool).sum()
            ax6.text(0.95, 0.95,
                     f'FDR q<0.05:\nGLM: {n_glm_fdr:,}\nLME: {n_lme_fdr:,}',
                     transform=ax6.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(FIGURES / 'glm_vs_lme_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / 'glm_vs_lme_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {FIGURES / 'glm_vs_lme_comparison.png'}")


def main():
    results = load_results()

    if 'glm_z' not in results or 'lme_z' not in results:
        print("Need both GLM and LME results to compare!")
        print(f"Found: {list(results.keys())}")
        return

    comparison = compute_equalized_comparison(results)

    pd.DataFrame([comparison]).to_json(FIGURES / 'comparison_stats.json', indent=2)

    create_comparison_figure(results)

    summary_rows = []
    if 'fdr' in comparison:
        summary_rows.append({
            'threshold': f'FDR q<{FDR_ALPHA}',
            'glm_n_sig': comparison['fdr']['glm_n_sig'],
            'lme_n_sig': comparison['fdr']['lme_n_sig'],
            'dice': comparison['fdr']['dice'],
        })
    if 'z_threshold' in comparison:
        summary_rows.append({
            'threshold': f'Z>{Z_THRESHOLD}',
            'glm_n_sig': comparison['z_threshold']['glm_n_sig'],
            'lme_n_sig': comparison['z_threshold']['lme_n_sig'],
            'dice': comparison['z_threshold']['dice'],
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(FIGURES / 'comparison_stats.csv', index=False)
        print(f"\nComparison table saved to {FIGURES / 'comparison_stats.csv'}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
