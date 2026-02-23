import numpy as np
import pandas as pd


def compute_framewise_displacement(confounds_df):
    SPHERE_RADIUS = 50  # mm, for converting rotations to mm displacement

    trans_cols = ['trans_x', 'trans_y', 'trans_z']
    rot_cols = ['rot_x', 'rot_y', 'rot_z']

    deriv_cols = ['trans_x_der', 'trans_y_der', 'trans_z_der',
                  'rot_x_der', 'rot_y_der', 'rot_z_der']
    has_derivatives = all(c in confounds_df.columns for c in deriv_cols)

    if has_derivatives:
        trans_disp = confounds_df[['trans_x_der', 'trans_y_der', 'trans_z_der']].values
        rot_disp = confounds_df[['rot_x_der', 'rot_y_der', 'rot_z_der']].values
    else:
        trans = confounds_df[trans_cols].values
        rot = confounds_df[rot_cols].values

        trans_disp = np.diff(trans, axis=0, prepend=trans[:1])
        rot_disp = np.diff(rot, axis=0, prepend=rot[:1])

    rot_disp_mm = rot_disp * SPHERE_RADIUS

    fd = np.abs(trans_disp).sum(axis=1) + np.abs(rot_disp_mm).sum(axis=1)

    fd[0] = 0

    return fd


def check_motion_exclusion(fd_values, fd_threshold=0.5, fd_percent_threshold=0.20):
    n_frames = len(fd_values)
    n_above = (fd_values > fd_threshold).sum()
    percent_above = n_above / n_frames * 100 if n_frames > 0 else 0
    fraction_above = n_above / n_frames if n_frames > 0 else 0

    exclude = fraction_above > fd_percent_threshold

    stats = {
        'n_frames': int(n_frames),
        'mean_fd': float(np.mean(fd_values)),
        'median_fd': float(np.median(fd_values)),
        'max_fd': float(np.max(fd_values)),
        'n_above_threshold': int(n_above),
        'percent_above': float(percent_above),
        'fd_threshold': float(fd_threshold),
        'excluded': exclude,
    }

    return exclude, stats


def generate_motion_summary(first_level_dir, subject_ids):
    rows = []
    for subj in subject_ids:
        qc_path = first_level_dir / subj / "motion_qc.csv"
        if qc_path.exists():
            df = pd.read_csv(qc_path, index_col=0)
            for run in df.columns:
                row = df[run].to_dict()
                row['subject'] = subj
                row['run'] = run
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    return summary
