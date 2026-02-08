import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path


def load_nifti(path):
    img = nib.load(str(path))
    return img, img.get_fdata()


def save_nifti(data, affine, header, out_path):
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(out_path))


def load_hcp_emotion_events(ev_dir):
    ev_dir = Path(ev_dir)
    events = []
    
    for cond in ["fear", "neut"]:
        ev_path = ev_dir / f"{cond}.txt"
        
        if not ev_path.exists():
            raise FileNotFoundError(f"Event file not found: {ev_path}")
        
        df = pd.read_csv(
            ev_path,
            sep=r"\s+",
            header=None,
            names=["onset", "duration", "amplitude"]
        )
        df["trial_type"] = cond
        events.append(df[["onset", "duration", "trial_type"]])
    
    return pd.concat(events, ignore_index=True).sort_values("onset")


def load_confounds(path):
    cols = [
        'trans_x', 'trans_y', 'trans_z',
        'rot_x', 'rot_y', 'rot_z',
        'trans_x_der', 'trans_y_der', 'trans_z_der',
        'rot_x_der', 'rot_y_der', 'rot_z_der'
    ]
    
    data = np.loadtxt(str(path))
    
    # HCP has 12 columns
    if data.shape[1] >= 12:
        return pd.DataFrame(data[:, :12], columns=cols)
    else:
        # Pad with zeros if fewer columns
        return pd.DataFrame(data, columns=cols[:data.shape[1]])
