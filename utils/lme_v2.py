import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import logging

logger = logging.getLogger(__name__)


def fit_lme_single_voxel(beta_values, run_info_df, method="powell", reml=True, max_iter=200):
    beta_values = np.asarray(beta_values, dtype=np.float64)
    
    # Basic validation
    if len(beta_values) < 10:
        return _failed_result("Too few observations")
    
    # Check for NaN
    valid_mask = np.isfinite(beta_values)
    if valid_mask.sum() < 10:
        return _failed_result("Too few valid observations")
    
    # Remove NaN observations
    beta_clean = beta_values[valid_mask]
    subjects_clean = run_info_df['subject'].values[valid_mask]
    
    if 'run' in run_info_df.columns:
        runs_clean = run_info_df['run'].values[valid_mask]
    else:
        runs_clean = None
    
    # Check we have multiple subjects with data
    unique_subjects = pd.Series(subjects_clean).nunique()
    if unique_subjects < 5:
        return _failed_result("Too few subjects")
    
    # Check for near-zero variance
    if np.std(beta_clean) < 1e-8:
        return _failed_result("Zero variance")
    
    # Create DataFrame
    df = pd.DataFrame({
        "beta": beta_clean,
        "subject": subjects_clean
    })
    
    if runs_clean is not None and len(np.unique(runs_clean)) >= 2:
        df['run'] = pd.Categorical(runs_clean)
        formula = "beta ~ 1 + run"
    else:
        formula = "beta ~ 1"
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Random effects covariance is singular")
            
            model = MixedLM.from_formula(
                formula,
                groups="subject",
                data=df
            )
            
            result = model.fit(
                reml=reml, 
                method=method,
                maxiter=max_iter,
                full_output=False,
                disp=False
            )
        
        # Extract results
        intercept = result.fe_params["Intercept"]
        se = result.bse["Intercept"]
        
        # Validate
        if not np.isfinite(intercept) or not np.isfinite(se) or se <= 0:
            return _failed_result("Invalid estimates")
        
        z_score = intercept / se
        p_value = result.pvalues["Intercept"]
        
        if not np.isfinite(z_score) or not np.isfinite(p_value):
            return _failed_result("Invalid statistics")
        
        # Calculate ICC
        try:
            re_var = float(result.cov_re.iloc[0, 0])
            resid_var = float(result.scale)
            total_var = re_var + resid_var
            
            if total_var > 0:
                icc = re_var / total_var
                icc = max(0, min(1, icc))
            else:
                icc = np.nan
        except:
            icc = np.nan
        
        return {
            "fixed_effect": float(intercept),
            "std_err": float(se),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "icc": float(icc) if np.isfinite(icc) else np.nan,
            "converged": True
        }
        
    except Exception as e:
        return _failed_result(f"Exception: {type(e).__name__}")


def _failed_result(reason="Unknown"):
    return {
        "fixed_effect": np.nan,
        "std_err": np.nan,
        "z_score": np.nan,
        "p_value": np.nan,
        "icc": np.nan,
        "converged": False
    }


def fit_lme_chunk(beta_matrix, run_info_df, voxel_indices, **lme_kwargs):
    results = []
    
    for i in range(beta_matrix.shape[1]):
        res = fit_lme_single_voxel(
            beta_matrix[:, i],
            run_info_df,
            **lme_kwargs
        )
        res["voxel_idx"] = voxel_indices[i]
        results.append(res)
    
    return results


def compute_fdr_correction(p_values, alpha=0.05):
    valid = np.isfinite(p_values) & (p_values >= 0) & (p_values <= 1)
    
    sig = np.zeros_like(p_values, dtype=bool)
    
    if valid.sum() == 0:
        return sig
    
    sig[valid] = _manual_bh_correction(p_values[valid], alpha)
    
    return sig


def _manual_bh_correction(p_values, alpha=0.05):
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    
    if not below.any():
        return np.zeros(n, dtype=bool)
    
    max_i = np.where(below)[0][-1]
    
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[:max_i + 1] = True
    
    unsort_idx = np.argsort(sorted_idx)
    reject = reject_sorted[unsort_idx]
    
    return reject