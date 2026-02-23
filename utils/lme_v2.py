import numpy as np
import pandas as pd
import warnings
import logging

logger = logging.getLogger(__name__)

_BACKEND = "statsmodels"
_PYMER4_VERSION = None
LmerClass = None
_HAS_POLARS = False

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    pass

try:
    from pymer4.models import lmer as _LmerClass
    import pymer4
    LmerClass = _LmerClass
    _PYMER4_VERSION = getattr(pymer4, '__version__', 'unknown')
    _BACKEND = "pymer4"
    logger.info(f"LME backend: pymer4 {_PYMER4_VERSION} (R lme4)")
except ImportError:
    try:
        from pymer4.models import Lmer as _LmerClass
        import pymer4
        LmerClass = _LmerClass
        _PYMER4_VERSION = getattr(pymer4, '__version__', 'unknown')
        _BACKEND = "pymer4_legacy"
        logger.info(f"LME backend: pymer4 {_PYMER4_VERSION} (legacy API)")
    except ImportError:
        from statsmodels.regression.mixed_linear_model import MixedLM
        from statsmodels.tools.sm_exceptions import ConvergenceWarning


def get_backend():
    return _BACKEND

def fit_lme_single_voxel(beta_values, run_info_df, method="powell", reml=True,
                         max_iter=200, include_run_effect=True,
                         fd_covariate=False):

    beta_values = np.asarray(beta_values, dtype=np.float64)

    if len(beta_values) < 10:
        return _failed_result("Too few observations")

    valid_mask = np.isfinite(beta_values)
    if valid_mask.sum() < 10:
        return _failed_result("Too few valid observations")

    beta_clean = beta_values[valid_mask]
    subjects_clean = run_info_df['subject'].values[valid_mask]

    runs_clean = None
    if 'run' in run_info_df.columns:
        runs_clean = run_info_df['run'].values[valid_mask]

    if pd.Series(subjects_clean).nunique() < 5:
        return _failed_result("Too few subjects")

    if np.std(beta_clean) < 1e-8:
        return _failed_result("Zero variance")

    data = {
        "beta": beta_clean.tolist(),
        "subject": [str(s) for s in subjects_clean],
    }

    has_run = (runs_clean is not None and
               len(np.unique(runs_clean)) >= 2 and
               include_run_effect)
    if has_run:
        data['run'] = [str(r) for r in runs_clean]

    has_fd = False
    if fd_covariate and 'mean_fd' in run_info_df.columns:
        fd_clean = run_info_df['mean_fd'].values[valid_mask]
        if np.isfinite(fd_clean).all():
            data['mean_fd'] = fd_clean.astype(np.float64).tolist()
            has_fd = True

    fixed_parts = ["1"]
    if has_run:
        fixed_parts.append("run")
    if has_fd:
        fixed_parts.append("mean_fd")
    formula = "beta ~ " + " + ".join(fixed_parts)

    if _BACKEND == "pymer4":
        return _fit_pymer4_v09(data, formula, reml=reml)
    elif _BACKEND == "pymer4_legacy":
        df_pd = pd.DataFrame(data)
        if has_run:
            df_pd['run'] = pd.Categorical(df_pd['run'])
        return _fit_pymer4_legacy(df_pd, formula, reml=reml)
    else:
        df_pd = pd.DataFrame(data)
        if has_run:
            df_pd['run'] = pd.Categorical(df_pd['run'])
        return _fit_statsmodels(df_pd, formula, method=method, reml=reml,
                                max_iter=max_iter)

def _fit_pymer4_v09(data_dict, formula, reml=True):
    try:
        df = pl.DataFrame(data_dict)

        full_formula = formula + " + (1|subject)"
        model = LmerClass(full_formula, data=df, REML=reml)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model.fit(summary=False)

        rf = model.result_fit
        if rf is None:
            return _failed_result("No result_fit")

        intercept_row = rf.filter(pl.col("term") == "(Intercept)")
        if intercept_row.height == 0:
            return _failed_result("No intercept in result_fit")

        intercept = float(intercept_row["estimate"][0])
        se = float(intercept_row["std_error"][0])
        t_val = float(intercept_row["t_stat"][0])
        p_val = float(intercept_row["p_value"][0])

        if not np.isfinite(intercept) or not np.isfinite(se) or se <= 0:
            return _failed_result("Invalid estimates (pymer4 0.9)")

        re_var, resid_var, icc = np.nan, np.nan, np.nan
        try:
            rv = model.ranef_var
            re_row = rv.filter(pl.col("group") == "subject")
            resid_row = rv.filter(pl.col("group") == "Residual")

            if re_row.height > 0 and resid_row.height > 0:
                re_sd = float(re_row["estimate"][0])
                resid_sd = float(resid_row["estimate"][0])
                re_var = re_sd ** 2
                resid_var = resid_sd ** 2
                total_var = re_var + resid_var
                icc = re_var / total_var if total_var > 0 else np.nan
                icc = max(0.0, min(1.0, icc))
        except Exception:
            pass

        residuals = None
        try:
            from rpy2.robjects import r as R
            resid_r = R('residuals')(model.r_model)
            residuals = np.array(resid_r, dtype=np.float64)
        except Exception:
            pass

        converged = True
        try:
            cs = model.convergence_status
            if cs is not None and isinstance(cs, str):
                converged = "TRUE" in cs.upper()
        except Exception:
            pass

        result = {
            "fixed_effect": float(intercept),
            "std_err": float(se),
            "z_score": float(t_val),
            "t_value": float(t_val),
            "p_value": float(p_val),
            "icc": float(icc) if np.isfinite(icc) else np.nan,
            "var_between": float(re_var) if np.isfinite(re_var) else np.nan,
            "var_within": float(resid_var) if np.isfinite(resid_var) else np.nan,
            "converged": converged,
            "backend": "pymer4/lme4",
        }
        if residuals is not None:
            result["residuals"] = residuals

        return result

    except Exception as e:
        return _failed_result(f"pymer4 0.9 exception: {type(e).__name__}: {e}")

def _fit_pymer4_legacy(df, formula, reml=True):
    try:
        full_formula = formula + " + (1|subject)"
        model = LmerClass(full_formula, data=df)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model.fit(REML=reml, summarize=False)

        coefs = model.coefs
        if coefs is None or 'Intercept' not in coefs.index:
            return _failed_result("No intercept in pymer4 legacy results")

        intercept = float(coefs.loc['Intercept', 'Estimate'])
        se = float(coefs.loc['Intercept', 'SE'])
        t_val = float(coefs.loc['Intercept', 'T-stat'])
        p_val = float(coefs.loc['Intercept', 'P-val'])

        if not np.isfinite(intercept) or not np.isfinite(se) or se <= 0:
            return _failed_result("Invalid estimates (pymer4 legacy)")

        try:
            ranef_var = model.ranef_var
            re_var = float(ranef_var[ranef_var['grp'] == 'subject']['vcov'].values[0])
            resid_var = float(ranef_var[ranef_var['grp'] == 'Residual']['vcov'].values[0])
            total_var = re_var + resid_var
            icc = re_var / total_var if total_var > 0 else np.nan
            icc = max(0.0, min(1.0, icc))
        except Exception:
            re_var, resid_var, icc = np.nan, np.nan, np.nan

        try:
            residuals = np.array(model.residuals, dtype=np.float64)
        except Exception:
            residuals = None

        result = {
            "fixed_effect": float(intercept),
            "std_err": float(se),
            "z_score": float(t_val),
            "t_value": float(t_val),
            "p_value": float(p_val),
            "icc": float(icc) if np.isfinite(icc) else np.nan,
            "var_between": float(re_var) if np.isfinite(re_var) else np.nan,
            "var_within": float(resid_var) if np.isfinite(resid_var) else np.nan,
            "converged": True,
            "backend": "pymer4/lme4",
        }
        if residuals is not None:
            result["residuals"] = residuals
        return result

    except Exception as e:
        return _failed_result(f"pymer4 legacy exception: {type(e).__name__}: {e}")

def _fit_statsmodels(df, formula, method="powell", reml=True, max_iter=200):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore",
                                    message="Random effects covariance is singular")

            model = MixedLM.from_formula(formula, groups="subject", data=df)
            result = model.fit(reml=reml, method=method, maxiter=max_iter,
                               full_output=False, disp=False)

        intercept = result.fe_params["Intercept"]
        se = result.bse["Intercept"]

        if not np.isfinite(intercept) or not np.isfinite(se) or se <= 0:
            return _failed_result("Invalid estimates (statsmodels)")

        z_score = intercept / se
        p_value = result.pvalues["Intercept"]

        if not np.isfinite(z_score) or not np.isfinite(p_value):
            return _failed_result("Invalid statistics (statsmodels)")

        try:
            re_var = float(result.cov_re.iloc[0, 0])
            resid_var = float(result.scale)
            total_var = re_var + resid_var
            icc = re_var / total_var if total_var > 0 else np.nan
            icc = max(0.0, min(1.0, icc))
        except Exception:
            re_var, resid_var, icc = np.nan, np.nan, np.nan

        try:
            residuals = np.array(result.resid, dtype=np.float64)
        except Exception:
            residuals = None

        res = {
            "fixed_effect": float(intercept),
            "std_err": float(se),
            "z_score": float(z_score),
            "t_value": float(z_score),
            "p_value": float(p_value),
            "icc": float(icc) if np.isfinite(icc) else np.nan,
            "var_between": float(re_var) if np.isfinite(re_var) else np.nan,
            "var_within": float(resid_var) if np.isfinite(resid_var) else np.nan,
            "converged": True,
            "backend": "statsmodels",
        }
        if residuals is not None:
            res["residuals"] = residuals
        return res

    except Exception as e:
        return _failed_result(f"statsmodels exception: {type(e).__name__}")

def _failed_result(reason="Unknown"):
    return {
        "fixed_effect": np.nan,
        "std_err": np.nan,
        "z_score": np.nan,
        "t_value": np.nan,
        "p_value": np.nan,
        "icc": np.nan,
        "var_between": np.nan,
        "var_within": np.nan,
        "converged": False,
        "backend": _BACKEND,
    }


def fit_lme_chunk(beta_matrix, run_info_df, voxel_indices, **lme_kwargs):
    results = []
    for i in range(beta_matrix.shape[1]):
        res = fit_lme_single_voxel(
            beta_matrix[:, i], run_info_df, **lme_kwargs
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
    return reject_sorted[unsort_idx]