from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


VAR_ORDER = [
    "eci_wage",
    "services_ex_housing",
    "housing",
    "core_goods",
    "core_pce",
]

INFLATION_TARGETS = [
    "services_ex_housing",
    "housing",
    "core_goods",
    "core_pce",
]

NO_WAGE_VAR_ORDER = INFLATION_TARGETS.copy()

VAR_LABELS = {
    "eci_wage": "ECI wage growth",
    "services_ex_housing": "Services excluding housing inflation",
    "housing": "Housing inflation",
    "core_goods": "Core goods inflation",
    "core_pce": "Core PCE inflation",
}

INDEX_LABELS = {
    "eci_wage": "ECI wage index",
    "services_ex_housing": "Services excluding housing index",
    "housing": "Housing index",
    "core_goods": "Core goods index",
    "core_pce": "Core PCE index",
}

MODEL_NAMES = ["AM-BVAR-SV", "AR(4)", "UC-SV"]
WAGE_MODEL_NAME = "AM-BVAR-SV with wage"
NO_WAGE_MODEL_NAME = "AM-BVAR-SV without wage"
RAW_INDEX_START = pd.Period("2001Q1", freq="Q")
FIRST_TRANSFORMED_OBSERVATION = pd.Period("2001Q2", freq="Q")
P_LAGS = 4
HORIZONS = [1, 2, 3, 4, 8]
FIRST_FORECAST_ORIGIN = pd.Period("2012Q4", freq="Q")
ROBUSTNESS_FORECAST_ORIGIN = pd.Period("2014Q4", freq="Q")

DEFAULT_GIBBS_N_ITER_RECURSIVE = 1200
DEFAULT_GIBBS_BURN_RECURSIVE = 400
DEFAULT_GIBBS_THIN_RECURSIVE = 2
DEFAULT_GIBBS_TRAJECTORIES_RECURSIVE = 3
DEFAULT_GIBBS_N_ITER_FINAL = 3000
DEFAULT_GIBBS_BURN_FINAL = 1000
DEFAULT_GIBBS_THIN_FINAL = 2
DEFAULT_GIBBS_TRAJECTORIES_FINAL = 10

# Kim, Shephard, and Chib seven-component approximation to log(chi-square_1).
# The tabulated means are shifted by -1.2704 for log(epsilon^2).
KSC_MIXTURE_WEIGHTS = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_MIXTURE_MEANS = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
KSC_MIXTURE_VARIANCES = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])

DATA_SPECS = [
    {
        "variable": "eci_wage",
        "provider": "FRED",
        "series_id": "CIS1020000000000I",
        "source": "FRED",
        "line_item": "CIS1020000000000I",
        "expected_terms": ["employment cost index", "wages"],
    },
    {
        "variable": "services_ex_housing",
        "provider": "BEA",
        "dataset": "NIPA",
        "table_name": "T20304",
        "source": "BEA Table 2.3.4",
        "line_number": 28,
        "line_item": "Line 28",
        "expected_terms": ["services", "housing"],
    },
    {
        "variable": "housing",
        "provider": "BEA",
        "dataset": "NIPA",
        "table_name": "T20304",
        "source": "BEA Table 2.3.4",
        "line_number": 29,
        "line_item": "Line 29",
        "expected_terms": ["housing"],
    },
    {
        "variable": "core_goods",
        "provider": "BEA",
        "dataset": "NIUnderlyingDetail",
        "table_name": "U20404",
        "source": "BEA Table 2.4.4U",
        "line_number": 375,
        "line_item": "Line 375",
        "expected_terms": ["goods", "excluding food", "energy"],
    },
    {
        "variable": "core_pce",
        "provider": "BEA",
        "dataset": "NIUnderlyingDetail",
        "table_name": "U20404",
        "source": "BEA Table 2.4.4U",
        "line_number": 374,
        "line_item": "Line 374",
        "expected_terms": ["excluding food", "energy"],
    },
]


def load_api_keys(path: Path) -> dict[str, str]:
    """Parse a simple 'NAME: value' key file without printing secrets."""
    keys: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        keys[name.strip().upper()] = value.strip()
    missing = {"BEA", "FRED"}.difference(keys)
    if missing:
        raise KeyError(f"Missing required API keys: {sorted(missing)}")
    return keys


def request_json(
    base_url: str,
    params: dict,
    *,
    retries: int = 3,
    sleep_seconds: float = 1.0,
) -> dict:
    """GET JSON with a small retry loop for transient API hiccups."""
    url = base_url + "?" + urlencode(params)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=60) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {base_url}") from last_exc


def clean_numeric(value) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "")
    if text in {"", ".", "NA", "N/A", "(NA)"}:
        return np.nan
    return float(text)


def quarter_from_bea_period(value: str) -> pd.Period:
    return pd.Period(str(value).replace(" ", ""), freq="Q")


def quarter_from_fred_date(value: str) -> pd.Period:
    return pd.Period(pd.Timestamp(value), freq="Q")


def description_check(description: str, expected_terms) -> str:
    desc = (description or "").lower()
    missing = [term for term in expected_terms if term.lower() not in desc]
    if missing:
        return "REVIEW: missing " + ", ".join(missing)
    return "OK"


def fetch_bea_line(spec: dict, api_key: str) -> tuple[pd.Series, dict]:
    params = {
        "UserID": api_key,
        "method": "GetData",
        "DataSetName": spec["dataset"],
        "TableName": spec["table_name"],
        "Frequency": "Q",
        "Year": "ALL",
        "ResultFormat": "JSON",
    }
    data = request_json("https://apps.bea.gov/api/data/", params)
    root = data.get("BEAAPI", {})
    error = root.get("Error") or root.get("Results", {}).get("Error")
    if error:
        raise RuntimeError(f"BEA API error for {spec['table_name']}: {error}")
    rows = root.get("Results", {}).get("Data", [])
    line_rows = [row for row in rows if str(row.get("LineNumber")) == str(spec["line_number"])]
    if not line_rows:
        raise ValueError(f"No BEA rows found for {spec['table_name']} line {spec['line_number']}")
    periods = [quarter_from_bea_period(row["TimePeriod"]) for row in line_rows]
    values = [clean_numeric(row.get("DataValue")) for row in line_rows]
    series = pd.Series(values, index=pd.PeriodIndex(periods, freq="Q"), name=spec["variable"]).sort_index()
    description = line_rows[0].get("LineDescription", "")
    meta = {
        "variable": spec["variable"],
        "variable_name": VAR_LABELS[spec["variable"]],
        "source_table_or_series": spec["source"],
        "api_dataset": spec["dataset"],
        "api_table_name": spec["table_name"],
        "line_item_or_series_id": spec["line_item"],
        "downloaded_description": description,
        "description_check": description_check(description, spec["expected_terms"]),
        "raw_index_start_date": str(series.dropna().index.min()) if series.notna().any() else None,
        "downloaded_latest_date": str(series.dropna().index.max()) if series.notna().any() else None,
    }
    return series, meta


def fetch_fred_series(spec: dict, api_key: str) -> tuple[pd.Series, dict]:
    meta_payload = request_json(
        "https://api.stlouisfed.org/fred/series",
        {"series_id": spec["series_id"], "api_key": api_key, "file_type": "json"},
    )
    series_meta = meta_payload.get("seriess", [{}])[0]
    obs_payload = request_json(
        "https://api.stlouisfed.org/fred/series/observations",
        {
            "series_id": spec["series_id"],
            "api_key": api_key,
            "file_type": "json",
            "observation_start": "2001-01-01",
            "units": "lin",
            "sort_order": "asc",
        },
    )
    observations = obs_payload.get("observations", [])
    periods = [quarter_from_fred_date(row["date"]) for row in observations]
    values = [clean_numeric(row.get("value")) for row in observations]
    series = pd.Series(values, index=pd.PeriodIndex(periods, freq="Q"), name=spec["variable"]).sort_index()
    title = series_meta.get("title", "")
    meta = {
        "variable": spec["variable"],
        "variable_name": VAR_LABELS[spec["variable"]],
        "source_table_or_series": spec["source"],
        "api_dataset": "FRED",
        "api_table_name": "fred/series/observations",
        "line_item_or_series_id": spec["series_id"],
        "downloaded_description": title,
        "description_check": description_check(title, spec["expected_terms"]),
        "raw_index_start_date": str(series.dropna().index.min()) if series.notna().any() else None,
        "downloaded_latest_date": str(series.dropna().index.max()) if series.notna().any() else None,
        "frequency": series_meta.get("frequency"),
        "units": series_meta.get("units"),
    }
    return series, meta


def download_index_levels(api_keys: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_map = {}
    metadata_records = []
    for spec in DATA_SPECS:
        if spec["provider"] == "BEA":
            series, meta = fetch_bea_line(spec, api_keys["BEA"])
        elif spec["provider"] == "FRED":
            series, meta = fetch_fred_series(spec, api_keys["FRED"])
        else:
            raise ValueError(f"Unsupported provider: {spec['provider']}")
        series_map[spec["variable"]] = series
        metadata_records.append(meta)

    index_levels = pd.concat(series_map, axis=1)
    index_levels = index_levels.loc[index_levels.index >= RAW_INDEX_START, VAR_ORDER].sort_index()
    metadata = pd.DataFrame(metadata_records)
    metadata["raw_index_start_date_after_constraint"] = str(RAW_INDEX_START)
    metadata["first_transformed_observation"] = str(FIRST_TRANSFORMED_OBSERVATION)
    metadata["transformation"] = "400 * log(index_t / index_t-1)"
    return index_levels, metadata


def load_or_download_data(
    project_dir: Path,
    api_keys: dict[str, str],
    *,
    force_recompute: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = project_dir / "data"
    output_dir = project_dir / "outputs"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    raw_cache = data_dir / "raw_index_levels.csv"
    metadata_cache = output_dir / "data_definitions.csv"
    if raw_cache.exists() and metadata_cache.exists() and not force_recompute:
        index_levels = pd.read_csv(raw_cache, index_col=0)
        index_levels.index = pd.PeriodIndex(index_levels.index, freq="Q")
        metadata = pd.read_csv(metadata_cache)
        return index_levels, metadata

    index_levels, metadata = download_index_levels(api_keys)
    index_levels.to_csv(raw_cache)
    metadata.to_csv(metadata_cache, index=False)
    return index_levels, metadata


def transform_index_levels(index_levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    growth_rates = 400.0 * np.log(index_levels / index_levels.shift(1))
    growth_rates = growth_rates.loc[growth_rates.index >= FIRST_TRANSFORMED_OBSERVATION, VAR_ORDER]
    common_growth = growth_rates.dropna(how="any")
    common_index_levels = index_levels.loc[common_growth.index, VAR_ORDER]
    if common_growth.empty:
        raise RuntimeError("No complete transformed observations were created.")
    if common_growth.index.min() != FIRST_TRANSFORMED_OBSERVATION:
        warnings.warn(
            f"First complete transformed observation is {common_growth.index.min()}, "
            f"not {FIRST_TRANSFORMED_OBSERVATION}. Check missing data."
        )
    return common_growth, common_index_levels


def as_2d_array(df_or_array):
    arr = np.asarray(df_or_array, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def make_lagged_xy(y, p: int):
    y = as_2d_array(y)
    t_obs, n = y.shape
    if t_obs <= p:
        raise ValueError(f"Need more than {p} observations, got {t_obs}")
    x = np.ones((t_obs - p, 1 + n * p))
    for lag in range(1, p + 1):
        start = 1 + (lag - 1) * n
        x[:, start:start + n] = y[p - lag:t_obs - lag, :]
    target = y[p:, :]
    return x, target


def lagged_forecast_row(history, p: int):
    history = as_2d_array(history)
    n = history.shape[1]
    x = np.ones(1 + n * p)
    for lag in range(1, p + 1):
        start = 1 + (lag - 1) * n
        x[start:start + n] = history[-lag, :]
    return x


def fit_ar_ols(y, p: int = P_LAGS, ridge: float = 1e-8):
    y = np.asarray(y, dtype=float)
    x, target = make_lagged_xy(y[:, None], p)
    xtx = x.T @ x
    penalty = ridge * np.eye(xtx.shape[0])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xtx + penalty, x.T @ target[:, 0])
    resid = target[:, 0] - x @ beta
    dof = max(len(resid) - len(beta), 1)
    sigma2 = float(resid @ resid / dof)
    return {"beta": beta, "sigma2": max(sigma2, 1e-10), "p": p, "resid": resid}


def forecast_ar_series(y_train, horizon: int, p: int = P_LAGS):
    fit = fit_ar_ols(y_train, p=p)
    history = list(np.asarray(y_train, dtype=float))
    forecasts = []
    for _ in range(horizon):
        row = np.array([1.0] + [history[-lag] for lag in range(1, p + 1)], dtype=float)
        yhat = float(row @ fit["beta"])
        forecasts.append(yhat)
        history.append(yhat)
    return np.array(forecasts)


def forecast_ar_panel(train_df: pd.DataFrame, max_horizon: int, p: int = P_LAGS, variables=None):
    variables = list(VAR_ORDER if variables is None else variables)
    out = np.zeros((max_horizon, len(variables)))
    for j, var in enumerate(variables):
        out[:, j] = forecast_ar_series(train_df[var].to_numpy(), max_horizon, p=p)
    return out


def rmse(actual, forecast) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(forecast)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual[mask] - forecast[mask]) ** 2)))


def inverse_gamma_draw(shape, scale, size, rng):
    return scale / rng.gamma(shape=shape, scale=1.0, size=size)


def ar_residual_variances(y, p: int = P_LAGS):
    y = as_2d_array(y)
    variances = []
    for j in range(y.shape[1]):
        try:
            fit = fit_ar_ols(y[:, j], p=p)
            variances.append(fit["sigma2"])
        except Exception:
            variances.append(float(np.nanvar(y[:, j], ddof=1)))
    return np.maximum(np.asarray(variances, dtype=float), 1e-8)


def minnesota_prior_for_equation(
    equation_index: int,
    n_vars: int,
    p: int,
    residual_variances: np.ndarray,
    lambda_shrink: float,
    decay: float = 1.0,
    cross_weight: float = 0.5,
    intercept_sd: float = 20.0,
    own_first_lag_mean: float = 0.0,
):
    k = 1 + n_vars * p
    b0 = np.zeros(k)
    v0 = np.zeros(k)
    sigma_i = residual_variances[equation_index]
    v0[0] = (intercept_sd**2) / sigma_i
    for lag in range(1, p + 1):
        for variable_index in range(n_vars):
            idx = 1 + (lag - 1) * n_vars + variable_index
            scale = (lambda_shrink**2) / (lag ** (2.0 * decay))
            if variable_index != equation_index:
                scale *= cross_weight**2
            v0[idx] = scale / residual_variances[variable_index]
    b0[1 + equation_index] = own_first_lag_mean
    return b0, np.maximum(v0, 1e-12)


def posterior_nig(y, x, b0, v0_diag, a0, d0):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    prior_precision_diag = 1.0 / v0_diag
    vn_inv = x.T @ x + np.diag(prior_precision_diag)
    rhs = x.T @ y + prior_precision_diag * b0
    bn = np.linalg.solve(vn_inv, rhs)
    vn = np.linalg.inv(vn_inv)
    an = a0 + 0.5 * len(y)
    quad_prior = float(b0 @ (prior_precision_diag * b0))
    quad_post = float(bn @ (vn_inv @ bn))
    dn = d0 + 0.5 * (float(y @ y) + quad_prior - quad_post)
    dn = max(float(dn), 1e-12)
    sign_inv, logdet_vn_inv = np.linalg.slogdet(vn_inv)
    if sign_inv <= 0:
        raise np.linalg.LinAlgError("Posterior precision is not positive definite")
    logdet_vn = -logdet_vn_inv
    logdet_v0 = float(np.sum(np.log(v0_diag)))
    log_ml = (
        -0.5 * len(y) * math.log(math.pi)
        + 0.5 * (logdet_vn - logdet_v0)
        + a0 * math.log(d0)
        - an * math.log(dn)
        + math.lgamma(an)
        - math.lgamma(a0)
    )
    return {"bn": bn, "vn": vn, "vn_inv": vn_inv, "an": an, "dn": dn, "log_ml": log_ml}


def make_posdef_corr(matrix, min_eig: float = 1e-6):
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, min_eig, None)
    out = (vectors * values) @ vectors.T
    scale = np.sqrt(np.clip(np.diag(out), min_eig, None))
    out = out / np.outer(scale, scale)
    np.fill_diagonal(out, 1.0)
    return out


def ewma(values, alpha: float):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = alpha * values[t] + (1.0 - alpha) * out[t - 1]
    return out


def estimate_stochastic_volatility_state(residuals, alpha: float = 0.15):
    residuals = as_2d_array(residuals)
    t_obs, n = residuals.shape
    resid_var = np.maximum(np.nanvar(residuals, axis=0, ddof=1), 1e-8)
    logvar_smooth = np.zeros((t_obs, n))
    sv_scales = np.zeros(n)
    for j in range(n):
        raw_log = np.log(residuals[:, j] ** 2 + 0.05 * resid_var[j])
        smooth = ewma(raw_log, alpha=alpha)
        logvar_smooth[:, j] = smooth
        if len(smooth) > 2:
            sv_scales[j] = np.nanstd(np.diff(smooth), ddof=1)
        else:
            sv_scales[j] = 0.10
    sv_scales = np.clip(sv_scales, 0.03, 0.35)
    standardized = residuals / np.sqrt(np.exp(logvar_smooth))
    if t_obs > 2:
        corr = np.corrcoef(standardized, rowvar=False)
    else:
        corr = np.eye(n)
    if not np.all(np.isfinite(corr)):
        corr = np.eye(n)
    corr = make_posdef_corr(0.95 * corr + 0.05 * np.eye(n))
    chol_corr = np.linalg.cholesky(corr)
    floor = np.nanpercentile(logvar_smooth, 1, axis=0) - 1.0
    ceiling = np.nanpercentile(logvar_smooth, 99, axis=0) + 1.0
    return {
        "last_log_var": logvar_smooth[-1, :],
        "sv_scales": sv_scales,
        "corr": corr,
        "chol_corr": chol_corr,
        "logvar_floor": floor,
        "logvar_ceiling": ceiling,
        "logvar_smooth": logvar_smooth,
    }


def symmetrize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def ensure_positive_definite(matrix, min_eig: float = 1e-10):
    matrix = symmetrize(matrix)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, min_eig)
    return symmetrize((vectors * values) @ vectors.T)


def cholesky_psd(matrix, *, jitter: float = 1e-10, max_tries: int = 7, diagnostics: dict | None = None):
    matrix = symmetrize(matrix)
    eye = np.eye(matrix.shape[0])
    current_jitter = 0.0
    for attempt in range(max_tries):
        try:
            chol = np.linalg.cholesky(matrix + current_jitter * eye)
            if attempt > 0 and diagnostics is not None:
                diagnostics["cholesky_jitter_events"] = diagnostics.get("cholesky_jitter_events", 0) + 1
            return chol
        except np.linalg.LinAlgError:
            current_jitter = jitter * (10.0**attempt)
    if diagnostics is not None:
        diagnostics["cholesky_jitter_events"] = diagnostics.get("cholesky_jitter_events", 0) + 1
    repaired = ensure_positive_definite(matrix, min_eig=max(current_jitter, jitter))
    return np.linalg.cholesky(repaired)


def draw_mvn_cov(mean, covariance, rng, diagnostics: dict | None = None):
    mean = np.asarray(mean, dtype=float)
    chol = cholesky_psd(covariance, diagnostics=diagnostics)
    return mean + chol @ rng.standard_normal(len(mean))


def draw_mvn_precision(mean, precision, rng, diagnostics: dict | None = None):
    mean = np.asarray(mean, dtype=float)
    chol = cholesky_psd(precision, diagnostics=diagnostics)
    return mean + np.linalg.solve(chol.T, rng.standard_normal(len(mean)))


def inverse_wishart_draw(df: float, scale, rng, diagnostics: dict | None = None):
    scale = ensure_positive_definite(scale)
    n = scale.shape[0]
    inv_scale = np.linalg.inv(scale)
    chol = cholesky_psd(inv_scale, diagnostics=diagnostics)
    bartlett = np.zeros((n, n))
    for i in range(n):
        bartlett[i, i] = math.sqrt(rng.chisquare(df - i))
        for j in range(i):
            bartlett[i, j] = rng.standard_normal()
    wishart = chol @ bartlett @ bartlett.T @ chol.T
    out = np.linalg.inv(ensure_positive_definite(wishart))
    return ensure_positive_definite(out)


def build_adaptive_minnesota_prior(
    n_vars: int,
    p: int,
    residual_variances: np.ndarray,
    kappa_1: float,
    kappa_2: float,
    *,
    intercept_sd: float = 20.0,
    use_scale: bool = True,
):
    k = 1 + n_vars * p
    beta0 = np.zeros(k * n_vars)
    v0_diag = np.zeros(k * n_vars)
    residual_variances = np.maximum(np.asarray(residual_variances, dtype=float), 1e-8)
    for equation_index in range(n_vars):
        offset = equation_index * k
        v0_diag[offset] = intercept_sd**2
        for lag in range(1, p + 1):
            for variable_index in range(n_vars):
                reg_index = 1 + (lag - 1) * n_vars + variable_index
                if variable_index == equation_index:
                    variance = kappa_1 / (lag**2)
                else:
                    scale = residual_variances[equation_index] / residual_variances[variable_index] if use_scale else 1.0
                    variance = kappa_2 * scale / (lag**2)
                v0_diag[offset + reg_index] = variance
    return beta0, np.maximum(v0_diag, 1e-12)


def log_adaptive_minnesota_prior_density(
    beta,
    n_vars: int,
    p: int,
    residual_variances: np.ndarray,
    kappa_1: float,
    kappa_2: float,
    *,
    theta_prior_sd: float = 0.75,
    use_scale: bool = True,
):
    if not (np.isfinite(kappa_1) and np.isfinite(kappa_2)) or kappa_1 <= 0.0 or kappa_2 <= 0.0:
        return -np.inf
    theta = np.log([kappa_1, kappa_2])
    theta_mean = np.log([0.1, 0.005])
    log_theta_prior = -0.5 * np.sum(((theta - theta_mean) / theta_prior_sd) ** 2)
    _, v0_diag = build_adaptive_minnesota_prior(
        n_vars,
        p,
        residual_variances,
        kappa_1,
        kappa_2,
        use_scale=use_scale,
    )
    beta = np.asarray(beta, dtype=float)
    return float(log_theta_prior - 0.5 * np.sum(np.log(v0_diag) + (beta**2) / v0_diag))


def draw_kappa_given_B(
    B,
    residual_variances: np.ndarray,
    current_kappa: np.ndarray,
    rng,
    *,
    p: int,
    proposal_sd=None,
    use_scale: bool = True,
):
    n_vars = B.shape[1]
    beta = B.reshape(-1, order="F")
    current_kappa = np.maximum(np.asarray(current_kappa, dtype=float), 1e-12)
    proposal_sd = np.asarray([0.10, 0.10] if proposal_sd is None else proposal_sd, dtype=float)
    current_theta = np.log(current_kappa)
    proposed_theta = current_theta + proposal_sd * rng.standard_normal(2)
    proposed_kappa = np.exp(proposed_theta)
    current_logpost = log_adaptive_minnesota_prior_density(
        beta,
        n_vars,
        p,
        residual_variances,
        current_kappa[0],
        current_kappa[1],
        use_scale=use_scale,
    )
    proposed_logpost = log_adaptive_minnesota_prior_density(
        beta,
        n_vars,
        p,
        residual_variances,
        proposed_kappa[0],
        proposed_kappa[1],
        use_scale=use_scale,
    )
    if math.log(rng.random()) < proposed_logpost - current_logpost:
        return proposed_kappa, True
    return current_kappa, False


def initialize_am_bvar_sv(
    train_df: pd.DataFrame,
    p: int = P_LAGS,
    variables=None,
    *,
    ridge: float = 1e-6,
):
    variables = list(VAR_ORDER if variables is None else variables)
    y = train_df[variables].dropna(how="any").to_numpy(dtype=float)
    x, target = make_lagged_xy(y, p)
    k = x.shape[1]
    n = target.shape[1]
    penalty = ridge * np.eye(k)
    penalty[0, 0] = 0.0
    B = np.linalg.solve(x.T @ x + penalty, x.T @ target)
    U = target - x @ B
    A = np.eye(n)
    for i in range(1, n):
        XA = -U[:, :i]
        yA = U[:, i]
        try:
            coef = np.linalg.solve(XA.T @ XA + ridge * np.eye(i), XA.T @ yA)
        except np.linalg.LinAlgError:
            coef = np.zeros(i)
        A[i, :i] = coef
    E = U @ A.T
    structural_variances = np.maximum(np.nanvar(E, axis=0, ddof=1), 1e-6)
    H = np.tile(np.log(structural_variances), (target.shape[0], 1))
    Phi = 0.05 * np.eye(n)
    residual_variances = ar_residual_variances(y, p=p)
    return {
        "variables": variables,
        "y_raw": y,
        "x": x,
        "target": target,
        "periods": train_df[variables].dropna(how="any").index[p:],
        "B": B,
        "A": A,
        "H": H,
        "Phi": Phi,
        "kappa": np.array([0.1, 0.005], dtype=float),
        "residual_variances": residual_variances,
    }


def draw_B_given_A_H_kappa(
    X,
    Y,
    A,
    H,
    kappa,
    residual_variances: np.ndarray,
    rng,
    *,
    p: int,
    use_adaptive_minnesota: bool = True,
    diagnostics: dict | None = None,
):
    T, k = X.shape
    n = Y.shape[1]
    kappa_1, kappa_2 = (float(kappa[0]), float(kappa[1])) if use_adaptive_minnesota else (0.1, 0.005)
    beta0, v0_diag = build_adaptive_minnesota_prior(
        n,
        p,
        residual_variances,
        kappa_1,
        kappa_2,
        use_scale=True,
    )
    precision = np.diag(1.0 / v0_diag)
    rhs = beta0 / v0_diag
    for t in range(T):
        inv_lambda = np.exp(np.clip(-H[t], -50.0, 50.0))
        sigma_inv = A.T @ (inv_lambda[:, None] * A)
        xx = np.outer(X[t], X[t])
        precision += np.kron(sigma_inv, xx)
        rhs += np.kron(sigma_inv @ Y[t], X[t])
    precision = ensure_positive_definite(precision)
    beta_mean = np.linalg.solve(precision, rhs)
    beta_draw = draw_mvn_precision(beta_mean, precision, rng, diagnostics=diagnostics)
    return beta_draw.reshape((k, n), order="F")


def draw_A_given_B_H(
    X,
    Y,
    B,
    H,
    rng,
    *,
    prior_variance: float = 100.0,
    diagnostics: dict | None = None,
):
    U = Y - X @ B
    n = U.shape[1]
    A = np.eye(n)
    for i in range(1, n):
        XA = -U[:, :i]
        yA = U[:, i]
        weights = np.exp(np.clip(-H[:, i], -50.0, 50.0))
        sqrt_w = np.sqrt(weights)
        Xw = XA * sqrt_w[:, None]
        yw = yA * sqrt_w
        prior_precision = np.eye(i) / prior_variance
        precision = prior_precision + Xw.T @ Xw
        rhs = Xw.T @ yw
        mean = np.linalg.solve(ensure_positive_definite(precision), rhs)
        covariance = np.linalg.inv(ensure_positive_definite(precision))
        A[i, :i] = draw_mvn_cov(mean, covariance, rng, diagnostics=diagnostics)
    return A


def draw_sv_mixture_indicators(E, H, rng, *, small_constant: float = 1e-8):
    E = np.asarray(E, dtype=float)
    H = np.asarray(H, dtype=float)
    z = np.log(E**2 + small_constant)
    means = KSC_MIXTURE_MEANS
    variances = KSC_MIXTURE_VARIANCES
    log_weights = (
        np.log(KSC_MIXTURE_WEIGHTS)[None, None, :]
        - 0.5 * np.log(variances)[None, None, :]
        - 0.5 * ((z[:, :, None] - H[:, :, None] - means[None, None, :]) ** 2) / variances[None, None, :]
    )
    log_weights -= np.max(log_weights, axis=2, keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=2, keepdims=True)
    cumulative = np.cumsum(weights, axis=2)
    uniforms = rng.random(size=E.shape)
    indicators = (uniforms[:, :, None] > cumulative).sum(axis=2)
    mixture_means = means[indicators]
    mixture_variances = variances[indicators]
    return indicators, mixture_means, mixture_variances, z


def draw_H_given_structural_residuals_mixture_Phi(
    E,
    mixture_means,
    mixture_variances,
    Phi,
    rng,
    *,
    h0_mean=None,
    h0_variance_scale: float = 10.0,
    small_constant: float = 1e-8,
    diagnostics: dict | None = None,
):
    E = np.asarray(E, dtype=float)
    T, n = E.shape
    z = np.log(E**2 + small_constant)
    y_tilde = z - mixture_means
    Phi = ensure_positive_definite(Phi)
    if h0_mean is None:
        h0_mean = np.log(np.maximum(np.nanvar(E, axis=0, ddof=1), 1e-6))
    h0_mean = np.asarray(h0_mean, dtype=float)
    h0_cov = h0_variance_scale * np.eye(n)
    filtered_means = np.zeros((T, n))
    filtered_covs = np.zeros((T, n, n))
    a = h0_mean.copy()
    P = h0_cov + Phi
    for t in range(T):
        R = np.diag(np.maximum(mixture_variances[t], 1e-8))
        Q = ensure_positive_definite(P + R)
        K_gain = np.linalg.solve(Q.T, P.T).T
        innovation = y_tilde[t] - a
        m = a + K_gain @ innovation
        C = ensure_positive_definite(P - K_gain @ Q @ K_gain.T)
        filtered_means[t] = m
        filtered_covs[t] = C
        a = m
        P = C + Phi
    H_draw = np.zeros((T, n))
    H_draw[-1] = draw_mvn_cov(filtered_means[-1], filtered_covs[-1], rng, diagnostics=diagnostics)
    for t in range(T - 2, -1, -1):
        S = ensure_positive_definite(filtered_covs[t] + Phi)
        J = np.linalg.solve(S.T, filtered_covs[t].T).T
        mean = filtered_means[t] + J @ (H_draw[t + 1] - filtered_means[t])
        covariance = ensure_positive_definite(filtered_covs[t] - J @ S @ J.T)
        H_draw[t] = draw_mvn_cov(mean, covariance, rng, diagnostics=diagnostics)
    return H_draw


def draw_Phi_given_H(H, rng, *, diagnostics: dict | None = None):
    H = np.asarray(H, dtype=float)
    n = H.shape[1]
    differences = np.diff(H, axis=0)
    df0 = n + 3
    S0 = 0.05 * (n + 3) * np.eye(n)
    df_post = df0 + differences.shape[0]
    S_post = S0 + differences.T @ differences
    return inverse_wishart_draw(df_post, S_post, rng, diagnostics=diagnostics)


def am_bvar_sv_log_likelihood_proxy(Y, X, B, A, H):
    U = Y - X @ B
    E = U @ A.T
    h = np.clip(H, -50.0, 50.0)
    return float(-0.5 * np.sum(math.log(2.0 * math.pi) + h + (E**2) / np.exp(h)))


def fit_am_bvar_sv_gibbs(
    train_df,
    p: int = P_LAGS,
    variables=VAR_ORDER,
    n_iter: int = 12000,
    burn: int = 2000,
    thin: int = 1,
    seed: int = 20260519,
    store_h_path: bool = False,
    use_adaptive_minnesota: bool = True,
    verbose: bool = False,
):
    if n_iter <= burn:
        raise ValueError("n_iter must exceed burn")
    if thin <= 0:
        raise ValueError("thin must be positive")
    rng = np.random.default_rng(seed)
    init = initialize_am_bvar_sv(train_df, p=p, variables=variables)
    X = init["x"]
    Y = init["target"]
    residual_variances = init["residual_variances"]
    T, k = X.shape
    n = Y.shape[1]
    retained = (n_iter - burn) // thin
    if retained <= 0:
        raise ValueError("No posterior draws retained; reduce burn or thin")

    B = init["B"].copy()
    A = init["A"].copy()
    H = init["H"].copy()
    Phi = init["Phi"].copy()
    kappa = init["kappa"].copy()
    proposal_sd = np.array([0.10, 0.10], dtype=float)
    diagnostics = {
        "cholesky_jitter_events": 0,
        "kappa_accepts": 0,
        "kappa_attempts": 0,
        "post_burn_kappa_accepts": 0,
        "post_burn_kappa_attempts": 0,
        "n_iter": int(n_iter),
        "burn": int(burn),
        "thin": int(thin),
    }
    B_draws = np.zeros((retained, k, n))
    A_draws = np.zeros((retained, n, n))
    H_T_draws = np.zeros((retained, n))
    Phi_draws = np.zeros((retained, n, n))
    kappa_draws = np.zeros((retained, 2))
    loglik_draws = np.zeros(retained)
    H_draws = np.zeros((retained, T, n)) if store_h_path else None

    store_idx = 0
    window_attempts = 0
    window_accepts = 0
    for iteration in range(1, n_iter + 1):
        B = draw_B_given_A_H_kappa(
            X,
            Y,
            A,
            H,
            kappa,
            residual_variances,
            rng,
            p=p,
            use_adaptive_minnesota=use_adaptive_minnesota,
            diagnostics=diagnostics,
        )
        U = Y - X @ B
        A = draw_A_given_B_H(X, Y, B, H, rng, diagnostics=diagnostics)
        E = U @ A.T
        _, mixture_means, mixture_variances, _ = draw_sv_mixture_indicators(E, H, rng)
        H = draw_H_given_structural_residuals_mixture_Phi(
            E,
            mixture_means,
            mixture_variances,
            Phi,
            rng,
            diagnostics=diagnostics,
        )
        Phi = draw_Phi_given_H(H, rng, diagnostics=diagnostics)
        if use_adaptive_minnesota:
            kappa, accepted = draw_kappa_given_B(
                B,
                residual_variances,
                kappa,
                rng,
                p=p,
                proposal_sd=proposal_sd,
            )
        else:
            accepted = False
        diagnostics["kappa_attempts"] += 1
        diagnostics["kappa_accepts"] += int(accepted)
        window_attempts += 1
        window_accepts += int(accepted)
        if iteration > burn:
            diagnostics["post_burn_kappa_attempts"] += 1
            diagnostics["post_burn_kappa_accepts"] += int(accepted)
        if iteration <= burn and iteration % 100 == 0 and use_adaptive_minnesota and window_attempts:
            rate = window_accepts / window_attempts
            if rate < 0.20:
                proposal_sd *= 0.85
            elif rate > 0.45:
                proposal_sd *= 1.15
            proposal_sd = np.clip(proposal_sd, 0.02, 0.40)
            window_attempts = 0
            window_accepts = 0
        if iteration > burn and ((iteration - burn) % thin == 0):
            B_draws[store_idx] = B
            A_draws[store_idx] = A
            H_T_draws[store_idx] = H[-1]
            Phi_draws[store_idx] = Phi
            kappa_draws[store_idx] = kappa
            loglik_draws[store_idx] = am_bvar_sv_log_likelihood_proxy(Y, X, B, A, H)
            if H_draws is not None:
                H_draws[store_idx] = H
            store_idx += 1
        if verbose and (iteration == 1 or iteration % 500 == 0 or iteration == n_iter):
            accept_rate = diagnostics["kappa_accepts"] / max(diagnostics["kappa_attempts"], 1)
            print(
                f"Gibbs iteration {iteration:>6}/{n_iter}: retained={store_idx}, "
                f"kappa=({kappa[0]:.4f}, {kappa[1]:.5f}), accept={accept_rate:.2f}"
            )

    diagnostics["kappa_acceptance_rate"] = diagnostics["kappa_accepts"] / max(diagnostics["kappa_attempts"], 1)
    diagnostics["post_burn_kappa_acceptance_rate"] = diagnostics["post_burn_kappa_accepts"] / max(
        diagnostics["post_burn_kappa_attempts"],
        1,
    )
    diagnostics["final_kappa_proposal_sd_1"] = float(proposal_sd[0])
    diagnostics["final_kappa_proposal_sd_2"] = float(proposal_sd[1])
    beta_mean = B_draws.mean(axis=0)
    out = {
        "p": p,
        "variables": list(variables),
        "periods": init["periods"],
        "B_draws": B_draws,
        "A_draws": A_draws,
        "H_T_draws": H_T_draws,
        "Phi_draws": Phi_draws,
        "kappa_draws": kappa_draws,
        "loglik_draws": loglik_draws,
        "beta_mean": beta_mean,
        "A_mean": A_draws.mean(axis=0),
        "Phi_mean": Phi_draws.mean(axis=0),
        "H_T_mean": H_T_draws.mean(axis=0),
        "residual_variances": residual_variances,
        "diagnostics": diagnostics,
        "n_iter": int(n_iter),
        "burn": int(burn),
        "thin": int(thin),
        "retained_draws": int(retained),
        "train_last_period": train_df.index[-1],
    }
    if H_draws is not None:
        out["H_draws"] = H_draws
    return out


def forecast_am_bvar_sv_from_gibbs_draws(
    train_df: pd.DataFrame,
    posterior_draws: dict,
    horizon: int,
    *,
    trajectories_per_draw: int = 5,
    seed: int = 20260519,
    rng=None,
    return_draws: bool = False,
):
    variables = list(posterior_draws.get("variables", VAR_ORDER))
    rng = np.random.default_rng(seed) if rng is None else rng
    history0 = train_df[variables].dropna(how="any").to_numpy(dtype=float)
    B_draws = posterior_draws["B_draws"]
    A_draws = posterior_draws["A_draws"]
    H_T_draws = posterior_draws["H_T_draws"]
    Phi_draws = posterior_draws["Phi_draws"]
    retained, _, n = B_draws.shape
    total_draws = retained * int(trajectories_per_draw)
    forecast_draws = np.zeros((total_draws, horizon, n))
    draw_index = 0
    for m in range(retained):
        B = B_draws[m]
        A = A_draws[m]
        Phi = ensure_positive_definite(Phi_draws[m])
        phi_chol = cholesky_psd(Phi)
        for _ in range(int(trajectories_per_draw)):
            history = history0.copy()
            h_current = H_T_draws[m].copy()
            for h in range(horizon):
                h_current = h_current + phi_chol @ rng.standard_normal(n)
                structural_shock = np.exp(0.5 * np.clip(h_current, -50.0, 50.0)) * rng.standard_normal(n)
                reduced_form_shock = np.linalg.solve(A, structural_shock)
                row = lagged_forecast_row(history, posterior_draws["p"])
                y_next = row @ B + reduced_form_shock
                forecast_draws[draw_index, h, :] = y_next
                history = np.vstack([history, y_next])
            draw_index += 1
    result = {
        "model": posterior_draws,
        "mean": forecast_draws.mean(axis=0),
        "trajectories_per_draw": int(trajectories_per_draw),
    }
    if return_draws:
        result["draws"] = forecast_draws
    return result


def summarize_am_bvar_sv_draws(
    posterior_draws: dict,
    *,
    origin=None,
    model_name: str = "AM-BVAR-SV",
    model_system: str | None = None,
    trajectories_per_draw: int | None = None,
):
    variables = list(posterior_draws["variables"])
    kappa_draws = posterior_draws["kappa_draws"]
    Phi_draws = posterior_draws["Phi_draws"]
    A_draws = posterior_draws["A_draws"]
    diagnostics = posterior_draws.get("diagnostics", {})
    row = {
        "origin": str(origin) if origin is not None else str(posterior_draws.get("train_last_period", "")),
        "model": model_name,
        "model_system": model_system or f"{len(variables)}_variable_system",
        "n_variables": len(variables),
        "variables": ", ".join(variables),
        "retained_draws": int(posterior_draws["retained_draws"]),
        "burn": int(posterior_draws["burn"]),
        "thin": int(posterior_draws["thin"]),
        "trajectories_per_draw": int(trajectories_per_draw) if trajectories_per_draw is not None else np.nan,
        "mean_kappa_1": float(np.mean(kappa_draws[:, 0])),
        "median_kappa_1": float(np.median(kappa_draws[:, 0])),
        "p05_kappa_1": float(np.quantile(kappa_draws[:, 0], 0.05)),
        "p95_kappa_1": float(np.quantile(kappa_draws[:, 0], 0.95)),
        "mean_kappa_2": float(np.mean(kappa_draws[:, 1])),
        "median_kappa_2": float(np.median(kappa_draws[:, 1])),
        "p05_kappa_2": float(np.quantile(kappa_draws[:, 1], 0.05)),
        "p95_kappa_2": float(np.quantile(kappa_draws[:, 1], 0.95)),
        "kappa_acceptance_rate": float(diagnostics.get("post_burn_kappa_acceptance_rate", diagnostics.get("kappa_acceptance_rate", np.nan))),
        "cholesky_jitter_events": int(diagnostics.get("cholesky_jitter_events", 0)),
        "mean_log_likelihood_proxy": float(np.mean(posterior_draws["loglik_draws"])),
    }
    for i in range(len(variables)):
        row[f"mean_phi_{i + 1}{i + 1}"] = float(np.mean(Phi_draws[:, i, i]))
    for i in range(1, len(variables)):
        for j in range(i):
            row[f"mean_a{i + 1}{j + 1}"] = float(np.mean(A_draws[:, i, j]))
    mean_phi = ensure_positive_definite(np.mean(Phi_draws, axis=0))
    phi_sd = np.sqrt(np.clip(np.diag(mean_phi), 1e-12, None))
    phi_corr = mean_phi / np.outer(phi_sd, phi_sd)
    for i in range(1, len(variables)):
        for j in range(i):
            row[f"mean_phi_corr_{i + 1}{j + 1}"] = float(phi_corr[i, j])
    accept_rate = row["kappa_acceptance_rate"]
    finite_checks = np.all(np.isfinite(kappa_draws)) and np.all(np.isfinite(Phi_draws)) and np.all(np.isfinite(A_draws))
    row["convergence_flag"] = "ok" if finite_checks and (0.05 <= accept_rate <= 0.80) else "check"
    return row


def am_bvar_sv_trace_dataframe(posterior_draws: dict) -> pd.DataFrame:
    variables = list(posterior_draws["variables"])
    retained = posterior_draws["retained_draws"]
    data = {
        "draw": np.arange(1, retained + 1),
        "kappa_1": posterior_draws["kappa_draws"][:, 0],
        "kappa_2": posterior_draws["kappa_draws"][:, 1],
        "log_likelihood_proxy": posterior_draws["loglik_draws"],
    }
    Phi = posterior_draws["Phi_draws"]
    A = posterior_draws["A_draws"]
    for i, var in enumerate(variables):
        data[f"phi_{var}"] = Phi[:, i, i]
    for i in range(1, len(variables)):
        for j in range(i):
            data[f"a{i + 1}{j + 1}"] = A[:, i, j]
    return pd.DataFrame(data)


def summarize_am_bvar_sv_volatility(posterior_draws: dict) -> pd.DataFrame:
    if "H_draws" not in posterior_draws:
        return pd.DataFrame()
    H_draws = posterior_draws["H_draws"]
    periods = posterior_draws["periods"]
    variables = posterior_draws["variables"]
    volatility = np.exp(0.5 * np.clip(H_draws, -50.0, 50.0))
    rows = []
    for t, period in enumerate(periods):
        for j, var in enumerate(variables):
            values = volatility[:, t, j]
            rows.append(
                {
                    "quarter": str(period),
                    "variable": var,
                    "variable_name": VAR_LABELS[var],
                    "median_volatility": float(np.quantile(values, 0.50)),
                    "p16_volatility": float(np.quantile(values, 0.16)),
                    "p84_volatility": float(np.quantile(values, 0.84)),
                    "p05_volatility": float(np.quantile(values, 0.05)),
                    "p95_volatility": float(np.quantile(values, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def fit_bvar_minnesota_sv(
    train_df: pd.DataFrame,
    p: int = P_LAGS,
    variables=None,
    lambda_grid=None,
    decay: float = 1.0,
    cross_weight: float = 0.5,
    own_first_lag_mean: float = 0.0,
):
    variables = list(VAR_ORDER if variables is None else variables)
    if lambda_grid is None:
        lambda_grid = np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.25])
    y = train_df[variables].to_numpy(dtype=float)
    x, target = make_lagged_xy(y, p)
    n = target.shape[1]
    resid_vars = ar_residual_variances(y, p=p)
    a0 = 2.5
    best_score = -np.inf
    best_lambda = None
    best_posts = None
    for lam in lambda_grid:
        score = 0.0
        posts = []
        for i in range(n):
            b0, v0 = minnesota_prior_for_equation(
                i,
                n,
                p,
                resid_vars,
                lam,
                decay=decay,
                cross_weight=cross_weight,
                own_first_lag_mean=own_first_lag_mean,
            )
            d0 = max((a0 - 1.0) * resid_vars[i], 1e-8)
            post = posterior_nig(target[:, i], x, b0, v0, a0, d0)
            score += post["log_ml"]
            posts.append(post)
        if score > best_score:
            best_score = score
            best_lambda = float(lam)
            best_posts = posts
    beta_mean = np.column_stack([post["bn"] for post in best_posts])
    residuals = target - x @ beta_mean
    sv_state = estimate_stochastic_volatility_state(residuals)
    return {
        "p": p,
        "variables": variables,
        "posts": best_posts,
        "beta_mean": beta_mean,
        "lambda": best_lambda,
        "log_marginal_likelihood": best_score,
        "residual_variances": resid_vars,
        "sv_state": sv_state,
        "train_last_period": train_df.index[-1],
    }


def draw_bvar_coefficients(model, rng):
    coeffs = []
    for post in model["posts"]:
        sigma2 = inverse_gamma_draw(post["an"], post["dn"], 1, rng)[0]
        beta = rng.multivariate_normal(post["bn"], sigma2 * post["vn"])
        coeffs.append(beta)
    return np.column_stack(coeffs)


def forecast_bvar_sv(
    train_df: pd.DataFrame,
    horizon: int,
    draws: int,
    rng,
    model=None,
    variables=None,
    return_draws: bool = False,
):
    if model is None:
        model = fit_bvar_minnesota_sv(train_df, variables=variables)
    variables = list(model.get("variables", VAR_ORDER if variables is None else variables))
    history0 = train_df[variables].to_numpy(dtype=float)
    n = history0.shape[1]
    forecast_draws = np.zeros((draws, horizon, n))
    sv = model["sv_state"]
    for m in range(draws):
        beta = draw_bvar_coefficients(model, rng)
        history = history0.copy()
        logv = sv["last_log_var"].copy()
        for h in range(horizon):
            row = lagged_forecast_row(history, model["p"])
            mean = row @ beta
            logv = logv + sv["sv_scales"] * rng.standard_normal(n)
            logv = np.clip(logv, sv["logvar_floor"], sv["logvar_ceiling"])
            correlated_z = sv["chol_corr"] @ rng.standard_normal(n)
            shock = np.sqrt(np.exp(logv)) * correlated_z
            y_next = mean + shock
            forecast_draws[m, h, :] = y_next
            history = np.vstack([history, y_next])
    result = {"model": model, "mean": forecast_draws.mean(axis=0)}
    if return_draws:
        result["draws"] = forecast_draws
    return result


def systematic_resample(weights, rng):
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def ucsv_starting_values(y):
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    scale = np.nanvar(dy, ddof=1) if len(dy) > 1 else np.nanvar(y, ddof=1)
    scale = max(float(scale), 1e-4)
    return {
        "tau0": float(y[0]),
        "obs_var0": max(0.7 * scale, 1e-4),
        "trend_var0": max(0.03 * scale, 1e-5),
        "obs_sv_scale": 0.18,
        "trend_sv_scale": 0.10,
    }


def normalize_log_weights(logw):
    max_logw = np.max(logw)
    weights = np.exp(logw - max_logw)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def ucsv_particle_forecast(
    y_train,
    horizon: int,
    n_particles: int,
    n_draws: int,
    rng,
    sv_scale_multiplier: float = 1.0,
    return_draws: bool = False,
):
    y = np.asarray(y_train, dtype=float)
    if len(y) < 5:
        raise ValueError("UC-SV needs at least a few observations")
    init = ucsv_starting_values(y)
    n = n_particles
    tau = rng.normal(init["tau0"], np.sqrt(init["obs_var0"]), size=n)
    logh_obs = rng.normal(np.log(init["obs_var0"]), 0.20, size=n)
    logh_trend = rng.normal(np.log(init["trend_var0"]), 0.20, size=n)
    weights = np.ones(n) / n
    obs_sv_scale = init["obs_sv_scale"] * sv_scale_multiplier
    trend_sv_scale = init["trend_sv_scale"] * sv_scale_multiplier
    floor = np.log(1e-5)
    ceiling = np.log(max(100.0 * np.nanvar(y, ddof=1), 1.0))

    for t, obs in enumerate(y):
        if t > 0:
            logh_obs = np.clip(logh_obs + obs_sv_scale * rng.standard_normal(n), floor, ceiling)
            logh_trend = np.clip(logh_trend + trend_sv_scale * rng.standard_normal(n), floor, ceiling)
            tau = tau + np.sqrt(np.exp(logh_trend)) * rng.standard_normal(n)
        variance = np.exp(logh_obs)
        logw = -0.5 * (np.log(2.0 * math.pi) + logh_obs + ((obs - tau) ** 2) / variance)
        weights = normalize_log_weights(logw)
        ess = 1.0 / np.sum(weights**2)
        if ess < 0.5 * n:
            idx = systematic_resample(weights, rng)
            tau = tau[idx]
            logh_obs = logh_obs[idx]
            logh_trend = logh_trend[idx]
            weights = np.ones(n) / n

    draw_idx = rng.choice(n, size=n_draws, replace=True, p=weights)
    tau_d = tau[draw_idx].copy()
    logh_obs_d = logh_obs[draw_idx].copy()
    logh_trend_d = logh_trend[draw_idx].copy()
    draws = np.zeros((n_draws, horizon))
    for h in range(horizon):
        logh_obs_d = np.clip(logh_obs_d + obs_sv_scale * rng.standard_normal(n_draws), floor, ceiling)
        logh_trend_d = np.clip(logh_trend_d + trend_sv_scale * rng.standard_normal(n_draws), floor, ceiling)
        tau_d = tau_d + np.sqrt(np.exp(logh_trend_d)) * rng.standard_normal(n_draws)
        draws[:, h] = tau_d + np.sqrt(np.exp(logh_obs_d)) * rng.standard_normal(n_draws)

    result = {"mean": draws.mean(axis=0)}
    if return_draws:
        result["draws"] = draws
    return result


def forecast_ucsv_panel(
    train_df: pd.DataFrame,
    horizon: int,
    n_particles: int,
    n_draws: int,
    rng,
    variables=None,
    return_draws: bool = False,
):
    variables = list(VAR_ORDER if variables is None else variables)
    n_vars = len(variables)
    means = np.zeros((horizon, n_vars))
    all_draws = np.zeros((n_draws, horizon, n_vars)) if return_draws else None
    for j, var in enumerate(variables):
        result = ucsv_particle_forecast(
            train_df[var].to_numpy(dtype=float),
            horizon=horizon,
            n_particles=n_particles,
            n_draws=n_draws,
            rng=rng,
            return_draws=return_draws,
        )
        means[:, j] = result["mean"]
        if return_draws:
            all_draws[:, :, j] = result["draws"]
    out = {"mean": means}
    if return_draws:
        out["draws"] = all_draws
    return out


def fitted_values_bvar(train_df: pd.DataFrame, model=None, p: int = P_LAGS):
    if model is None:
        model = fit_bvar_minnesota_sv(train_df, p=p)
    variables = list(model.get("variables", VAR_ORDER))
    y = train_df[variables].to_numpy(dtype=float)
    x, _ = make_lagged_xy(y, model["p"])
    fitted = x @ model["beta_mean"]
    return pd.DataFrame(fitted, index=train_df.index[model["p"]:], columns=variables)


def fitted_values_ar_panel(train_df: pd.DataFrame, p: int = P_LAGS, variables=None):
    variables = list(VAR_ORDER if variables is None else variables)
    fitted = pd.DataFrame(index=train_df.index[p:], columns=variables, dtype=float)
    for var in variables:
        y = train_df[var].to_numpy(dtype=float)
        x, _ = make_lagged_xy(y[:, None], p)
        fit = fit_ar_ols(y, p=p)
        fitted[var] = x @ fit["beta"]
    return fitted


def ucsv_filtered_mean(
    y_train,
    n_particles: int,
    rng,
    sv_scale_multiplier: float = 1.0,
):
    y = np.asarray(y_train, dtype=float)
    if len(y) < 5:
        raise ValueError("UC-SV needs at least a few observations")
    init = ucsv_starting_values(y)
    n = n_particles
    tau = rng.normal(init["tau0"], np.sqrt(init["obs_var0"]), size=n)
    logh_obs = rng.normal(np.log(init["obs_var0"]), 0.20, size=n)
    logh_trend = rng.normal(np.log(init["trend_var0"]), 0.20, size=n)
    weights = np.ones(n) / n
    obs_sv_scale = init["obs_sv_scale"] * sv_scale_multiplier
    trend_sv_scale = init["trend_sv_scale"] * sv_scale_multiplier
    floor = np.log(1e-5)
    ceiling = np.log(max(100.0 * np.nanvar(y, ddof=1), 1.0))
    filtered = np.zeros(len(y))

    for t, obs in enumerate(y):
        if t > 0:
            logh_obs = np.clip(logh_obs + obs_sv_scale * rng.standard_normal(n), floor, ceiling)
            logh_trend = np.clip(logh_trend + trend_sv_scale * rng.standard_normal(n), floor, ceiling)
            tau = tau + np.sqrt(np.exp(logh_trend)) * rng.standard_normal(n)
        variance = np.exp(logh_obs)
        logw = -0.5 * (np.log(2.0 * math.pi) + logh_obs + ((obs - tau) ** 2) / variance)
        weights = normalize_log_weights(logw)
        filtered[t] = float(np.sum(weights * tau))
        ess = 1.0 / np.sum(weights**2)
        if ess < 0.5 * n:
            idx = systematic_resample(weights, rng)
            tau = tau[idx]
            logh_obs = logh_obs[idx]
            logh_trend = logh_trend[idx]
            weights = np.ones(n) / n

    return filtered


def fitted_values_ucsv_panel(
    train_df: pd.DataFrame,
    n_particles: int,
    rng,
    p: int = P_LAGS,
    variables=None,
):
    variables = list(VAR_ORDER if variables is None else variables)
    fitted = pd.DataFrame(index=train_df.index, columns=variables, dtype=float)
    for var in variables:
        fitted[var] = ucsv_filtered_mean(
            train_df[var].to_numpy(dtype=float),
            n_particles=n_particles,
            rng=rng,
        )
    return fitted.loc[train_df.index[p:]]


def fitted_panel_to_records(model_name: str, fitted_df: pd.DataFrame, actual_df: pd.DataFrame, variables=None):
    variables = list(fitted_df.columns if variables is None else variables)
    records = []
    for quarter in fitted_df.index:
        for var in variables:
            records.append(
                {
                    "model": model_name,
                    "quarter": str(quarter),
                    "variable": var,
                    "variable_name": VAR_LABELS[var],
                    "actual": float(actual_df.loc[quarter, var]),
                    "fitted": float(fitted_df.loc[quarter, var]),
                }
            )
    return records


def make_fitted_value_table(
    growth_df: pd.DataFrame,
    *,
    bvar_model=None,
    seed: int = 20260519,
    ucsv_particles: int = 1000,
    p: int = P_LAGS,
):
    train_df = growth_df[VAR_ORDER].dropna(how="any")
    rng = np.random.default_rng(seed + 202)
    bvar_fitted = fitted_values_bvar(train_df, model=bvar_model, p=p)
    ar_fitted = fitted_values_ar_panel(train_df, p=p)
    ucsv_fitted = fitted_values_ucsv_panel(train_df, n_particles=ucsv_particles, rng=rng, p=p)
    records = []
    records.extend(fitted_panel_to_records("AM-BVAR-SV", bvar_fitted, train_df))
    records.extend(fitted_panel_to_records("AR(4)", ar_fitted, train_df))
    records.extend(fitted_panel_to_records("UC-SV", ucsv_fitted, train_df))
    return pd.DataFrame(records)


def recursive_origins(growth_df: pd.DataFrame, first_origin: pd.Period, min_horizon: int = 1):
    latest = growth_df.index.max()
    candidates = [p for p in growth_df.index if first_origin <= p <= latest - min_horizon]
    if not candidates:
        raise ValueError(f"No recursive origins available from {first_origin} through {latest - min_horizon}")
    return candidates


def forecasts_to_records(model_name, origin, forecast_matrix, growth_df, variables=None):
    variables = list(VAR_ORDER if variables is None else variables)
    records = []
    max_h = forecast_matrix.shape[0]
    for h in range(1, max_h + 1):
        target = origin + h
        for j, var in enumerate(variables):
            actual = growth_df.loc[target, var] if target in growth_df.index else np.nan
            records.append(
                {
                    "model": model_name,
                    "origin": str(origin),
                    "target_quarter": str(target),
                    "horizon": h,
                    "variable": var,
                    "variable_name": VAR_LABELS[var],
                    "forecast": float(forecast_matrix[h - 1, j]),
                    "actual": float(actual) if np.isfinite(actual) else np.nan,
                }
            )
    return records


def run_recursive_evaluation(
    growth_df: pd.DataFrame,
    first_origin: pd.Period,
    horizons,
    *,
    seed: int = 20260519,
    bvar_n_iter: int = DEFAULT_GIBBS_N_ITER_RECURSIVE,
    bvar_burn: int = DEFAULT_GIBBS_BURN_RECURSIVE,
    bvar_thin: int = DEFAULT_GIBBS_THIN_RECURSIVE,
    bvar_trajectories_per_draw: int = DEFAULT_GIBBS_TRAJECTORIES_RECURSIVE,
    ucsv_particles: int = 500,
    ucsv_draws: int = 500,
    progress: bool = True,
):
    rng = np.random.default_rng(seed)
    max_h = max(horizons)
    origins = recursive_origins(growth_df, first_origin, min_horizon=1)
    records = []
    hyperparameter_records = []
    total = len(origins)
    for idx, origin in enumerate(origins, start=1):
        train = growth_df.loc[:origin, VAR_ORDER]
        if progress and (idx == 1 or idx % 5 == 0 or idx == total):
            print(f"Origin {idx:>3}/{total}: {origin}, train n={len(train)}")

        origin_seed = seed + idx * 1000
        bvar_model = fit_am_bvar_sv_gibbs(
            train,
            p=P_LAGS,
            variables=VAR_ORDER,
            n_iter=bvar_n_iter,
            burn=bvar_burn,
            thin=bvar_thin,
            seed=origin_seed,
        )
        bvar_result = forecast_am_bvar_sv_from_gibbs_draws(
            train,
            posterior_draws=bvar_model,
            horizon=max_h,
            trajectories_per_draw=bvar_trajectories_per_draw,
            seed=origin_seed + 100000,
        )
        ar_forecast = forecast_ar_panel(train, max_horizon=max_h, p=P_LAGS)
        ucsv_forecast = forecast_ucsv_panel(
            train,
            horizon=max_h,
            n_particles=ucsv_particles,
            n_draws=ucsv_draws,
            rng=rng,
        )["mean"]

        records.extend(forecasts_to_records("AM-BVAR-SV", origin, bvar_result["mean"], growth_df))
        records.extend(forecasts_to_records("AR(4)", origin, ar_forecast, growth_df))
        records.extend(forecasts_to_records("UC-SV", origin, ucsv_forecast, growth_df))
        hyperparameter_records.append(
            summarize_am_bvar_sv_draws(
                bvar_model,
                origin=origin,
                model_name="AM-BVAR-SV",
                model_system="five_variable_with_wage",
                trajectories_per_draw=bvar_trajectories_per_draw,
            )
        )

    forecast_df = pd.DataFrame(records)
    forecast_df = forecast_df[forecast_df["horizon"].isin(horizons)].reset_index(drop=True)
    hyperparameter_df = pd.DataFrame(hyperparameter_records)
    return forecast_df, hyperparameter_df


def compute_rmse_table(forecast_df: pd.DataFrame):
    eval_df = forecast_df.dropna(subset=["actual", "forecast"]).copy()
    rows = []
    for (model, var, horizon), group in eval_df.groupby(["model", "variable", "horizon"], sort=True):
        rows.append(
            {
                "model": model,
                "variable": var,
                "variable_name": VAR_LABELS[var],
                "horizon": int(horizon),
                "n_forecasts": int(len(group)),
                "RMSE": rmse(group["actual"], group["forecast"]),
            }
        )
    rmse_df = pd.DataFrame(rows)
    baseline = rmse_df.loc[rmse_df["model"] == "AM-BVAR-SV", ["variable", "horizon", "RMSE"]]
    baseline = baseline.rename(columns={"RMSE": "AM_BVAR_SV_RMSE"})
    rmse_df = rmse_df.merge(baseline, on=["variable", "horizon"], how="left")
    rmse_df["Relative RMSE versus AM-BVAR-SV"] = rmse_df["RMSE"] / rmse_df["AM_BVAR_SV_RMSE"]
    return rmse_df.sort_values(["horizon", "variable", "model"]).reset_index(drop=True)


def normal_p_value(t_stat: float, alternative: str = "greater") -> float:
    if not np.isfinite(t_stat):
        return np.nan
    if alternative == "greater":
        return float(0.5 * math.erfc(t_stat / math.sqrt(2.0)))
    if alternative == "less":
        return float(0.5 * math.erfc(-t_stat / math.sqrt(2.0)))
    if alternative == "two-sided":
        return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")


def newey_west_lrv(values, lags: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= 1:
        return np.nan
    x = x - np.mean(x)
    max_lag = min(int(lags), n - 1)
    lrv = float(np.dot(x, x) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(x[lag:], x[:-lag]) / n)
        lrv += 2.0 * weight * gamma
    return max(lrv, 0.0)


def loss_differential_test(values, nw_lags: int, alternative: str = "greater") -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {
            "mean_loss_diff": np.nan,
            "newey_west_lrv": np.nan,
            "standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n": 0,
            "nw_lags": int(nw_lags),
        }
    dbar = float(np.mean(x))
    lrv = newey_west_lrv(x, nw_lags)
    if not np.isfinite(lrv) or lrv <= 0.0 or n <= 1:
        se = np.nan
        t_stat = np.nan
        p_value = np.nan
    else:
        se = math.sqrt(lrv / n)
        t_stat = dbar / se
        p_value = normal_p_value(t_stat, alternative=alternative)
    return {
        "mean_loss_diff": dbar,
        "newey_west_lrv": lrv,
        "standard_error": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "n": int(n),
        "nw_lags": int(min(nw_lags, max(n - 1, 0))),
    }


def p_value_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def wage_result_conclusion(relative_rmse: float, p_value: float, close_threshold: float = 0.01) -> str:
    if not np.isfinite(relative_rmse):
        return "Insufficient data"
    if abs(relative_rmse - 1.0) <= close_threshold:
        return "No clear gain"
    if relative_rmse < 1.0 and np.isfinite(p_value) and p_value < 0.10:
        return "Wage helps"
    if relative_rmse < 1.0:
        return "Weak evidence wage helps"
    return "Wage hurts"


def make_wage_comparison_panel(forecast_df: pd.DataFrame) -> pd.DataFrame:
    needed_models = [WAGE_MODEL_NAME, NO_WAGE_MODEL_NAME]
    eval_df = forecast_df.loc[forecast_df["model"].isin(needed_models)].dropna(subset=["actual", "forecast"]).copy()
    if eval_df.empty:
        return pd.DataFrame()
    pivot = eval_df.pivot_table(
        index=["origin", "target_quarter", "horizon", "variable", "variable_name", "actual"],
        columns="model",
        values="forecast",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.dropna(subset=needed_models).copy()
    pivot["origin_period"] = pd.PeriodIndex(pivot["origin"], freq="Q")
    pivot["target_period"] = pd.PeriodIndex(pivot["target_quarter"], freq="Q")
    pivot["error_with_wage"] = pivot["actual"] - pivot[WAGE_MODEL_NAME]
    pivot["error_without_wage"] = pivot["actual"] - pivot[NO_WAGE_MODEL_NAME]
    pivot["squared_error_with_wage"] = pivot["error_with_wage"] ** 2
    pivot["squared_error_without_wage"] = pivot["error_without_wage"] ** 2
    pivot["dmw_loss_diff"] = pivot["squared_error_without_wage"] - pivot["squared_error_with_wage"]
    pivot["cw_loss_diff"] = pivot["squared_error_without_wage"] - (
        pivot["squared_error_with_wage"] - (pivot[NO_WAGE_MODEL_NAME] - pivot[WAGE_MODEL_NAME]) ** 2
    )
    return pivot.sort_values(["variable", "horizon", "origin_period"]).reset_index(drop=True)


def compute_wage_growth_predictive_power_table(
    forecast_df: pd.DataFrame,
    *,
    sample_name: str | None = None,
    alternative: str = "greater",
) -> pd.DataFrame:
    comparison = make_wage_comparison_panel(forecast_df)
    rows = []
    if comparison.empty:
        return pd.DataFrame(rows)
    for (var, horizon), group in comparison.groupby(["variable", "horizon"], sort=True):
        horizon = int(horizon)
        rmse_with = rmse(group["actual"], group[WAGE_MODEL_NAME])
        rmse_without = rmse(group["actual"], group[NO_WAGE_MODEL_NAME])
        relative = rmse_with / rmse_without if rmse_without > 0 else np.nan
        gain = 100.0 * (1.0 - relative) if np.isfinite(relative) else np.nan
        nw_lags = horizon + 1
        dmw = loss_differential_test(group["dmw_loss_diff"], nw_lags=nw_lags, alternative=alternative)
        cw = loss_differential_test(group["cw_loss_diff"], nw_lags=nw_lags, alternative=alternative)
        row = {
            "target_variable": var,
            "target_variable_name": VAR_LABELS[var],
            "horizon": horizon,
            "rmse_with_wage": rmse_with,
            "rmse_without_wage": rmse_without,
            "relative_rmse_with_vs_without_wage": relative,
            "rmse_gain_from_wage_growth_pct": gain,
            "n_forecast_origins": int(len(group)),
            "newey_west_lags": dmw["nw_lags"],
            "dmw_mean_loss_diff": dmw["mean_loss_diff"],
            "dmw_t_stat": dmw["t_stat"],
            "dmw_one_sided_p_value": dmw["p_value"],
            "dmw_significance": p_value_stars(dmw["p_value"]),
            "cw_mean_loss_diff": cw["mean_loss_diff"],
            "cw_t_stat": cw["t_stat"],
            "cw_one_sided_p_value": cw["p_value"],
            "cw_significance": p_value_stars(cw["p_value"]),
            "conclusion": wage_result_conclusion(relative, dmw["p_value"]),
        }
        if sample_name is not None:
            row["sample"] = sample_name
        rows.append(row)
    table = pd.DataFrame(rows)
    sort_cols = ["target_variable", "horizon"]
    if sample_name is not None:
        sort_cols = ["sample"] + sort_cols
    return table.sort_values(sort_cols).reset_index(drop=True)


def filter_wage_forecasts_by_outcome_window(
    forecast_df: pd.DataFrame,
    *,
    start: pd.Period | None = None,
    end: pd.Period | None = None,
) -> pd.DataFrame:
    if forecast_df.empty:
        return forecast_df.copy()
    out = forecast_df.copy()
    target_period = pd.PeriodIndex(out["target_quarter"], freq="Q")
    mask = np.ones(len(out), dtype=bool)
    if start is not None:
        mask &= target_period >= start
    if end is not None:
        mask &= target_period <= end
    return out.loc[mask].copy()


def compute_wage_growth_subsample_tables(forecast_df: pd.DataFrame) -> pd.DataFrame:
    windows = [
        ("Full out-of-sample", None, None),
        ("Pre-pandemic outcomes", None, pd.Period("2019Q4", freq="Q")),
        ("Pandemic/post-pandemic outcomes", pd.Period("2020Q1", freq="Q"), None),
    ]
    tables = []
    for name, start, end in windows:
        subset = filter_wage_forecasts_by_outcome_window(forecast_df, start=start, end=end)
        table = compute_wage_growth_predictive_power_table(subset, sample_name=name)
        if not table.empty:
            tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def wage_cumulative_loss_difference(
    forecast_df: pd.DataFrame,
    *,
    variables=None,
    horizons=(4, 8),
) -> pd.DataFrame:
    variables = list(INFLATION_TARGETS if variables is None else variables)
    horizons = set(int(h) for h in horizons)
    comparison = make_wage_comparison_panel(forecast_df)
    if comparison.empty:
        return comparison
    comparison = comparison.loc[
        comparison["variable"].isin(variables) & comparison["horizon"].isin(horizons)
    ].copy()
    if comparison.empty:
        return comparison
    comparison = comparison.sort_values(["variable", "horizon", "origin_period"])
    comparison["cumulative_loss_diff"] = comparison.groupby(["variable", "horizon"])["dmw_loss_diff"].cumsum()
    return comparison.reset_index(drop=True)


def run_wage_growth_predictive_power_evaluation(
    growth_df: pd.DataFrame,
    first_origin: pd.Period,
    horizons,
    *,
    seed: int = 20260519,
    bvar_n_iter: int = DEFAULT_GIBBS_N_ITER_RECURSIVE,
    bvar_burn: int = DEFAULT_GIBBS_BURN_RECURSIVE,
    bvar_thin: int = DEFAULT_GIBBS_THIN_RECURSIVE,
    bvar_trajectories_per_draw: int = DEFAULT_GIBBS_TRAJECTORIES_RECURSIVE,
    progress: bool = True,
):
    max_h = max(horizons)
    origins = recursive_origins(growth_df[VAR_ORDER].dropna(how="any"), first_origin, min_horizon=1)
    records = []
    hyperparameter_records = []
    total = len(origins)
    for idx, origin in enumerate(origins, start=1):
        train_with_wage = growth_df.loc[:origin, VAR_ORDER]
        train_without_wage = growth_df.loc[:origin, NO_WAGE_VAR_ORDER]
        if progress and (idx == 1 or idx % 5 == 0 or idx == total):
            print(f"Wage extension origin {idx:>3}/{total}: {origin}, train n={len(train_with_wage)}")

        origin_seed = seed + idx * 2000
        model_with_wage = fit_am_bvar_sv_gibbs(
            train_with_wage,
            p=P_LAGS,
            variables=VAR_ORDER,
            n_iter=bvar_n_iter,
            burn=bvar_burn,
            thin=bvar_thin,
            seed=origin_seed,
        )
        forecast_with_wage = forecast_am_bvar_sv_from_gibbs_draws(
            train_with_wage,
            posterior_draws=model_with_wage,
            horizon=max_h,
            trajectories_per_draw=bvar_trajectories_per_draw,
            seed=origin_seed + 100000,
        )
        model_without_wage = fit_am_bvar_sv_gibbs(
            train_without_wage,
            p=P_LAGS,
            variables=NO_WAGE_VAR_ORDER,
            n_iter=bvar_n_iter,
            burn=bvar_burn,
            thin=bvar_thin,
            seed=origin_seed + 500,
        )
        forecast_without_wage = forecast_am_bvar_sv_from_gibbs_draws(
            train_without_wage,
            posterior_draws=model_without_wage,
            horizon=max_h,
            trajectories_per_draw=bvar_trajectories_per_draw,
            seed=origin_seed + 100500,
        )

        with_wage_records = forecasts_to_records(
            WAGE_MODEL_NAME,
            origin,
            forecast_with_wage["mean"],
            growth_df,
            variables=VAR_ORDER,
        )
        records.extend([record for record in with_wage_records if record["variable"] in INFLATION_TARGETS])
        records.extend(
            forecasts_to_records(
                NO_WAGE_MODEL_NAME,
                origin,
                forecast_without_wage["mean"],
                growth_df,
                variables=NO_WAGE_VAR_ORDER,
            )
        )
        hyperparameter_records.extend(
            [
                summarize_am_bvar_sv_draws(
                    model_with_wage,
                    origin=origin,
                    model_name=WAGE_MODEL_NAME,
                    model_system="five_variable_with_wage",
                    trajectories_per_draw=bvar_trajectories_per_draw,
                ),
                summarize_am_bvar_sv_draws(
                    model_without_wage,
                    origin=origin,
                    model_name=NO_WAGE_MODEL_NAME,
                    model_system="four_variable_without_wage",
                    trajectories_per_draw=bvar_trajectories_per_draw,
                ),
            ]
        )

    forecast_df = pd.DataFrame(records)
    forecast_df = forecast_df[forecast_df["horizon"].isin(horizons)].reset_index(drop=True)
    hyperparameter_df = pd.DataFrame(hyperparameter_records)
    return forecast_df, hyperparameter_df


def reconstruct_recursive_index_paths(index_df: pd.DataFrame, forecast_df: pd.DataFrame):
    rows = []
    for (model, origin_text, var), group in forecast_df.groupby(["model", "origin", "variable"], sort=True):
        origin = pd.Period(origin_text, freq="Q")
        if origin not in index_df.index:
            continue
        group = group.sort_values("horizon")
        previous_level = float(index_df.loc[origin, var])
        previous_quarter = origin
        for _, row in group.iterrows():
            horizon = int(row["horizon"])
            target = origin + horizon
            forecast_growth = float(row["forecast"])
            forecast_level = previous_level * math.exp(forecast_growth / 400.0)
            implied_growth = 400.0 * math.log(forecast_level / previous_level)
            rows.append(
                {
                    "model": model,
                    "origin": str(origin),
                    "target_quarter": str(target),
                    "horizon": horizon,
                    "variable": var,
                    "variable_name": VAR_LABELS[var],
                    "forecast_growth": forecast_growth,
                    "reconstructed_index_level": forecast_level,
                    "implied_growth_from_reconstructed_index": implied_growth,
                    "growth_reconstruction_error": implied_growth - forecast_growth,
                    "previous_quarter_used": str(previous_quarter),
                }
            )
            previous_level = forecast_level
            previous_quarter = target
    return pd.DataFrame(rows)


def forecast_index_from_growth_draws(growth_draws, anchor_levels):
    growth_draws = np.asarray(growth_draws, dtype=float)
    if growth_draws.ndim == 2:
        growth_draws = growth_draws[None, :, :]
    anchor = np.asarray(anchor_levels, dtype=float)
    draws, horizon, n_vars = growth_draws.shape
    index_paths = np.zeros((draws, horizon, n_vars))
    previous = np.tile(anchor, (draws, 1))
    for h in range(horizon):
        current = previous * np.exp(growth_draws[:, h, :] / 400.0)
        index_paths[:, h, :] = current
        previous = current
    return index_paths


def matrix_to_panel_df(matrix, periods, columns, model_name, value_type):
    df = pd.DataFrame(matrix, index=[str(p) for p in periods], columns=columns)
    df.index.name = "forecast_quarter"
    out = df.reset_index()
    out.insert(0, "model", model_name)
    out.insert(1, "value_type", value_type)
    return out


def summarize_draws(draws, periods, variables, labels=None):
    if labels is None:
        labels = VAR_LABELS
    probs = {
        "p05": 0.05,
        "p16": 0.16,
        "p25": 0.25,
        "median": 0.50,
        "p75": 0.75,
        "p84": 0.84,
        "p95": 0.95,
    }
    rows = []
    for h, period in enumerate(periods):
        for j, var in enumerate(variables):
            values = draws[:, h, j]
            record = {
                "forecast_quarter": str(period),
                "variable": var,
                "variable_name": labels[var],
                "mean": float(np.mean(values)),
            }
            for name, prob in probs.items():
                record[name] = float(np.quantile(values, prob))
            record["50_percent_interval"] = f"[{record['p25']:.3f}, {record['p75']:.3f}]"
            record["68_percent_interval"] = f"[{record['p16']:.3f}, {record['p84']:.3f}]"
            record["90_percent_interval"] = f"[{record['p05']:.3f}, {record['p95']:.3f}]"
            rows.append(record)
    return pd.DataFrame(rows)


def fit_final_forecasts(
    common_growth: pd.DataFrame,
    index_levels: pd.DataFrame,
    *,
    seed: int = 20260519,
    forecast_horizon: int = 7,
    bvar_n_iter: int = DEFAULT_GIBBS_N_ITER_FINAL,
    bvar_burn: int = DEFAULT_GIBBS_BURN_FINAL,
    bvar_thin: int = DEFAULT_GIBBS_THIN_FINAL,
    bvar_trajectories_per_draw: int = DEFAULT_GIBBS_TRAJECTORIES_FINAL,
    ucsv_particles: int = 2500,
    ucsv_draws: int = 5000,
):
    latest_common_quarter = common_growth.index.max()
    forecast_quarters = [latest_common_quarter + h for h in range(1, forecast_horizon + 1)]
    full_train = common_growth.loc[:latest_common_quarter, VAR_ORDER]
    anchor_levels = index_levels.loc[latest_common_quarter, VAR_ORDER].to_numpy(dtype=float)
    rng_final = np.random.default_rng(seed + 999)

    final_bvar_model = fit_am_bvar_sv_gibbs(
        full_train,
        p=P_LAGS,
        variables=VAR_ORDER,
        n_iter=bvar_n_iter,
        burn=bvar_burn,
        thin=bvar_thin,
        seed=seed + 777,
        store_h_path=True,
        verbose=False,
    )
    final_bvar = forecast_am_bvar_sv_from_gibbs_draws(
        full_train,
        posterior_draws=final_bvar_model,
        horizon=forecast_horizon,
        trajectories_per_draw=bvar_trajectories_per_draw,
        seed=seed + 888,
        return_draws=True,
    )
    final_ar_mean = forecast_ar_panel(full_train, max_horizon=forecast_horizon, p=P_LAGS)
    final_ucsv = forecast_ucsv_panel(
        full_train,
        horizon=forecast_horizon,
        n_particles=ucsv_particles,
        n_draws=ucsv_draws,
        rng=rng_final,
        return_draws=True,
    )

    final_growth_means = {
        "AM-BVAR-SV": final_bvar["mean"],
        "AR(4)": final_ar_mean,
        "UC-SV": final_ucsv["mean"],
    }
    final_growth_draws = {
        "AM-BVAR-SV": final_bvar["draws"],
        "UC-SV": final_ucsv["draws"],
    }
    final_index_means = {}
    final_index_draws = {}
    for model_name, growth_mean in final_growth_means.items():
        final_index_means[model_name] = forecast_index_from_growth_draws(growth_mean, anchor_levels)[0]
    for model_name, draws in final_growth_draws.items():
        final_index_draws[model_name] = forecast_index_from_growth_draws(draws, anchor_levels)

    return {
        "latest_common_quarter": latest_common_quarter,
        "forecast_quarters": forecast_quarters,
        "full_train": full_train,
        "anchor_levels": anchor_levels,
        "bvar_model": final_bvar_model,
        "growth_means": final_growth_means,
        "growth_draws": final_growth_draws,
        "index_means": final_index_means,
        "index_draws": final_index_draws,
    }


def build_final_output_tables(final_result: dict):
    forecast_quarters = final_result["forecast_quarters"]
    growth_panels = []
    index_panels = []
    for model_name in MODEL_NAMES:
        growth_panels.append(
            matrix_to_panel_df(
                final_result["growth_means"][model_name],
                forecast_quarters,
                [VAR_LABELS[v] for v in VAR_ORDER],
                model_name,
                "annualized_qoq_log_growth_rate",
            )
        )
        index_panels.append(
            matrix_to_panel_df(
                final_result["index_means"][model_name],
                forecast_quarters,
                [INDEX_LABELS[v] for v in VAR_ORDER],
                model_name,
                "implied_index_level",
            )
        )
    final_growth_table = pd.concat(growth_panels, ignore_index=True)
    final_index_table = pd.concat(index_panels, ignore_index=True)
    am_rate_intervals = summarize_draws(final_result["growth_draws"]["AM-BVAR-SV"], forecast_quarters, VAR_ORDER)
    am_index_intervals = summarize_draws(
        final_result["index_draws"]["AM-BVAR-SV"],
        forecast_quarters,
        VAR_ORDER,
        labels=INDEX_LABELS,
    )
    return final_growth_table, final_index_table, am_rate_intervals, am_index_intervals


def save_final_output_tables(output_dir: Path, final_growth_table, final_index_table, am_rate_intervals, am_index_intervals):
    output_dir.mkdir(exist_ok=True)
    final_growth_table.to_csv(output_dir / "final_forecast_annualized_qoq_growth_rates.csv", index=False)
    final_index_table.to_csv(output_dir / "final_forecast_implied_index_levels.csv", index=False)
    am_rate_intervals.to_csv(output_dir / "am_bvar_sv_rate_intervals.csv", index=False)
    am_index_intervals.to_csv(output_dir / "am_bvar_sv_index_intervals.csv", index=False)
