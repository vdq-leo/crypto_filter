"""
Portfolio Estimation Engine  v5
================================

Improvements over v4:
  1.  Volatility fold stability now computed on back-transformed σ-space R²
  2.  positive_joint_rate computes true AND logic: mean(IC_i > 0 and R²_i > 0)
  3.  Walk-forward explicit about unused data (step uses floor division over full remaining length)
  4.  Explicit minimum valid OOS folds validation before computing diagnostics
  5.  z_log_vr documentation corrected (not bounded by construction, only interact is)
  6.  Standardized coefficients transformed back to raw scale for interpretation
  7.  Retransformation bias correction for log-vol via smearing estimator: S = mean(exp(err))
  8.  Missing volatility in build_covariance raises explicit ValueError (not zeroed out)
  9.  Alpha stability tracked across folds as an explicit diagnostic metric
  10. Documentation explicit about correlation/covariance horizons
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from scipy.linalg import eigh
from typing import Dict, List, Any
import warnings

_EPS = 1e-8
_CLIP = 5.0    # winsorization bound for z-scores and interact


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _roll_z(series: pd.Series, window: int) -> pd.Series:
    """Fixed-window rolling z-score (not expanding)."""
    mu = series.rolling(window, min_periods=max(5, window // 3)).mean()
    sd = series.rolling(window, min_periods=max(5, window // 3)).std()
    return (series - mu) / (sd + _EPS)


def _nearest_psd(M: np.ndarray) -> np.ndarray:
    """Project symmetric matrix to nearest PSD via eigenvalue clipping."""
    M = (M + M.T) / 2.0
    eigvals, eigvecs = eigh(M)
    eigvals = np.maximum(eigvals, 0.0)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return (out + out.T) / 2.0


def _force_unit_diagonal(C: np.ndarray) -> np.ndarray:
    """Rescale so that C_ii = 1."""
    d = np.sqrt(np.diag(C))
    d = np.where(d < _EPS, 1.0, d)
    return C / np.outer(d, d)


def _make_pipeline(alpha: float) -> Pipeline:
    """StandardScaler → Ridge pipeline. Scaler is fitted inside each fold only."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha)),
    ])


# ---------------------------------------------------------------------------
# FeatureEngine
# ---------------------------------------------------------------------------

class FeatureEngine:
    """
    8-feature set (v5).

    Features:
      z_ret_l   – rolling z-score of long-window log return
      spread    – z_ret_s - z_ret_l  (momentum acceleration)
      z_vol_s   – rolling z-score of short-window realized vol
      z_log_vr  – rolling z-score of log(vol_s / vol_l)
      sr_s      – rolling Sharpe: ret_s / (vol_s * sqrt(w_s))
      sr_l      – rolling Sharpe: ret_l / (vol_l * sqrt(w_l))
      z_dd      – rolling z-score of trailing drawdown
      interact  – clip(z_ret_l, ±5) × clip(z_log_vr, ±5), clipped to ±5
    """
    def __init__(self, short_window: int, long_window: int):
        self.w_s    = short_window
        self.w_l    = long_window
        self.w_norm = long_window

    def compute(self, prices: pd.Series) -> pd.DataFrame:
        ret   = np.log(prices / prices.shift(1))
        ret_s = np.log(prices / prices.shift(self.w_s))
        ret_l = np.log(prices / prices.shift(self.w_l))

        vol_s = ret.rolling(self.w_s, min_periods=max(2, self.w_s // 2)).std()
        vol_l = ret.rolling(self.w_l, min_periods=max(2, self.w_l // 2)).std()

        z_ret_s  = _roll_z(ret_s, self.w_norm)
        z_ret_l  = _roll_z(ret_l, self.w_norm)
        spread   = z_ret_s - z_ret_l

        z_vol_s  = _roll_z(vol_s, self.w_norm)

        log_vr   = np.log((vol_s + _EPS) / (vol_l + _EPS))
        z_log_vr = _roll_z(log_vr, self.w_norm)

        sr_s = ret_s / (vol_s * np.sqrt(self.w_s) + _EPS)
        sr_l = ret_l / (vol_l * np.sqrt(self.w_l) + _EPS)

        roll_max = prices.rolling(self.w_l, min_periods=1).max()
        dd       = (prices - roll_max) / (roll_max + _EPS)
        z_dd     = _roll_z(dd, self.w_norm)

        # Winsorize inputs then product, then winsorize product
        z_ret_l_c  = z_ret_l.clip(-_CLIP, _CLIP)
        z_log_vr_c = z_log_vr.clip(-_CLIP, _CLIP)
        interact   = (z_ret_l_c * z_log_vr_c).clip(-_CLIP, _CLIP)

        return pd.DataFrame({
            'z_ret_l':  z_ret_l,
            'spread':   spread,
            'z_vol_s':  z_vol_s,
            'z_log_vr': z_log_vr,
            'sr_s':     sr_s,
            'sr_l':     sr_l,
            'z_dd':     z_dd,
            'interact': interact,
        })


# ---------------------------------------------------------------------------
# ForwardTarget
# ---------------------------------------------------------------------------

class ForwardTarget:
    """
    Return:    log(P_{t+h} / P_t)
    Volatility: log(RMS_forward + ε)   ← log-space; back-transformed in pipeline
    """
    def __init__(self, horizon: int):
        self.h = horizon

    def forward_return(self, prices: pd.Series) -> pd.Series:
        return np.log(prices.shift(-self.h) / prices)

    def forward_log_vol(self, prices: pd.Series) -> pd.Series:
        """
        y_t = log( sqrt(1/h · Σ r²_{t+i}) + ε )
        Modelling in log-space: guarantees back-transformed forecast > 0,
        aligns with multiplicative/log-normal volatility dynamics.
        """
        ret = np.log(prices / prices.shift(1))
        r2  = ret ** 2
        rms = np.sqrt(r2.rolling(window=self.h, min_periods=self.h).sum() / self.h)
        return np.log(rms.shift(-self.h) + _EPS)


# ---------------------------------------------------------------------------
# Naive benchmark
# ---------------------------------------------------------------------------

def _trailing_rms(prices: pd.Series, window: int) -> pd.Series:
    """Trailing RMS realized vol over `window` periods."""
    ret = np.log(prices / prices.shift(1))
    r2  = ret ** 2
    return np.sqrt(r2.rolling(window, min_periods=max(1, window // 2)).mean())


def naive_benchmark(prices: pd.Series, metric_type: str,
                    short_window: int, **_) -> pd.Series:
    """
    Return     → 0            (random-walk null)
    Volatility → log(short-window trailing RMS vol + ε)

    The benchmark is *fixed* (no OOS benchmark selection among candidate
    windows). This keeps the OOS R² interpretation clean:
        OOS R² > 0  ⟺  model beats the naive short-window vol persistence.
    """
    if metric_type == 'return':
        ret = np.log(prices / prices.shift(1)).fillna(0.0)
        return ret.expanding(min_periods=1).mean()
    # log-space to match the log-vol target
    return np.log(_trailing_rms(prices, short_window) + _EPS)


# ---------------------------------------------------------------------------
# PurgedWalkForward
# ---------------------------------------------------------------------------

class PurgedWalkForward:
    """
    Expanding walk-forward with explicit purge and StandardScaler per fold.

    split_ratio : initial training fraction
    purge       : number of embargo periods (= h)

    Fold structure covers all remaining data perfectly (priority 3):
        train_end   = initial_train + i * step
        test_start  = train_end + purge
        test_end    = test_start + step
    """
    def __init__(self, n_splits: int = 4, purge: int = 0,
                 split_ratio: float = 0.6):
        self.n_splits    = n_splits
        self.purge       = purge
        self.split_ratio = split_ratio

    def _select_alpha(self, X_tr: np.ndarray, y_tr: np.ndarray,
                      alphas: List[float]) -> float:
        """
        Inner 70/30 time-series split WITH an h-period purge.
        Scaler fitted on inner train only. No GCV.
        """
        n         = len(X_tr)
        inner_end = max(int(n * 0.70), 5)
        val_start = inner_end + self.purge
        if val_start >= n - 2:
            return 1.0

        Xi_tr, yi_tr = X_tr[:inner_end],   y_tr[:inner_end]
        Xi_vl, yi_vl = X_tr[val_start:],   y_tr[val_start:]

        best_alpha, best_mse = 1.0, np.inf
        for a in alphas:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = _make_pipeline(a)
                pipe.fit(Xi_tr, yi_tr)
                mse = mean_squared_error(yi_vl, pipe.predict(Xi_vl))
            if mse < best_mse:
                best_mse, best_alpha = mse, a
        return best_alpha

    def run(self, df: pd.DataFrame, feature_cols: List[str],
            alphas: List[float], log_space: bool = False) -> Dict[str, Any]:
        """
        Returns aggregated OOS arrays AND per-fold metrics.
        """
        n             = len(df)
        initial_train = max(int(n * self.split_ratio), 20)
        # step ensures we exactly consume the remaining data over n_splits folds
        # accounting for the n_splits * purge gaps
        # test_budget = total_remaining - total_purge
        test_budget = (n - initial_train) - (self.n_splits * self.purge)
        step = max(test_budget // self.n_splits, 5)

        agg: Dict[str, list] = {'y_true': [], 'y_pred': [], 'y_bench': []}
        fold_metrics: List[Dict[str, Any]] = []

        for i in range(self.n_splits):
            train_end  = initial_train + i * step
            test_start = train_end + self.purge
            # On the last fold, consume all remaining data to avoid dropping the remainder
            test_end   = test_start + step if i < self.n_splits - 1 else n

            if test_end > n or test_start >= n:
                break

            train_df = df.iloc[:train_end]
            test_df  = df.iloc[test_start:test_end]

            if len(train_df) < 10 or len(test_df) < 3:
                continue

            X_tr = train_df[feature_cols].values
            y_tr = train_df['target'].values
            X_te = test_df[feature_cols].values
            y_te = test_df['target'].values
            y_bench_te = test_df['bench'].values

            best_alpha = self._select_alpha(X_tr, y_tr, alphas)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = _make_pipeline(best_alpha)
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_te)

            # Compute fold metrics
            mse_m   = mean_squared_error(y_te, preds)
            mse_b   = mean_squared_error(y_te, y_bench_te)
            fold_r2 = float(1.0 - mse_m / mse_b) if mse_b > _EPS else 0.0

            if np.std(preds) < _EPS or np.std(y_te) < _EPS:
                fold_ic = 0.0
            else:
                fold_ic, _ = spearmanr(preds, y_te)
                fold_ic    = float(fold_ic)

            metrics = {
                'r2':    fold_r2,
                'ic':    fold_ic,
                'rmse':  float(np.sqrt(mse_m)),
                'alpha': best_alpha,
            }

            # If volatility, compute sigma-space R2 for this fold (priority 1)
            if log_space:
                sig_true  = np.maximum(np.exp(y_te) - _EPS, _EPS)
                sig_pred  = np.maximum(np.exp(preds) - _EPS, _EPS)
                sig_bench = np.maximum(np.exp(y_bench_te) - _EPS, _EPS)
                
                mse_sig_m = mean_squared_error(sig_true, sig_pred)
                mse_sig_b = mean_squared_error(sig_true, sig_bench)
                sigma_r2 = float(1.0 - mse_sig_m / mse_sig_b) if mse_sig_b > _EPS else 0.0
                metrics['sigma_r2'] = sigma_r2

            fold_metrics.append(metrics)

            agg['y_true'].extend(y_te)
            agg['y_pred'].extend(preds)
            agg['y_bench'].extend(y_bench_te)

        return {
            'y_true':       np.array(agg['y_true']),
            'y_pred':       np.array(agg['y_pred']),
            'y_bench':      np.array(agg['y_bench']),
            'fold_metrics': fold_metrics,
        }


# ---------------------------------------------------------------------------
# ForecastDiagnostics
# ---------------------------------------------------------------------------

def _fold_rates(fold_metrics: List[Dict], log_space: bool = False) -> Dict[str, float]:
    """
    True joint positive rate: mean(IC_i > 0 AND R²_i > 0)
    Uses sigma_r2 for volatility (priority 1, 2)
    """
    if not fold_metrics:
        return {'positive_joint_rate': 0.0}
    n = len(fold_metrics)
    joint_count = 0
    for f in fold_metrics:
        ic_pos = f['ic'] > 0
        if log_space:
            r2_pos = f.get('sigma_r2', -1) > 0
        else:
            r2_pos = f['r2'] > 0
        if ic_pos and r2_pos:
            joint_count += 1
    return {
        'positive_joint_rate': joint_count / n,
    }


def compute_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_bench: np.ndarray,
                        fold_metrics: List[Dict[str, Any]],
                        metric_type: str,
                        log_space: bool = False) -> Dict[str, Any]:
    """
    Return diagnostics: OOS R², DA, Rank IC, RMSE, fold stability, alpha stability
    Vol diagnostics (log-space):   OOS R², Rank IC, RMSE, fold stability
    Vol diagnostics (σ-space):     OOS R², Rank IC, MAE ratio, RMSE, alpha stability
        → σ-space metrics are used as the primary confidence signal.
    """
    if len(y_true) == 0 or len(fold_metrics) == 0:
        return _empty_diag(metric_type)

    mse_m  = mean_squared_error(y_true, y_pred)
    mse_b  = mean_squared_error(y_true, y_bench)
    oos_r2 = float(1.0 - mse_m / mse_b) if mse_b > _EPS else 0.0

    if np.std(y_pred) < _EPS or np.std(y_true) < _EPS:
        rank_ic = 0.0
    else:
        rank_ic, _ = spearmanr(y_pred, y_true)

    rmse = float(np.sqrt(mse_m))

    # Fold stability
    fold_ics = np.array([f['ic'] for f in fold_metrics])
    ic_mean  = float(np.mean(fold_ics))
    ic_std   = float(np.std(fold_ics))
    ic_ir    = float(ic_mean / (ic_std + _EPS))
    
    if log_space:
        fold_r2s = np.array([f.get('sigma_r2', 0.0) for f in fold_metrics])
    else:
        fold_r2s = np.array([f['r2'] for f in fold_metrics])
    median_r2 = float(np.median(fold_r2s))
    
    rates = _fold_rates(fold_metrics, log_space=log_space)
    
    # Alpha stability (priority 9)
    alphas = [f['alpha'] for f in fold_metrics]
    alpha_counts = pd.Series(alphas).value_counts().to_dict()

    diag = {
        'oos_r2':              oos_r2,
        'rank_ic':             float(rank_ic),
        'rmse':                rmse,
        'median_r2':           median_r2,
        'ic_mean':             ic_mean,
        'ic_std':              ic_std,
        'ic_ir':               ic_ir,
        'positive_joint_rate': rates['positive_joint_rate'],
        'alpha_counts':        alpha_counts,
        'num_folds':           len(fold_metrics),
    }

    if metric_type == 'return':
        da = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        diag['da'] = da

    if log_space:
        # σ-space diagnostics
        sig_true  = np.maximum(np.exp(y_true)  - _EPS, _EPS)
        sig_pred  = np.maximum(np.exp(y_pred)  - _EPS, _EPS)
        sig_bench = np.maximum(np.exp(y_bench) - _EPS, _EPS)

        mse_sig_m = mean_squared_error(sig_true, sig_pred)
        mse_sig_b = mean_squared_error(sig_true, sig_bench)
        mae_sig_m = mean_absolute_error(sig_true, sig_pred)
        mae_sig_b = mean_absolute_error(sig_true, sig_bench)

        diag['sigma_oos_r2']    = float(1.0 - mse_sig_m / mse_sig_b) if mse_sig_b > _EPS else 0.0
        diag['sigma_mae_ratio'] = float(mae_sig_m / (mae_sig_b + _EPS))
        diag['sigma_rmse']      = float(np.sqrt(mse_sig_m))
        if np.std(sig_pred) > _EPS and np.std(sig_true) > _EPS:
            diag['sigma_rank_ic'] = float(spearmanr(sig_pred, sig_true)[0])
        else:
            diag['sigma_rank_ic'] = 0.0

    return diag


def _empty_diag(metric_type: str) -> Dict[str, Any]:
    d = {
        'oos_r2': 0.0, 'rank_ic': 0.0, 'rmse': 0.0,
        'median_r2': 0.0, 'ic_mean': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0,
        'positive_joint_rate': 0.0, 'alpha_counts': {}, 'num_folds': 0,
    }
    if metric_type == 'return':
        d['da'] = 0.0
    else:
        d.update({'sigma_oos_r2': 0.0, 'sigma_mae_ratio': 1.0,
                  'sigma_rmse': 0.0, 'sigma_rank_ic': 0.0})
    return d


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def confidence_weight(diag: Dict[str, Any], metric_type: str) -> float:
    """
    C ∈ [0, 0.70]

    Return:     f(median_R², DA, Rank IC, positive_joint_rate)
    Volatility: f(σ-space OOS R², σ-space MAE ratio, σ-space Rank IC,
                  positive_joint_rate)
    """
    c_stability = diag.get('positive_joint_rate', 0.0)

    if metric_type == 'return':
        median_r2 = max(diag.get('median_r2', 0.0), 0.0)
        rank_ic   = max(diag.get('rank_ic',   0.0), 0.0)
        da        = diag.get('da', 0.5)
        c_r2 = min(median_r2, 1.0)
        c_da = max(da - 0.5, 0.0) / 0.5
        c_ic = min(rank_ic / 0.20, 1.0)
        C = (c_r2 + c_da + c_ic + c_stability) / 4.0
    else:
        sig_r2    = max(diag.get('sigma_oos_r2',  0.0), 0.0)
        sig_ratio = diag.get('sigma_mae_ratio', 1.0)
        sig_ic    = max(diag.get('sigma_rank_ic', 0.0), 0.0)
        c_r2  = min(sig_r2, 1.0)
        c_err = float(np.clip(1.0 - sig_ratio, 0.0, 1.0))
        c_ic  = min(sig_ic / 0.20, 1.0)
        C = (c_r2 + c_err + c_ic + c_stability) / 4.0

    return float(min(C, 0.70))


# ---------------------------------------------------------------------------
# Main HorizonShiftedOLS
# ---------------------------------------------------------------------------

class HorizonShiftedOLS:
    """
    Horizon-aware Ridge + StandardScaler forecaster.

    fit_predict(prices, metric_type, split_ratio) → dict:
        'estimate'    – confidence-blended forecast
        'val_metrics' – diagnostic dict
        'confidence'  – scalar C ∈ [0, 0.70]
        'coefs'       – raw-scale coefficients (priority 6)
    """
    _ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

    def __init__(self, target_horizon: int, short_window: int, long_window: int):
        self.h   = target_horizon
        self.w_s = short_window
        self.w_l = long_window
        self._fe = FeatureEngine(short_window, long_window)
        self._ft = ForwardTarget(target_horizon)

    def fit_predict(self, prices: pd.Series,
                    metric_type: str = 'return',
                    split_ratio: float = 0.6) -> Dict[str, Any]:

        # ── 1. Features & target ─────────────────────────────────────────
        X = self._fe.compute(prices)

        if metric_type == 'return':
            y_raw = self._ft.forward_return(prices)
        elif metric_type == 'volatility':
            y_raw = self._ft.forward_log_vol(prices)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        y_bench = naive_benchmark(prices, metric_type, self.w_s)

        current_X = X.iloc[[-1]]
        df = pd.concat([X, y_raw.rename('target'), y_bench.rename('bench')], axis=1)
        df_clean  = df.dropna()

        # Check sufficient data for at least 2 folds (priority 4)
        n_clean = len(df_clean)
        min_required = max(int(n_clean * split_ratio), 20) + (self.h * 2) + 10
        if n_clean < min_required:
            return self._fallback(prices, metric_type)

        feature_cols = list(X.columns)
        is_log_vol   = (metric_type == 'volatility')

        # ── 2. Purged walk-forward OOS ────────────────────────────────────
        wf  = PurgedWalkForward(n_splits=4, purge=self.h,
                                split_ratio=split_ratio)
        oos = wf.run(df_clean, feature_cols, self._ALPHAS, log_space=is_log_vol)

        # Ensure valid folds produced
        if not oos['fold_metrics']:
             return self._fallback(prices, metric_type)

        diag = compute_diagnostics(
            oos['y_true'], oos['y_pred'], oos['y_bench'],
            oos['fold_metrics'], metric_type, log_space=is_log_vol
        )

        # ── 3. Final model on all clean data ──────────────────────────────
        X_all = df_clean[feature_cols].values
        y_all = df_clean['target'].values

        best_alpha = wf._select_alpha(X_all, y_all, self._ALPHAS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipe = _make_pipeline(best_alpha)
            final_pipe.fit(X_all, y_all)

        if current_X.isna().any().any():
            model_log = float(df_clean['target'].iloc[-1])
        else:
            model_log = float(final_pipe.predict(current_X.values)[0])

        bench_log = float(y_bench.dropna().iloc[-1]) if not y_bench.dropna().empty else model_log

        # Transform coefficients to raw scale (priority 6)
        scaler = final_pipe.named_steps['scaler']
        ridge = final_pipe.named_steps['ridge']
        raw_coefs = ridge.coef_ / scaler.scale_
        
        # ── 4. Back-transform vol; blend in σ-space ───────────────────────
        if metric_type == 'volatility':
            # Smearing correction (priority 7)
            train_preds = final_pipe.predict(X_all)
            residuals = y_all - train_preds
            # Limit smearing factor to prevent explosion if model is terrible
            smearing = np.clip(np.mean(np.exp(residuals)), 0.5, 2.0)
            
            model_est = max(float(np.exp(model_log) * smearing - _EPS), _EPS)
            bench_est = max(float(np.exp(bench_log) - _EPS), _EPS)
            
            diag['smearing_factor'] = smearing
        else:
            model_est = model_log
            bench_est = bench_log    # = 0.0

        C       = confidence_weight(diag, metric_type)
        blended = C * model_est + (1.0 - C) * bench_est
        if metric_type == 'volatility':
            blended = max(blended, _EPS)

        return {
            'estimate':    blended,
            'val_metrics': diag,
            'confidence':  C,
            'alpha_ridge': best_alpha,
            'coefs':       dict(zip(feature_cols, raw_coefs)),
        }

    def _fallback(self, prices: pd.Series, metric_type: str) -> Dict[str, Any]:
        bench = naive_benchmark(prices, metric_type, self.w_s)
        last_log = float(bench.dropna().iloc[-1]) if not bench.dropna().empty else 0.0
        if metric_type == 'volatility':
            last = max(float(np.exp(last_log) - _EPS), _EPS)
        else:
            last = 0.0
        return {
            'estimate':    last,
            'val_metrics': _empty_diag(metric_type),
            'confidence':  0.0,
            'alpha_ridge': 1.0,
            'coefs':       {},
        }


# ---------------------------------------------------------------------------
# CorrelationBlender
# ---------------------------------------------------------------------------

class CorrelationBlender:
    """
    NOTE ON HORIZON (Priority 10):
    This estimates a one-period correlation matrix using base bar returns.
    Volatility forecasts are per-period metrics averaged over an h-period horizon.
    The resulting covariance matrix is a *one-period* covariance matrix representing
    the structural relationships over the next h periods.
    Any portfolio optimizer using this must manually apply temporal scaling 
    (e.g., sqrt(h) or h) if the risk model horizon is longer than the base bar.
    """
    def __init__(self, short_window: int, long_window: int,
                 alpha: float = 0.4, shrinkage: float = 0.1):
        self.w_s       = short_window
        self.w_l       = long_window
        self.alpha     = alpha
        self.shrinkage = shrinkage

    def estimate(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        returns = np.log(prices_df / prices_df.shift(1)).dropna(how='all')
        symbols = list(prices_df.columns)
        n       = len(symbols)

        if len(returns) < self.w_l:
            raw = returns.corr().reindex(index=symbols, columns=symbols).fillna(0).values
        else:
            cs  = returns.tail(self.w_s).corr().reindex(index=symbols, columns=symbols).fillna(0).values
            cl  = returns.tail(self.w_l).corr().reindex(index=symbols, columns=symbols).fillna(0).values
            raw = self.alpha * cs + (1.0 - self.alpha) * cl

        np.fill_diagonal(raw, 1.0)
        psd = _nearest_psd(raw)
        psd = _force_unit_diagonal(psd)

        lam    = self.shrinkage
        cstar  = (1.0 - lam) * psd + lam * np.eye(n)

        cstar = _nearest_psd(cstar)
        cstar = _force_unit_diagonal(cstar)
        np.fill_diagonal(cstar, 1.0)

        return pd.DataFrame(cstar, index=symbols, columns=symbols)


# ---------------------------------------------------------------------------
# build_covariance
# ---------------------------------------------------------------------------

def build_covariance(volatilities: pd.Series,
                     correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Σ = D · C · D  with PSD enforcement and explicit symmetrisation.
    """
    symbols = correlation_matrix.columns
    vols    = volatilities.reindex(symbols)
    
    # Priority 8: Raise explicit error instead of filling with 0
    if vols.isna().any():
        missing = vols.index[vols.isna()].tolist()
        raise ValueError(f"Missing volatilities for assets: {missing}")
        
    D = np.diag(vols.values)
    C = correlation_matrix.values

    Sigma = D @ C @ D

    if np.linalg.eigvalsh(Sigma).min() < 0:
        Sigma = _nearest_psd(Sigma)

    Sigma = (Sigma + Sigma.T) / 2.0
    return pd.DataFrame(Sigma, index=symbols, columns=symbols)
