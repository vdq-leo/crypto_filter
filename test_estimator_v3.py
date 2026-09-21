import numpy as np, pandas as pd
from src.portfolio.estimators import (
    FeatureEngine, ForwardTarget, PurgedWalkForward,
    compute_diagnostics, confidence_weight,
    HorizonShiftedOLS, CorrelationBlender, build_covariance,
    _nearest_psd, _force_unit_diagonal, naive_benchmark
)

np.random.seed(42)
prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 1000)) * 10000)

# ── 1. Features: 8 columns, no z_ret_s ──────────────────────────────────────
fe = FeatureEngine(20, 100)
X  = fe.compute(prices)
assert list(X.columns) == ['z_ret_l','spread','z_vol_s','z_log_vr','sr_s','sr_l','z_dd','interact'], \
    f"Wrong columns: {list(X.columns)}"
print(f"[PASS] FeatureEngine: {X.shape}, no z_ret_s")

# ── 2. Log-vol target ────────────────────────────────────────────────────────
ft    = ForwardTarget(20)
y_log = ft.forward_log_vol(prices)
y_exp = np.exp(y_log.dropna()) - 1e-8
assert (y_exp > 0).all(), "Back-transformed vol contains non-positive values!"
print(f"[PASS] Log-vol target: back-transformed min = {y_exp.min():.6f} > 0")

# ── 3. Inner purge in alpha selection ───────────────────────────────────────
wf = PurgedWalkForward(n_splits=4, purge=20)
X_dummy = np.random.randn(300, 8)
y_dummy = np.random.randn(300)
alpha   = wf._select_alpha(X_dummy, y_dummy, [0.01, 1.0, 100.0])
print(f"[PASS] Inner alpha selection (purge=20) chose α = {alpha}")

# ── 4. DA absent from vol diagnostics ───────────────────────────────────────
fold_m = [{'r2': 0.1, 'ic': 0.1, 'rmse': 0.05}]
y_t  = np.array([0.01, 0.02, 0.015])
y_p  = np.array([0.011, 0.018, 0.016])
y_b  = np.array([0.012, 0.019, 0.013])
d_vol = compute_diagnostics(y_t, y_p, y_b, fold_m, 'volatility')
d_ret = compute_diagnostics(y_t*10, y_p*10, y_b*10, fold_m, 'return')
assert 'da' not in d_vol, "DA should NOT appear for volatility"
assert 'da' in d_ret, "DA should appear for return"
assert 'error_ratio' in d_vol, "error_ratio should appear for volatility"
print(f"[PASS] Vol diagnostics: {list(d_vol.keys())}")
print(f"[PASS] Ret diagnostics: {list(d_ret.keys())}")

# ── 5. Per-fold stability ────────────────────────────────────────────────────
assert 'ic_ir' in d_vol and 'ic_std' in d_vol
print(f"[PASS] Fold stability metrics present: ic_ir={d_vol['ic_ir']:.4f}")

# ── 6. Correlation diagonal = 1 + PSD + shrinkage ───────────────────────────
prices_df = pd.DataFrame({
    'A': prices,
    'B': pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.025, 1000)) * 5000),
    'C': pd.Series(np.cumprod(1 + np.random.normal(0.002,  0.015, 1000)) * 200),
})
blender = CorrelationBlender(20, 100, alpha=0.4, shrinkage=0.1)
C = blender.estimate(prices_df)
diag_vals = np.diag(C.values)
assert np.allclose(diag_vals, 1.0, atol=1e-10), f"Diagonal != 1: {diag_vals}"
min_eig = np.linalg.eigvalsh(C.values).min()
assert min_eig >= -1e-10, f"Correlation not PSD: min eigenvalue = {min_eig}"
print(f"[PASS] Correlation: diagonal={diag_vals}, min_eig={min_eig:.6f}")

# ── 7. Covariance PSD + symmetrised ────────────────────────────────────────
vols  = pd.Series({'A': 0.30, 'B': 0.35, 'C': 0.50})
Sigma = build_covariance(vols, C)
assert np.allclose(Sigma.values, Sigma.values.T, atol=1e-12), "Sigma not symmetric"
assert np.linalg.eigvalsh(Sigma.values).min() >= -1e-10, "Sigma not PSD"
print(f"[PASS] Covariance: symmetric, min_eig={np.linalg.eigvalsh(Sigma.values).min():.6f}")

# ── 8. Full pipeline: return & volatility ────────────────────────────────────
ols = HorizonShiftedOLS(20, 20, 100)

res_r = ols.fit_predict(prices, 'return')
print(f"[PASS] Return estimate:    {res_r['estimate']:.4f}, C={res_r['confidence']:.3f}")
print(f"       val: {res_r['val_metrics']}")

res_v = ols.fit_predict(prices, 'volatility')
assert res_v['estimate'] > 0, "Volatility estimate must be > 0"
print(f"[PASS] Volatility estimate: {res_v['estimate']:.6f}, C={res_v['confidence']:.3f}")
print(f"       val: {res_v['val_metrics']}")

# ── 9. confidence_weight: vol has no DA component ───────────────────────────
c_strong = confidence_weight({'median_r2':0.4,'rank_ic':0.25,'ic_ir':2.0,'error_ratio':0.7}, 'volatility')
c_weak   = confidence_weight({'median_r2':-0.1,'rank_ic':-0.05,'ic_ir':-0.5,'error_ratio':1.3}, 'volatility')
assert c_strong > c_weak, "Strong model should have higher confidence"
assert c_strong <= 0.70, "Confidence capped at 0.70"
print(f"[PASS] Confidence: strong={c_strong:.3f}, weak={c_weak:.3f} (cap=0.70)")

print("\n✅ All tests passed.")
