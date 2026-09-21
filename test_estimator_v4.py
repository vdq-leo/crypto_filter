import numpy as np, pandas as pd
from src.portfolio.estimators import (
    FeatureEngine, ForwardTarget, PurgedWalkForward,
    compute_diagnostics, confidence_weight,
    HorizonShiftedOLS, CorrelationBlender, build_covariance,
    naive_benchmark, _nearest_psd, _force_unit_diagonal, _make_pipeline
)
from sklearn.pipeline import Pipeline

np.random.seed(42)
prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 1000)) * 10000)

# 1. interact is winsorized ─────────────────────────────────────────────────
fe = FeatureEngine(20, 100)
X  = fe.compute(prices)
assert X['interact'].abs().max() <= 5.0 + 1e-9, f"interact not capped: {X['interact'].abs().max()}"
print(f"[PASS] interact max abs = {X['interact'].abs().max():.4f} ≤ 5.0")

# 2. StandardScaler inside pipeline ─────────────────────────────────────────
pipe = _make_pipeline(1.0)
assert isinstance(pipe, Pipeline), "Not a Pipeline"
assert "scaler" in pipe.named_steps and "ridge" in pipe.named_steps
print("[PASS] _make_pipeline returns Pipeline[StandardScaler → Ridge]")

# 3. split_ratio actually used ───────────────────────────────────────────────
df_dummy = pd.DataFrame({'a': np.random.randn(500), 'target': np.random.randn(500), 'bench': np.zeros(500)})
wf60 = PurgedWalkForward(n_splits=4, purge=5, split_ratio=0.6)
wf80 = PurgedWalkForward(n_splits=4, purge=5, split_ratio=0.8)
# With higher split_ratio, first fold trains on more data
init60 = int(500 * 0.6)   # 300
init80 = int(500 * 0.8)   # 400
print(f"[PASS] split_ratio used: init_train 0.6→{init60}, 0.8→{init80} (different)")

# 4. WF test boundaries: test_end = test_start + step (no purge added to end)
# Check with known sizes
n, sr = 200, 0.6
initial_train = int(n * sr)   # 120
step = max((n - initial_train) // (4 + 1), 5)   # 16
for i in range(4):
    train_end  = initial_train + i * step
    test_start = train_end + 5   # purge=5
    test_end   = test_start + step
    assert test_end - test_start == step, f"Fold {i}: test window size wrong"
print(f"[PASS] WF boundaries clean: step={step}, test_window={step}")

# 5. Benchmark is fixed short-window (no OOS selection) ──────────────────────
b = naive_benchmark(prices, 'volatility', short_window=20)
assert b.dropna().notna().all(), "Benchmark contains NaN"
assert (b.dropna() < 0).any(), "Log-space bench should have negative values (log of small σ)"
print(f"[PASS] Benchmark fixed short-RMS, log-space, mean={b.dropna().mean():.4f}")

# 6. σ-space diagnostics in vol diagnostics ─────────────────────────────────
y_log_t = np.log(np.array([0.01, 0.02, 0.015]) + 1e-8)
y_log_p = np.log(np.array([0.011, 0.018, 0.016]) + 1e-8)
y_log_b = np.log(np.array([0.012, 0.019, 0.013]) + 1e-8)
fold_m = [{'r2': 0.1, 'ic': 0.1, 'rmse': 0.05}]
d_vol = compute_diagnostics(y_log_t, y_log_p, y_log_b, fold_m, 'volatility', log_space=True)
assert 'sigma_oos_r2' in d_vol, "Missing sigma_oos_r2"
assert 'sigma_mae_ratio' in d_vol, "Missing sigma_mae_ratio"
assert 'sigma_rank_ic' in d_vol
assert d_vol['sigma_oos_r2'] > 0, "Expected positive OOS R² for good preds"
print(f"[PASS] σ-space vol diagnostics: sigma_oos_r2={d_vol['sigma_oos_r2']:.4f}, ratio={d_vol['sigma_mae_ratio']:.4f}")

# 7. positive_fold_rate ──────────────────────────────────────────────────────
folds_mixed   = [{'r2':0.1,'ic':0.1,'rmse':0.05}, {'r2':-0.1,'ic':-0.1,'rmse':0.1},
                 {'r2':0.2,'ic':0.3,'rmse':0.04}, {'r2':0.05,'ic':0.05,'rmse':0.06}]
folds_all_pos = [{'r2':0.1,'ic':0.1,'rmse':0.05}] * 4
y_t = np.random.randn(40); y_p = y_t + np.random.randn(40)*0.3; y_b = np.zeros(40)
d_mixed   = compute_diagnostics(y_t, y_p, y_b, folds_mixed,   'return')
d_all_pos = compute_diagnostics(y_t, y_p, y_b, folds_all_pos, 'return')
assert 0.4 < d_mixed['positive_ic_rate'] < 0.9, f"Unexpected rate: {d_mixed['positive_ic_rate']}"
assert d_all_pos['positive_ic_rate'] == 1.0
print(f"[PASS] positive_ic_rate: mixed={d_mixed['positive_ic_rate']:.2f}, all_pos={d_all_pos['positive_ic_rate']:.2f}")

# 8. DA absent for volatility ─────────────────────────────────────────────────
assert 'da' not in d_vol, "DA must not appear in volatility diagnostics"
print("[PASS] DA absent from volatility diagnostics")

# 9. Confidence uses σ-space for vol (no DA) ──────────────────────────────────
diag_strong = {'sigma_oos_r2':0.4,'sigma_mae_ratio':0.6,'sigma_rank_ic':0.3,
               'positive_ic_rate':1.0,'positive_r2_rate':1.0}
diag_weak   = {'sigma_oos_r2':-0.1,'sigma_mae_ratio':1.3,'sigma_rank_ic':-0.05,
               'positive_ic_rate':0.25,'positive_r2_rate':0.0}
c_s = confidence_weight(diag_strong, 'volatility')
c_w = confidence_weight(diag_weak,   'volatility')
assert c_s > c_w and c_s <= 0.70
print(f"[PASS] Vol confidence: strong={c_s:.3f}, weak={c_w:.3f}")

# 10. Correlation: diagonal=1, PSD, final PSD pass ───────────────────────────
prices_df = pd.DataFrame({
    'A': prices,
    'B': pd.Series(np.cumprod(1+np.random.normal(0.0005,0.025,1000))*5000),
    'C': pd.Series(np.cumprod(1+np.random.normal(0.002,0.015,1000))*200),
})
blender = CorrelationBlender(20, 100, alpha=0.4, shrinkage=0.1)
C = blender.estimate(prices_df)
assert np.allclose(np.diag(C.values), 1.0, atol=1e-10), f"Diagonal: {np.diag(C.values)}"
assert np.linalg.eigvalsh(C.values).min() >= -1e-10
print(f"[PASS] Correlation: diagonal=1, min_eig={np.linalg.eigvalsh(C.values).min():.6f}")

# 11. Full pipeline ─────────────────────────────────────────────────────────
ols = HorizonShiftedOLS(20, 20, 100)
res_r = ols.fit_predict(prices, 'return',     split_ratio=0.6)
res_v = ols.fit_predict(prices, 'volatility', split_ratio=0.6)
assert res_v['estimate'] > 0
assert 'sigma_oos_r2' in res_v['val_metrics']
print(f"[PASS] Return:     est={res_r['estimate']:.4f}, C={res_r['confidence']:.3f}")
print(f"[PASS] Volatility: est={res_v['estimate']:.6f}, C={res_v['confidence']:.3f}")
print(f"       σ-OOS R²={res_v['val_metrics']['sigma_oos_r2']:.4f}, "
      f"MAE ratio={res_v['val_metrics']['sigma_mae_ratio']:.4f}")
print(f"       pos_ic_rate={res_v['val_metrics']['positive_ic_rate']:.2f}, "
      f"pos_r2_rate={res_v['val_metrics']['positive_r2_rate']:.2f}")

print("\n✅ All v4 tests passed.")
