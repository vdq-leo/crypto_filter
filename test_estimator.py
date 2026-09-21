import numpy as np
import pandas as pd
from src.portfolio.estimators import (
    FeatureEngine, ForwardTarget, PurgedWalkForward,
    compute_diagnostics, confidence_weight,
    HorizonShiftedOLS, CorrelationBlender, build_covariance
)

# Synthetic price series — 1000 daily prices
np.random.seed(42)
prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 1000)) * 10000)

print("=== FeatureEngine ===")
fe = FeatureEngine(20, 100)
X = fe.compute(prices)
print(f"Shape: {X.shape}, Columns: {list(X.columns)}")
print(f"NaN rows: {X.isna().any(axis=1).sum()} (expected: first ~100 rows)")

print("\n=== ForwardTarget ===")
ft = ForwardTarget(20)
y_ret = ft.forward_return(prices)
y_vol = ft.forward_vol_rms(prices)
print(f"Return target: {y_ret.dropna().describe()['mean']:.4f} mean (should be ~0)")
print(f"Volatility target: {y_vol.dropna().describe()['mean']:.4f} mean (should be ~0.02*sqrt(20)≈0.089)")

print("\n=== HorizonShiftedOLS (return) ===")
ols = HorizonShiftedOLS(20, 20, 100)
res_ret = ols.fit_predict(prices, 'return')
print(f"Estimate:   {res_ret['estimate']:.4f}")
print(f"Confidence: {res_ret['confidence']:.3f}")
print(f"OOS R²:     {res_ret['val_metrics']['oos_r2']:.4f}")
print(f"DA:         {res_ret['val_metrics']['da']:.4f}")
print(f"Rank IC:    {res_ret['val_metrics']['rank_ic']:.4f}")

print("\n=== HorizonShiftedOLS (volatility) ===")
res_vol = ols.fit_predict(prices, 'volatility')
print(f"Estimate (daily):   {res_vol['estimate']:.6f}")
print(f"Confidence:         {res_vol['confidence']:.3f}")
print(f"OOS R²:             {res_vol['val_metrics']['oos_r2']:.4f}")

print("\n=== CorrelationBlender (with shrinkage) ===")
prices_df = pd.DataFrame({
    'A': prices,
    'B': pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.025, 1000)) * 5000),
    'C': pd.Series(np.cumprod(1 + np.random.normal(0.002, 0.015, 1000)) * 200),
})
blender = CorrelationBlender(20, 100, alpha=0.4, shrinkage=0.1)
C = blender.estimate(prices_df)
print("Correlation matrix:\n", C.round(3))
print("Diagonal (should be 1.0):", np.diag(C.values).round(4))

print("\n=== build_covariance (PSD check) ===")
vols = pd.Series({'A': 0.30, 'B': 0.35, 'C': 0.50})
Sigma = build_covariance(vols, C)
eigenvalues = np.linalg.eigvalsh(Sigma.values)
print(f"Min eigenvalue: {eigenvalues.min():.6f} (should be ≥ 0)")
print("Covariance matrix:\n", Sigma.round(4))

print("\n=== confidence_weight edge cases ===")
print("Strong: ", confidence_weight({'oos_r2': 0.5, 'da': 0.65, 'rank_ic': 0.3}))
print("Weak:   ", confidence_weight({'oos_r2': -0.2, 'da': 0.48, 'rank_ic': -0.05}))
print("Neutral:", confidence_weight({'oos_r2': 0.0, 'da': 0.50, 'rank_ic': 0.0}))
