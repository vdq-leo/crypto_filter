import numpy as np
import pandas as pd
from src.portfolio.optimizers import PortfolioOptimizer, VolatilityTargeter

# Dummy inputs
mu = pd.Series({'A': 0.05, 'B': 0.1, 'C': 0.15})
cov = pd.DataFrame(
    [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]],
    index=['A', 'B', 'C'], columns=['A', 'B', 'C']
)

# Initialize Optimizer
opt = PortfolioOptimizer(mu, cov, periods_per_year=1.0) # no scaling for simple test

# 1. Sharpe
res_sharpe = opt.max_sharpe()
assert res_sharpe['success'], res_sharpe['message']
print(f"[PASS] max_sharpe: {res_sharpe['weights'].to_dict()}")

# 2. Zero vol protection in Sharpe
cov_zero = cov.copy()
cov_zero.loc['A', 'A'] = 0.0
opt_zero = PortfolioOptimizer(mu, cov_zero, periods_per_year=1.0)
res_sharpe_zero = opt_zero.max_sharpe()
# Asset A has zero vol, it should be excluded or Sharpe should succeed without it. 
# Our implementation forces its bounds to (0,0) and successfully optimizes the rest.
assert res_sharpe_zero['success'], "Optimization should succeed by excluding A"
assert res_sharpe_zero['weights']['A'] == 0.0, "Zero vol asset A should have 0 weight"
print("[PASS] max_sharpe with zero-vol asset handles correctly")

# 3. VolatilityTargeter scaling up and down
targeter_up = VolatilityTargeter(0.30)
targeter_down = VolatilityTargeter(0.05)
rel_w = res_sharpe['weights']
# cov is already per-period, but targeter expects annualized in the context, so we pass scaled
ann_cov = cov * 1.0

res_up = targeter_up.scale_weights(rel_w, ann_cov)
assert res_up['gross_leverage'] > 1.0, f"Expected leverage > 1, got {res_up['gross_leverage']}"
assert res_up['cash_weight'] < 0.0, "Expected negative cash"
print(f"[PASS] Targeter Up (Leverage): {res_up['gross_leverage']:.2f}x, cash: {res_up['cash_weight']:.2f}")

res_down = targeter_down.scale_weights(rel_w, ann_cov)
assert res_down['gross_leverage'] < 1.0, "Expected leverage < 1"
assert res_down['cash_weight'] > 0.0, "Expected positive cash"
print(f"[PASS] Targeter Down (Cash): {res_down['gross_leverage']:.2f}x, cash: {res_down['cash_weight']:.2f}")

# 4. Hierarchical (Riskfolio wrapper)
res_herc = opt.hierarchical_optimization(model='HERC')
assert res_herc['success'], res_herc['message']
print("[PASS] HERC optimization works with custom_cov")

# 5. generate_random_portfolios Rejection sampling
rand_ports = opt.generate_random_portfolios(100)
assert len(rand_ports['returns']) == 100
print("[PASS] generate_random_portfolios works with rejection sampling")

print("\n✅ Optimizer tests passed.")
