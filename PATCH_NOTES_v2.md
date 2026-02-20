# whale_hunter_v0_2 — backtest evolution patch

## What changed
- **Optional SciPy fallback**: synthetic backtest now runs even without `scipy` (uses Student-t heavy-tail fallback).
- **Econo filter sub-reasons**: prints which gate blocks entries (`hurst_fail`, `entropy_fail`, `lle_fail`, `te_fail`, `multi_fail`).
- **Indicator distributions**: quantile stats for Hurst/Entropy/LLE to calibrate thresholds against actual data.
- **PnL breakdown**: realized vs forced-close vs open proxy PnL printed explicitly.
- **Performance improvement**: backtest no longer recomputes cumulative PnL with `sum()` every tick (O(n²)-ish behavior removed).
- **PowerShell helpers**: `run_backtest_debug.ps1` and `run_backtest_debug.bat` for Windows-friendly execution.
- **Ablation suite (optional)**: quickly tests H-only / E-only / L-only / H+E+L using env `BACKTEST_RUN_ABLATION=1`.

## Useful commands (PowerShell)
```powershell
./run_backtest_debug.ps1
# or with custom params:
./run_backtest_debug.ps1 -N 50000 -Stride 10 -Ablation:$true -ForceCloseEod:$true
```

## Where PnL is computed
- `backtest.py` → `_settle_open_positions(...)` computes trade PnL per closed position.
- `simulate_backtest(...)` accumulates `realized_pnl`, updates `bal`, and prints the PnL breakdown.
