# Whale Hunter v3 — Max Verstappen Patch

## What changed

### Backtest engine (major)
- **Metric cache precomputation** for Hurst / Entropy / LLE (reused across OFF/ON/sanity/ablation runs)
- **Automatic threshold calibration by quantiles**
  - `HURST_MIN` from configurable quantile (default p60)
  - `ENTROPY_MAX` from configurable quantile (default p80)
  - `LLE_MAX` from configurable quantile (default p80)
- **Signal-aware calibration mode** (calibrate only on bars with signal >= threshold)
- **Walk-forward mode** (optional calibration split + out-of-sample evaluation)
- **Forced close at end of backtest** (cleaner realized PnL accounting)
- **PnL report split**: realized / forced-close / open-estimate
- **Richer econ filter diagnostics**
  - block reasons by sub-filter (`hurst_fail`, `entropy_fail`, `lle_fail`, `multi_fail`)
  - indicator distribution stats (min/p10/p50/p90/max/mean)
- **JSON report export** (`BACKTEST_SAVE_REPORT`) for later analysis

### Robustness improvements
- **LLE fallback proxy** when Rosenstein estimate is unstable/NaN on synthetic series
  - prevents the filter from collapsing into `metric_nan` everywhere
- **NaN-safe live strategy gate** (fails closed if required metrics are non-finite)
- **Indicator sentinels normalized to NaN** in utils (less hidden behavior)

### Debug scripts
- Updated PowerShell and BAT helpers with v3 defaults and calibration flags

## New env flags (backtest)
- `BACKTEST_SIGNAL_THRESHOLD`
- `BACKTEST_AUTO_CALIBRATE`
- `BACKTEST_CALIB_HURST_Q`
- `BACKTEST_CALIB_ENTROPY_Q`
- `BACKTEST_CALIB_LLE_Q`
- `BACKTEST_CALIB_ON_SIGNAL_ONLY`
- `BACKTEST_CALIB_MIN_SAMPLES`
- `BACKTEST_PRECOMPUTE_METRICS`
- `BACKTEST_RUN_WALKFORWARD`
- `BACKTEST_WF_CALIB_FRAC`
- `BACKTEST_SAVE_REPORT`

## Quick run (PowerShell)
```powershell
.\run_backtest_debug.ps1 -N 20000 -Stride 5 -AutoCalibrate -Ablation -Sanity
```

## Where PnL is calculated
`backtest.py -> simulate_backtest()`
- **Normal exit**: inside the expiry exit block (`pnl = +/- stake - fee`)
- **Forced close EOD**: final section that closes remaining positions at the last price
