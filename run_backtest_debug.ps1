param(
  [int]$N = 20000,
  [int]$Stride = 5,
  [switch]$DebugGates = $true,
  [switch]$Sanity = $true,
  [switch]$Ablation = $true,
  [switch]$AutoCalibrate = $true,
  [switch]$WalkForward = $false,
  [switch]$ForceCloseEod = $true,
  [string]$ReportPath = ".\\artifacts\\backtest_v3_report.json"
)

$env:BACKTEST_SYNTH_N = "$N"
$env:BACKTEST_METRIC_STRIDE = "$Stride"
$env:BACKTEST_DEBUG_GATES = $(if ($DebugGates) { "1" } else { "0" })
$env:BACKTEST_RUN_SANITY = $(if ($Sanity) { "1" } else { "0" })
$env:BACKTEST_RUN_ABLATION = $(if ($Ablation) { "1" } else { "0" })
$env:BACKTEST_AUTO_CALIBRATE = $(if ($AutoCalibrate) { "1" } else { "0" })
$env:BACKTEST_RUN_WALKFORWARD = $(if ($WalkForward) { "1" } else { "0" })
$env:BACKTEST_FORCE_CLOSE_EOD = $(if ($ForceCloseEod) { "1" } else { "0" })
$env:BACKTEST_PRECOMPUTE_METRICS = "1"
$env:BACKTEST_CALIB_MIN_SAMPLES = "200"
$env:BACKTEST_CALIB_HURST_Q = "0.60"
$env:BACKTEST_CALIB_ENTROPY_Q = "0.80"
$env:BACKTEST_CALIB_LLE_Q = "0.80"
$env:BACKTEST_SAVE_REPORT = "$ReportPath"
$env:BACKTEST_DEBUG_INDICATOR_STATS = "1"

python .\backtest.py
