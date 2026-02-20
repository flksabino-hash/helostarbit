from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from scipy.stats import levy_stable  # optional, faster/better synthetic sampling
except Exception:  # pragma: no cover
    levy_stable = None

from config import load_settings, Settings
from utils import hurst_rs, shannon_entropy_returns, lyapunov_rosenstein, sample_alpha_stable


def qstats(x):
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _pct(p: float) -> float:
    # Accept 0.8 or 80 interchangeably.
    p = float(p)
    if p > 1.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))


def _settings_dict(s: Settings) -> Dict[str, Any]:
    d = asdict(s)
    # tuples render fine, but cast for JSON/report consistency
    if isinstance(d.get("BINANCE_SYMBOLS"), tuple):
        d["BINANCE_SYMBOLS"] = list(d["BINANCE_SYMBOLS"])
    return d


def _warmup_meta(settings: Settings, n: int) -> Dict[str, int]:
    signal_window = max(int(settings.MAX_TTE_SEC), 30)
    hurst_window = max(64, min(600, n))
    entropy_window = max(64, min(600, n))
    lle_window = max(128, min(800, n))
    expiry_horizon = max(int(settings.MIN_TTE_SEC), 60)
    max_lookback = max(signal_window, hurst_window, entropy_window, lle_window)
    return {
        "signal_window": int(signal_window),
        "hurst_window": int(hurst_window),
        "entropy_window": int(entropy_window),
        "lle_window": int(lle_window),
        "expiry_horizon": int(expiry_horizon),
        "max_lookback": int(max_lookback),
        "ticks_after_warmup": int(max(0, n - max_lookback - 1)),
    }


def _gen_synthetic_returns(n: int, alpha: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if levy_stable is not None:
        try:
            r = levy_stable.rvs(alpha, 0.0, size=n, scale=0.001, random_state=rng)
            return np.asarray(r, dtype=float)
        except Exception:
            pass
    return sample_alpha_stable(int(n), alpha=alpha, scale=0.001, seed=seed)




def _compute_lle_metric(window: np.ndarray) -> float:
    """Best-effort LLE metric with deterministic fallback proxy.

    The Rosenstein estimate can return NaN/inf on short/noisy binary-like price paths.
    Fallback proxy uses lag-1 return persistence (log |phi|), which preserves the
    intended ordering: more negative -> more stable/mean-reverting, near-zero/positive -> noisier/chaotic.
    """
    try:
        v = float(lyapunov_rosenstein(window, m=2, tau=1))
        if np.isfinite(v):
            return v
    except Exception:
        pass

    px = np.asarray(window, dtype=float)
    if px.size < 10:
        return np.nan
    r = np.diff(np.log(np.clip(px, 1e-9, None)))
    r = r[np.isfinite(r)]
    if r.size < 8:
        return np.nan
    x = r[:-1] - np.mean(r[:-1])
    y = r[1:] - np.mean(r[1:])
    denom = float(np.dot(x, x))
    if denom <= 1e-18:
        return np.nan
    phi = float(np.dot(x, y) / denom)
    # clamp to avoid log blowups; abs(phi) > 1 implies explosive local dynamics
    return float(np.log(max(min(abs(phi), 5.0), 1e-8)))

def build_metric_cache(prices: np.ndarray, settings: Settings) -> Dict[str, Any]:
    """Precompute expensive econophysics metrics once and reuse across runs.

    This is the biggest performance win in v3 because OFF/ON/sanity/ablation/walk-forward
    otherwise recompute Hurst/Entropy/LLE repeatedly on the same series.
    """
    n = int(len(prices))
    w = _warmup_meta(settings, n)
    stride = max(1, int(getattr(settings, "BACKTEST_METRIC_STRIDE", 1)))

    hurst_vals = np.full(n, np.nan, dtype=float)
    entropy_vals = np.full(n, np.nan, dtype=float)
    lle_vals = np.full(n, np.nan, dtype=float)

    last_h, last_e, last_l = np.nan, np.nan, np.nan
    for i in range(w["max_lookback"], n):
        do_compute = (i == w["max_lookback"]) or (((i - w["max_lookback"]) % stride) == 0)
        if do_compute:
            h_win = prices[i - w["hurst_window"] : i]
            e_win = prices[i - w["entropy_window"] : i]
            l_win = prices[i - w["lle_window"] : i]
            try:
                last_h = float(hurst_rs(h_win))
            except Exception:
                last_h = np.nan
            try:
                last_e = float(shannon_entropy_returns(e_win))
            except Exception:
                last_e = np.nan
            try:
                last_l = _compute_lle_metric(l_win)
            except Exception:
                last_l = np.nan
        hurst_vals[i] = last_h
        entropy_vals[i] = last_e
        lle_vals[i] = last_l

    return {
        "hurst": hurst_vals,
        "entropy": entropy_vals,
        "lle": lle_vals,
        "warmup": w,
        "metric_stride": stride,
    }


def _signal_scores(prices: np.ndarray, signal_window: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    for i in range(signal_window, n):
        window = prices[i - signal_window : i]
        vol = float(np.std(window))
        if vol <= 0:
            out[i] = 0.0
            continue
        z = (prices[i] - float(np.mean(window))) / vol
        out[i] = float(abs(z))
    return out


def derive_auto_calibration(
    prices: np.ndarray,
    settings: Settings,
    cache: Optional[Dict[str, Any]],
    *,
    label: str = "global",
) -> Dict[str, Any]:
    """Calibrate econ thresholds by quantiles from observed indicator distributions.

    Working theory: hard-coded thresholds are often incompatible with the synthetic generator.
    Quantile calibration turns them into instrument settings rather than scripture.
    """
    if cache is None:
        cache = build_metric_cache(prices, settings)

    w = cache["warmup"]
    n = len(prices)
    idx = np.arange(n)
    mask = idx >= w["max_lookback"]

    signal_thr = float(getattr(settings, "BACKTEST_SIGNAL_THRESHOLD", 0.5))
    if bool(getattr(settings, "BACKTEST_CALIB_ON_SIGNAL_ONLY", True)):
        scores = _signal_scores(prices, w["signal_window"])
        mask = mask & np.isfinite(scores) & (scores >= signal_thr)
    else:
        scores = None

    h_arr = np.asarray(cache["hurst"], dtype=float)
    e_arr = np.asarray(cache["entropy"], dtype=float)
    l_arr = np.asarray(cache["lle"], dtype=float)

    h = h_arr[mask & np.isfinite(h_arr)]
    e = e_arr[mask & np.isfinite(e_arr)]
    l = l_arr[mask & np.isfinite(l_arr)]

    report: Dict[str, Any] = {
        "label": label,
        "enabled": bool(getattr(settings, "BACKTEST_AUTO_CALIBRATE", False)),
        "signal_only": bool(getattr(settings, "BACKTEST_CALIB_ON_SIGNAL_ONLY", True)),
        "signal_threshold": signal_thr,
        "sample_counts": {"hurst": int(h.size), "entropy": int(e.size), "lle": int(l.size)},
        "source_stats": {"hurst": qstats(h), "entropy": qstats(e), "lle": qstats(l)},
        "old_thresholds": {
            "HURST_MIN": float(settings.HURST_MIN),
            "ENTROPY_MAX": float(settings.ENTROPY_MAX),
            "LLE_MAX": float(settings.LLE_MAX),
        },
        "applied": False,
        "new_thresholds": None,
        "reason": None,
    }

    min_samples = int(getattr(settings, "BACKTEST_CALIB_MIN_SAMPLES", 300))
    if min(h.size, e.size, l.size) < min_samples:
        report["reason"] = f"insufficient_samples(min={min_samples})"
        return report

    q_h = _pct(getattr(settings, "BACKTEST_CALIB_HURST_Q", 0.60))
    q_e = _pct(getattr(settings, "BACKTEST_CALIB_ENTROPY_Q", 0.80))
    q_l = _pct(getattr(settings, "BACKTEST_CALIB_LLE_Q", 0.80))

    new_h = float(np.percentile(h, q_h * 100.0))
    new_e = float(np.percentile(e, q_e * 100.0))
    new_l = float(np.percentile(l, q_l * 100.0))

    report["quantiles"] = {"hurst": q_h, "entropy": q_e, "lle": q_l}
    report["new_thresholds"] = {
        "HURST_MIN": new_h,
        "ENTROPY_MAX": new_e,
        "LLE_MAX": new_l,
    }
    report["applied"] = True
    return report


def _apply_calibration(settings: Settings, report: Optional[Dict[str, Any]]) -> Settings:
    if not report or not report.get("applied"):
        return settings
    nt = report["new_thresholds"]
    return replace(
        settings,
        HURST_MIN=float(nt["HURST_MIN"]),
        ENTROPY_MAX=float(nt["ENTROPY_MAX"]),
        LLE_MAX=float(nt["LLE_MAX"]),
    )


def simulate_backtest(
    prices: np.ndarray,
    settings: Settings,
    use_filters: bool,
    *,
    sanity_mode: bool = False,
    metric_cache: Optional[Dict[str, Any]] = None,
    label: str = "",
) -> Dict[str, Any]:
    n = len(prices)
    warm = metric_cache["warmup"] if metric_cache is not None else _warmup_meta(settings, n)
    signal_window = warm["signal_window"]
    expiry_horizon = warm["expiry_horizon"]
    max_lookback = warm["max_lookback"]

    metric_stride = max(1, int(getattr(settings, "BACKTEST_METRIC_STRIDE", 1)))
    signal_thr = float(getattr(settings, "BACKTEST_SIGNAL_THRESHOLD", 0.5))
    force_close_eod = bool(getattr(settings, "BACKTEST_FORCE_CLOSE_EOD", False))

    print(
        "[BT] warmup check:",
        {
            "N_sintetico": int(n),
            "signal_window": int(signal_window),
            "hurst_window": int(warm["hurst_window"]),
            "entropy_window": int(warm["entropy_window"]),
            "lle_window": int(warm["lle_window"]),
            "expiry_horizon": int(expiry_horizon),
            "max_lookback": int(max_lookback),
            "ticks_after_warmup": int(max(0, n - max_lookback - 1)),
            "metric_stride": int(metric_stride),
            "filters": "ON" if use_filters else "OFF",
            "sanity_mode": bool(sanity_mode),
            "label": label or None,
            "signal_threshold": signal_thr,
        },
    )

    debug_counts = {
        "warmup": 0,
        "regime_block": 0,
        "signal_below_threshold": 0,
        "econ_filter_block": 0,
        "position_open_no_reentry": 0,
        "entry_ok": 0,
        "exit_ok": 0,
        "metric_nan": 0,
        "insufficient_balance": 0,
        "forced_exit_eod": 0,
    }
    econ_block_reasons = {
        "hurst_fail": 0,
        "entropy_fail": 0,
        "lle_fail": 0,
        "te_fail": 0,
        "mfdfa_fail": 0,
        "multi_fail": 0,
    }

    indicator_samples = {
        "hurst": [],
        "entropy": [],
        "lle": [],
        "mfd": [],
    }
    sample_max = int(getattr(settings, "BACKTEST_DEBUG_SAMPLE_MAX", 50000))

    balance = float(settings.PAPER_START_USDC)
    entry_count = 0
    exit_count = 0
    closed_trades = []
    realized_pnl_usdc = 0.0
    forced_close_pnl_usdc = 0.0
    positions = []

    # If no cache was passed (e.g., custom CSV one-off), compute metrics on demand and respect stride.
    last_hurst = np.nan
    last_entropy = np.nan
    last_lle = np.nan
    cache_h = cache_e = cache_l = None
    if metric_cache is not None:
        cache_h = np.asarray(metric_cache["hurst"], dtype=float)
        cache_e = np.asarray(metric_cache["entropy"], dtype=float)
        cache_l = np.asarray(metric_cache["lle"], dtype=float)

    for i in range(n):
        if i < max_lookback:
            debug_counts["warmup"] += 1
            continue

        # 1) Signal gate
        window = prices[i - signal_window : i]
        vol = float(np.std(window))
        if vol <= 0:
            score = 0.0
        else:
            z = (prices[i] - float(np.mean(window))) / vol
            score = float(abs(z))
        if score < signal_thr and not sanity_mode:
            debug_counts["signal_below_threshold"] += 1
            continue

        # 2) Econo metrics (from cache if available)
        if cache_h is not None:
            hurst = float(cache_h[i])
            entropy = float(cache_e[i])
            lle = float(cache_l[i])
        else:
            do_compute = (i == max_lookback) or (((i - max_lookback) % metric_stride) == 0)
            if do_compute:
                h_win = prices[i - warm["hurst_window"] : i]
                e_win = prices[i - warm["entropy_window"] : i]
                l_win = prices[i - warm["lle_window"] : i]
                try:
                    last_hurst = float(hurst_rs(h_win))
                except Exception:
                    last_hurst = np.nan
                try:
                    last_entropy = float(shannon_entropy_returns(e_win))
                except Exception:
                    last_entropy = np.nan
                try:
                    last_lle = _compute_lle_metric(l_win)
                except Exception:
                    last_lle = np.nan
            hurst, entropy, lle = float(last_hurst), float(last_entropy), float(last_lle)

        rets = np.diff(np.log(np.clip(prices[max(0, i - 60): i + 1], 1e-9, None)))
        mfd = float(np.std(rets) / (np.mean(np.abs(rets)) + 1e-9)) if len(rets) else 1.0

        if np.isfinite(hurst) and len(indicator_samples["hurst"]) < sample_max:
            indicator_samples["hurst"].append(float(hurst))
        if np.isfinite(entropy) and len(indicator_samples["entropy"]) < sample_max:
            indicator_samples["entropy"].append(float(entropy))
        if np.isfinite(lle) and len(indicator_samples["lle"]) < sample_max:
            indicator_samples["lle"].append(float(lle))
        if np.isfinite(mfd) and len(indicator_samples["mfd"]) < sample_max:
            indicator_samples["mfd"].append(float(mfd))

        # 3) Econo filter gate with subreason breakdown
        if use_filters and not sanity_mode:
            if not (np.isfinite(hurst) and np.isfinite(entropy) and np.isfinite(lle)):
                debug_counts["metric_nan"] += 1
                continue

            fails = []
            if settings.USE_HURST and hurst < float(settings.HURST_MIN):
                fails.append("hurst_fail")
            if settings.USE_ENTROPY and entropy > float(settings.ENTROPY_MAX):
                fails.append("entropy_fail")
            if settings.USE_LLE and lle > float(settings.LLE_MAX):
                fails.append("lle_fail")
            # TE/MFDFA are not strict gates in this backtest path yet; left in breakdown for future parity.

            if fails:
                debug_counts["econ_filter_block"] += 1
                if len(fails) > 1:
                    econ_block_reasons["multi_fail"] += 1
                else:
                    econ_block_reasons[fails[0]] += 1
                continue

        # 4) Single-position-at-a-time gate
        if positions:
            debug_counts["position_open_no_reentry"] += 1
        else:
            stake = min(balance * float(settings.RISK_PER_TRADE), float(settings.MAX_USDC_PER_TRADE))
            if stake <= 0:
                debug_counts["insufficient_balance"] += 1
            else:
                positions.append(
                    {
                        "entry_px": float(prices[i]),
                        "entry_i": int(i),
                        "stake": float(stake),
                        "expiry_i": int(min(n - 1, i + expiry_horizon)),
                    }
                )
                entry_count += 1
                debug_counts["entry_ok"] += 1

        # 5) Exits for any open positions
        j = 0
        while j < len(positions):
            pos = positions[j]
            if i >= pos["expiry_i"]:
                pnl = (1.0 if prices[i] > pos["entry_px"] else -1.0) * pos["stake"]
                fee = float(settings.TAKER_FEE_BPS) / 10000.0 * pos["stake"]
                pnl -= fee
                balance += pnl
                realized_pnl_usdc += pnl
                exit_count += 1
                debug_counts["exit_ok"] += 1
                closed_trades.append(
                    {
                        "entry_i": pos["entry_i"],
                        "exit_i": int(i),
                        "entry_px": pos["entry_px"],
                        "exit_px": float(prices[i]),
                        "stake": pos["stake"],
                        "pnl": float(pnl),
                        "fee": float(fee),
                        "forced": False,
                    }
                )
                positions.pop(j)
            else:
                j += 1

    # Optional forced mark/close at end of series for cleaner accounting reports.
    if force_close_eod and positions:
        last_i = n - 1
        last_px = float(prices[-1])
        for pos in list(positions):
            pnl = (1.0 if last_px > pos["entry_px"] else -1.0) * pos["stake"]
            fee = float(settings.TAKER_FEE_BPS) / 10000.0 * pos["stake"]
            pnl -= fee
            balance += pnl
            realized_pnl_usdc += pnl
            forced_close_pnl_usdc += pnl
            exit_count += 1
            debug_counts["exit_ok"] += 1
            debug_counts["forced_exit_eod"] += 1
            closed_trades.append(
                {
                    "entry_i": pos["entry_i"],
                    "exit_i": int(last_i),
                    "entry_px": pos["entry_px"],
                    "exit_px": last_px,
                    "stake": pos["stake"],
                    "pnl": float(pnl),
                    "fee": float(fee),
                    "forced": True,
                }
            )
        positions.clear()

    # Mark-to-market open PnL if not forcing close (or if anything somehow remains).
    open_pnl_est = 0.0
    if positions:
        mark_px = float(prices[-1])
        for pos in positions:
            fee = float(settings.TAKER_FEE_BPS) / 10000.0 * pos["stake"]
            open_pnl_est += ((1.0 if mark_px > pos["entry_px"] else -1.0) * pos["stake"]) - fee

    pnls = np.array([t["pnl"] for t in closed_trades], dtype=float) if closed_trades else np.array([], dtype=float)
    sharpe = 0.0
    if pnls.size > 1 and np.std(pnls) > 1e-12:
        sharpe = float((np.mean(pnls) / np.std(pnls)) * math.sqrt(252.0))

    if bool(getattr(settings, "BACKTEST_DEBUG_GATES", True)):
        print("DEBUG GATES:", debug_counts)
        print(
            "entries:",
            entry_count,
            "exits:",
            exit_count,
            "closed_trades:",
            len(closed_trades),
            "open_positions_end:",
            len(positions),
        )
        if use_filters and not sanity_mode:
            print("ECON BLOCK REASONS:", econ_block_reasons)
            blocked = max(1, int(debug_counts["econ_filter_block"]))
            print(
                "ECON BLOCK RATES:",
                {k: round(v / blocked, 4) for k, v in econ_block_reasons.items() if v > 0},
            )
        if bool(getattr(settings, "BACKTEST_DEBUG_INDICATOR_STATS", True)):
            print(
                "HURST stats:",
                qstats(indicator_samples["hurst"]),
                "threshold min =",
                float(settings.HURST_MIN),
            )
            print(
                "ENTROPY stats:",
                qstats(indicator_samples["entropy"]),
                "threshold max =",
                float(settings.ENTROPY_MAX),
            )
            print(
                "LLE stats:",
                qstats(indicator_samples["lle"]),
                "threshold max =",
                float(settings.LLE_MAX),
            )
            print("MFD proxy stats:", qstats(indicator_samples["mfd"]))

    return {
        "label": label,
        "closed_trades": closed_trades,
        "num_trades": int(len(closed_trades)),
        "entries": int(entry_count),
        "exits": int(exit_count),
        "open_positions_end": int(len(positions)),
        "open_pnl_est_usdc": float(open_pnl_est),
        "realized_pnl_usdc": float(realized_pnl_usdc),
        "forced_close_pnl_usdc": float(forced_close_pnl_usdc),
        "balance": float(balance),
        "sharpe": float(sharpe),
        "debug_counts": debug_counts,
        "econ_block_reasons": econ_block_reasons,
        "indicator_stats": {
            "hurst": qstats(indicator_samples["hurst"]),
            "entropy": qstats(indicator_samples["entropy"]),
            "lle": qstats(indicator_samples["lle"]),
            "mfd": qstats(indicator_samples["mfd"]),
        },
        "warmup": warm,
        "signal_threshold": signal_thr,
    }


def _print_run_summary(title: str, res: Dict[str, Any]) -> None:
    print(f"Trades ({title}):", res["num_trades"])
    print(
        "  entries/exits/closed/open:",
        res["entries"],
        res["exits"],
        res["num_trades"],
        res["open_positions_end"],
    )
    print(
        "  PnL realized/forced/open_est:",
        round(float(res["realized_pnl_usdc"]), 4),
        round(float(res["forced_close_pnl_usdc"]), 4),
        round(float(res["open_pnl_est_usdc"]), 4),
    )
    print("  Sharpe:", round(float(res["sharpe"]), 4), "Final balance:", round(float(res["balance"]), 4))


def _run_ablation(prices: np.ndarray, settings: Settings, cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    print("\nAblation suite (single-filter attribution)")
    runs = {
        "HURST_ONLY": replace(settings, USE_HURST=True, USE_ENTROPY=False, USE_LLE=False),
        "ENTROPY_ONLY": replace(settings, USE_HURST=False, USE_ENTROPY=True, USE_LLE=False),
        "LLE_ONLY": replace(settings, USE_HURST=False, USE_ENTROPY=False, USE_LLE=True),
        "H_E_L": replace(settings, USE_HURST=True, USE_ENTROPY=True, USE_LLE=True),
    }
    out = {}
    for name, s2 in runs.items():
        res = simulate_backtest(prices, s2, use_filters=True, metric_cache=cache, label=name)
        _print_run_summary(name, res)
        out[name] = {
            "trades": res["num_trades"],
            "balance": res["balance"],
            "sharpe": res["sharpe"],
            "econ_block_reasons": res["econ_block_reasons"],
            "thresholds": {"HURST_MIN": s2.HURST_MIN, "ENTROPY_MAX": s2.ENTROPY_MAX, "LLE_MAX": s2.LLE_MAX},
        }
    return out


def _run_walkforward(prices: np.ndarray, settings: Settings) -> Dict[str, Any]:
    n = len(prices)
    wf_frac = float(getattr(settings, "BACKTEST_WF_CALIB_FRAC", 0.40))
    wf_frac = min(0.9, max(0.1, wf_frac))
    split = int(n * wf_frac)
    warm = _warmup_meta(settings, n)
    # pad so eval segment still has enough room for warmup after split
    min_tail = warm["max_lookback"] + 200
    if n - split < min_tail:
        split = max(warm["max_lookback"] + 200, n - min_tail)
    if split <= warm["max_lookback"] + 50 or (n - split) <= warm["max_lookback"] + 50:
        return {"skipped": True, "reason": "series_too_short_for_walkforward"}

    cal_prices = prices[:split]
    eval_prices = prices[split:]
    print("\nWalk-forward calibration/eval")
    print({"cal_n": len(cal_prices), "eval_n": len(eval_prices), "split": split, "frac": wf_frac})

    t0 = time.perf_counter()
    cal_cache = build_metric_cache(cal_prices, settings)
    cal_ms = (time.perf_counter() - t0) * 1000.0
    report = derive_auto_calibration(cal_prices, settings, cal_cache, label="walkforward_cal")
    cal_settings = _apply_calibration(settings, report)
    if report.get("applied"):
        print("WF calibration thresholds:", report["new_thresholds"])  # evidence, not vibes
    else:
        print("WF calibration skipped:", report.get("reason"))

    t1 = time.perf_counter()
    eval_cache = build_metric_cache(eval_prices, cal_settings)
    eval_ms = (time.perf_counter() - t1) * 1000.0

    off = simulate_backtest(eval_prices, settings, use_filters=False, metric_cache=eval_cache, label="WF_OFF")
    on = simulate_backtest(eval_prices, cal_settings, use_filters=True, metric_cache=eval_cache, label="WF_ON_CALIB")
    _print_run_summary("WF_OFF", off)
    _print_run_summary("WF_ON_CALIB", on)

    return {
        "skipped": False,
        "split": split,
        "calibration": report,
        "cache_ms": {"cal": round(cal_ms, 3), "eval": round(eval_ms, 3)},
        "eval_off": {"trades": off["num_trades"], "balance": off["balance"], "sharpe": off["sharpe"]},
        "eval_on": {"trades": on["num_trades"], "balance": on["balance"], "sharpe": on["sharpe"]},
    }


def main():
    settings = load_settings()
    print("SETTINGS:", _settings_dict(settings))

    if settings.BACKTEST_CSV:
        arr = np.loadtxt(settings.BACKTEST_CSV, delimiter=",", dtype=float)
        if arr.ndim == 1:
            prices = arr
        else:
            prices = arr[:, -1]
        prices = np.asarray(prices, dtype=float)
        print(f"Backtest on CSV: {settings.BACKTEST_CSV} rows={len(prices)}")
    else:
        print("Synthetic Backtest (heavy-tailed alpha-stable returns)")
        n = int(settings.BACKTEST_SYNTH_N)
        rng = np.random.default_rng(int(settings.BACKTEST_SEED))
        base = 0.5 + np.cumsum(_gen_synthetic_returns(n, alpha=max(1.05, float(settings.LEVY_ALPHA_MIN)), seed=int(settings.BACKTEST_SEED)))
        prices = np.clip(base, 0.01, 0.99)
        # tiny microstructure jitter to avoid degenerate flat windows in some seeds
        prices = np.clip(prices + rng.normal(0.0, 1e-4, size=len(prices)), 0.01, 0.99)

    # Precompute metrics once for speed when enabled
    metric_cache = None
    cache_ms = None
    if bool(getattr(settings, "BACKTEST_PRECOMPUTE_METRICS", True)):
        t0 = time.perf_counter()
        metric_cache = build_metric_cache(prices, settings)
        cache_ms = (time.perf_counter() - t0) * 1000.0
        print(
            "[BT] metric cache built:",
            {
                "ms": round(cache_ms, 2),
                "metric_stride": metric_cache["metric_stride"],
                "warmup": metric_cache["warmup"],
            },
        )

    # Optional automatic quantile calibration for econ filters
    calib_report = None
    filter_on_settings = settings
    if bool(getattr(settings, "BACKTEST_AUTO_CALIBRATE", False)):
        calib_report = derive_auto_calibration(prices, settings, metric_cache, label="main")
        if calib_report.get("applied"):
            filter_on_settings = _apply_calibration(settings, calib_report)
            print("[BT] AUTO CALIBRATION APPLIED:", calib_report["new_thresholds"])
        else:
            print("[BT] AUTO CALIBRATION SKIPPED:", calib_report.get("reason"))
        print("[BT] calibration sample counts:", calib_report.get("sample_counts"))

    report_bundle: Dict[str, Any] = {
        "settings": _settings_dict(settings),
        "meta": {"price_rows": int(len(prices)), "metric_cache_ms": round(cache_ms, 3) if cache_ms else None},
        "calibration": calib_report,
        "runs": {},
    }

    # 1) Base strategy without filters (pipeline sanity)
    res_off = simulate_backtest(prices, settings, use_filters=False, metric_cache=metric_cache, label="OFF")
    _print_run_summary("filters OFF", res_off)
    report_bundle["runs"]["filters_off"] = {
        "trades": res_off["num_trades"],
        "entries": res_off["entries"],
        "exits": res_off["exits"],
        "open": res_off["open_positions_end"],
        "balance": res_off["balance"],
        "sharpe": res_off["sharpe"],
        "debug_counts": res_off["debug_counts"],
        "indicator_stats": res_off["indicator_stats"],
    }

    # 2) Filters ON (optionally auto-calibrated thresholds)
    res_on = simulate_backtest(prices, filter_on_settings, use_filters=True, metric_cache=metric_cache, label="ON")
    _print_run_summary("filters ON", res_on)
    report_bundle["runs"]["filters_on"] = {
        "trades": res_on["num_trades"],
        "entries": res_on["entries"],
        "exits": res_on["exits"],
        "open": res_on["open_positions_end"],
        "balance": res_on["balance"],
        "sharpe": res_on["sharpe"],
        "debug_counts": res_on["debug_counts"],
        "econ_block_reasons": res_on["econ_block_reasons"],
        "indicator_stats": res_on["indicator_stats"],
        "effective_thresholds": {
            "HURST_MIN": float(filter_on_settings.HURST_MIN),
            "ENTROPY_MAX": float(filter_on_settings.ENTROPY_MAX),
            "LLE_MAX": float(filter_on_settings.LLE_MAX),
        },
    }

    # 3) Hard sanity mode: disable all gates except core mechanics
    if bool(getattr(settings, "BACKTEST_RUN_SANITY", True)):
        res_sanity = simulate_backtest(prices, settings, use_filters=False, sanity_mode=True, metric_cache=metric_cache, label="SANITY")
        _print_run_summary("SANITY / no-gates", res_sanity)
        report_bundle["runs"]["sanity"] = {
            "trades": res_sanity["num_trades"],
            "balance": res_sanity["balance"],
            "sharpe": res_sanity["sharpe"],
            "debug_counts": res_sanity["debug_counts"],
        }

    # 4) Ablation suite: identify which econ filter is the serial killer
    if bool(getattr(settings, "BACKTEST_RUN_ABLATION", False)):
        report_bundle["ablation"] = _run_ablation(prices, filter_on_settings, metric_cache)

    # 5) Optional walk-forward calibration/evaluation split
    if bool(getattr(settings, "BACKTEST_RUN_WALKFORWARD", False)):
        report_bundle["walkforward"] = _run_walkforward(prices, settings)

    save_path = str(getattr(settings, "BACKTEST_SAVE_REPORT", "") or "").strip()
    if save_path:
        p = Path(save_path)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report_bundle, indent=2, ensure_ascii=False))
        print("[BT] report saved:", str(p))


if __name__ == "__main__":
    main()
