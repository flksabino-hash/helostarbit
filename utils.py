"""
utils.py — shared helpers + econophysics / ML utilities.

Notes:
- Econophysics features here are intentionally lightweight (numpy-only) for mobile friendliness.
- Transfer entropy and Lyapunov exponent are computationally heavier; keep lookbacks small.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import re
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

# Optional crypto-at-rest (like WhaleHunter): if missing, falls back to plaintext.
try:
    from cryptography.fernet import Fernet  # type: ignore
    HAS_FERNET = True
except Exception:
    HAS_FERNET = False

# Optional: requests retry session (handy for debugging / simple scripts)
try:
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False


_LOG = logging.getLogger(__name__)
_HURST_ERR_COUNT = 0


def _log_hurst_err(where: str, err: Exception) -> None:
    global _HURST_ERR_COUNT
    _HURST_ERR_COUNT += 1
    if _HURST_ERR_COUNT <= 5 or _HURST_ERR_COUNT % 100 == 0:
        _LOG.warning("HURST_ERR[%s] %r (count=%s)", where, err, _HURST_ERR_COUNT)


class SecretRedactionFilter(logging.Filter):
    """
    Redacts common secret formats and any exact values present in selected env vars.
    This is a *defense-in-depth* measure to reduce accidental leakage to console/log files.
    """
    _PATTERNS = [
        # 0x + 64 hex (private keys / tx hashes)
        re.compile(r"0x[a-fA-F0-9]{64}"),
        # UUIDs (API keys are often UUIDs)
        re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", re.IGNORECASE),
        # long base64-ish tokens
        re.compile(r"[A-Za-z0-9_\-]{32,}={0,2}"),
    ]

    _ENV_KEYS = (
        "POLY_PRIVATE_KEY",
        "POLY_API_KEY",
        "POLY_API_SECRET",
        "POLY_API_PASSPHRASE",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
    )

    def __init__(self) -> None:
        super().__init__()
        self._exact: list[str] = []
        for k in self._ENV_KEYS:
            v = os.getenv(k)
            if v and len(v) >= 6:
                self._exact.append(v)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        red = msg

        # exact-value redaction
        for v in self._exact:
            if v in red:
                red = red.replace(v, "***REDACTED***")

        # pattern redaction
        for pat in self._PATTERNS:
            red = pat.sub("***REDACTED***", red)

        if red != msg:
            record.msg = red
            record.args = ()
        return True


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Add redaction filter to every handler
    flt = SecretRedactionFilter()
    root = logging.getLogger()
    for h in root.handlers:
        h.addFilter(flt)



def utc_ts() -> int:
    return int(time.time())


def safe_json_dumps(obj: Any) -> str:
    # Match Polymarket clients: stable JSON, double quotes, no spaces
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def build_requests_session(total_retries: int = 4, backoff: float = 0.6) -> "requests.Session":
    """
    Retry/backoff session pattern (mirrors your WhaleHunter mobile files).
    """
    if not HAS_REQUESTS:
        raise RuntimeError("requests not installed")
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "PUT", "DELETE"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def quantize_down(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return float(x - (x % step))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def tick_round(price: float, tick: float, *, direction: str = 'nearest') -> float:
    """Round a price to the allowed tick. direction: 'down'|'up'|'nearest'."""
    price = float(price)
    tick = float(tick)
    if tick <= 0:
        return price
    n = price / tick
    if direction == 'down':
        return float(math.floor(n) * tick)
    if direction == 'up':
        return float(math.ceil(n) * tick)
    return float(round(n) * tick)


# ──────────────────────────────────────────────────────────────────────────────
# Secrets: encrypted JSON KV store (Fernet if available)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class EncryptedKV:
    path: Path
    master_key_env: str = "BOT_MASTER_KEY"  # base64 fernet key
    _fernet: Any = None

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_FERNET:
            key = os.getenv(self.master_key_env, "").strip()
            if key:
                self._fernet = Fernet(key.encode("utf-8"))

    @staticmethod
    def generate_master_key() -> str:
        if not HAS_FERNET:
            raise RuntimeError("cryptography not installed")
        return Fernet.generate_key().decode("utf-8")

    def _read_raw(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        data = self.path.read_bytes()
        if self._fernet:
            try:
                data = self._fernet.decrypt(data)
            except Exception:
                # wrong key or plaintext; best effort
                pass
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return {}

    def _write_raw(self, obj: Dict[str, Any]) -> None:
        data = safe_json_dumps(obj).encode("utf-8")
        if self._fernet:
            data = self._fernet.encrypt(data)
        self.path.write_bytes(data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._read_raw().get(key, default)

    def set(self, key: str, value: Any) -> None:
        obj = self._read_raw()
        obj[key] = value
        self._write_raw(obj)


# ──────────────────────────────────────────────────────────────────────────────
# Econophysics primitives
# ──────────────────────────────────────────────────────────────────────────────

def log_returns(prices: Sequence[float]) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    if p.size < 3:
        return np.array([], dtype=float)
    p = np.clip(p, 1e-12, np.inf)
    return np.diff(np.log(p))


def hurst_rs(series: Sequence[float], max_lag: int = 20) -> float:
    x = np.asarray(series, dtype=float)
    if x.size < max_lag + 5:
        return float(np.nan)
    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        d = x[lag:] - x[:-lag]
        tau.append(np.sqrt(np.std(d)))
    tau = np.asarray(tau, dtype=float)
    if tau.size == 0:
        return float(np.nan)
    mask = np.isfinite(tau) & (tau > 0)
    if mask.sum() < 3:
        return float(np.nan)
    try:
        slope = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]
        return float(2.0 * slope)
    except Exception as e:
        _log_hurst_err("hurst_rs", e)
        return float(np.nan)
def shannon_entropy_returns(prices: Sequence[float], bins: int = 30) -> float:
    """
    Shannon entropy of *returns* histogram.
    Lower entropy ~= more concentrated regime; higher entropy ~= noisier/chaotic.
    """
    r = log_returns(prices)
    if r.size < 50:
        return float(np.nan)
    hist, _ = np.histogram(r, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return float(np.nan)
    p = hist / np.sum(hist)
    return float(-np.sum(p * np.log(p)))


def lyapunov_rosenstein(series: Sequence[float], emb_dim: int = 6, tau: int = 1, max_t: int = 20, lag: int = None, max_iter: int = None) -> float:
    """
    Rosenstein-style Largest Lyapunov Exponent estimate (approx).
    Negative/near-zero suggests non-chaotic / stable; positive suggests divergence/chaos.

    For micro-horizons this is noisy — treat as a *filter*, not a predictor.
    """
    if lag is not None:
        tau = int(lag)
    if max_iter is not None:
        max_t = int(max_iter)
    x = np.asarray(series, dtype=float)
    if x.size < 200:
        return float(np.nan)
    # embed
    m = emb_dim
    n = x.size - (m - 1) * tau
    if n <= max_t + 5:
        return float(np.nan)
    Y = np.empty((n, m), dtype=float)
    for i in range(m):
        Y[:, i] = x[i * tau:i * tau + n]
    # nearest neighbor per point (exclude temporal neighbors)
    dists = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    # Theiler window
    theiler = 10
    for i in range(n):
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        dists[i, lo:hi] = np.inf
    nn = np.argmin(dists, axis=1)
    # divergence curve
    div = []
    for t in range(1, max_t + 1):
        valid = (np.arange(n - t) >= 0)
        i_idx = np.arange(n - t)[valid]
        j_idx = nn[i_idx]
        j_ok = j_idx + t < n
        i_idx = i_idx[j_ok]
        j_idx = j_idx[j_ok]
        if i_idx.size < 50:
            continue
        dist_t = np.linalg.norm(Y[i_idx + t] - Y[j_idx + t], axis=1)
        dist_0 = np.linalg.norm(Y[i_idx] - Y[j_idx], axis=1)
        ratio = dist_t / np.clip(dist_0, 1e-12, np.inf)
        div.append(np.mean(np.log(np.clip(ratio, 1e-12, np.inf))))
    if len(div) < 5:
        return float(np.nan)
    # slope vs time
    t = np.arange(1, len(div) + 1)
    slope, _ = np.polyfit(t, np.asarray(div), 1)
    return float(slope)


def _discretize(x: np.ndarray, bins: int) -> np.ndarray:
    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    # Avoid zero-width bins
    qs = np.unique(qs)
    if qs.size <= 2:
        return np.zeros_like(x, dtype=int)
    return np.digitize(x, qs[1:-1], right=True)


def transfer_entropy(x: Sequence[float], y: Sequence[float], k: int = 1, bins: int = 6) -> float:
    """
    Discrete plug-in estimator of transfer entropy TE(X→Y) with k-lag Markov.
    Lightweight but biased; use comparatively (higher/lower) not absolutely.

    TE(X→Y) = sum p(y_t, y_{t-1}, x_{t-1}) log p(y_t|y_{t-1},x_{t-1}) / p(y_t|y_{t-1})
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    if n < 400:
        return 0.0
    x = x[-n:]
    y = y[-n:]
    xd = _discretize(x, bins=bins)
    yd = _discretize(y, bins=bins)
    # Build counts
    from collections import Counter
    c_xyz = Counter()
    c_yz = Counter()
    c_y = Counter()
    # k=1 only (keep it fast)
    for t in range(1, n):
        yt = int(yd[t])
        y1 = int(yd[t-1])
        x1 = int(xd[t-1])
        c_xyz[(yt, y1, x1)] += 1
        c_yz[(y1, x1)] += 1
        c_y[(y1,)] += 1
    te = 0.0
    total = float(n - 1)
    for (yt, y1, x1), cnt in c_xyz.items():
        p_yt_y1_x1 = cnt / total
        p_yt_given_y1_x1 = cnt / c_yz[(y1, x1)]
        # p(yt|y1) estimated by marginalizing x1
        denom = sum(v for (yt2, y1b, _), v in c_xyz.items() if yt2 == yt and y1b == y1)
        p_yt_given_y1 = denom / sum(v for (y1b,), v in c_y.items() if y1b == y1)
        if p_yt_given_y1 > 0 and p_yt_given_y1_x1 > 0:
            te += p_yt_y1_x1 * math.log(p_yt_given_y1_x1 / p_yt_given_y1)
    return float(max(0.0, te))


def mfdfa_proxy(prices: Sequence[float], window: int = 200) -> float:
    """
    Proxy for volatility clustering via DFA scaling exponent (q=2 only).
    Returns ~0.5 random walk; >0.5 persistent; <0.5 anti.
    """
    r = log_returns(prices)
    if r.size < window:
        return 0.5
    x = np.cumsum(r - np.mean(r))
    # Scales
    scales = np.array([8, 16, 32, 64], dtype=int)
    F = []
    for s in scales:
        if x.size < s * 4:
            continue
        nseg = x.size // s
        rms = []
        for v in range(nseg):
            seg = x[v*s:(v+1)*s]
            t = np.arange(s)
            coeff = np.polyfit(t, seg, 1)
            trend = coeff[0]*t + coeff[1]
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        F.append(np.mean(rms))
    if len(F) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(scales[:len(F)]), np.log(np.asarray(F)), 1)
    return float(clamp(slope, 0.0, 1.5))


# ──────────────────────────────────────────────────────────────────────────────
# Heavy tail risk tools (power-laws, Levy-stable)
# ──────────────────────────────────────────────────────────────────────────────

def hill_tail_index(returns: Sequence[float], k: int = 50) -> float:
    """
    Hill estimator for tail index alpha on |returns| (heavy tail: smaller alpha).
    """
    r = np.abs(np.asarray(returns, dtype=float))
    r = r[np.isfinite(r)]
    if r.size < max(200, k + 5):
        return 2.0
    r = np.sort(r)[::-1]
    k = int(min(k, r.size - 2))
    xk = r[k]
    if xk <= 0:
        return 2.0
    alpha_inv = np.mean(np.log(r[:k] / xk))
    if alpha_inv <= 0 or not np.isfinite(alpha_inv):
        return 2.0
    return float(1.0 / alpha_inv)


def sample_alpha_stable(alpha: float, beta: float, size: int, scale: float = 1.0, loc: float = 0.0) -> np.ndarray:
    """
    Chambers–Mallows–Stuck sampler for α-stable distributions.
    alpha in (0,2], beta in [-1,1].
    """
    alpha = float(clamp(alpha, 0.1, 2.0))
    beta = float(clamp(beta, -1.0, 1.0))
    U = np.random.uniform(-math.pi/2, math.pi/2, size=size)
    W = np.random.exponential(1.0, size=size)
    if abs(alpha - 1.0) > 1e-8:
        b = math.atan(beta * math.tan(math.pi*alpha/2)) / alpha
        S = (1 + (beta * math.tan(math.pi*alpha/2))**2)**(1/(2*alpha))
        X = S * (np.sin(alpha*(U + b)) / (np.cos(U)**(1/alpha))) * ((np.cos(U - alpha*(U + b)) / W)**((1-alpha)/alpha))
    else:
        # alpha == 1
        X = (2/math.pi) * ((math.pi/2 + beta*U) * np.tan(U) - beta * np.log((math.pi/2*W*np.cos(U)) / (math.pi/2 + beta*U)))
    return loc + scale * X


# ──────────────────────────────────────────────────────────────────────────────
# Simple symbolic regression (enumerative) for micro-forecast drift
# ──────────────────────────────────────────────────────────────────────────────

def symbolic_drift(prices: Sequence[float], horizon_sec: int = 120, max_terms: int = 4) -> float:
    """
    Very small symbolic-regression proxy: fit a handful of basis functions to log-price over time
    and output implied drift per second at horizon.
    Uses numpy least squares; returns drift in log-price units / second.
    """
    p = np.asarray(prices, dtype=float)
    if p.size < 30:
        return 0.0
    y = np.log(np.clip(p, 1e-12, np.inf))
    t = np.arange(y.size, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)

    # Candidate bases (keep tiny for speed/stability)
    bases = [
        [np.ones_like(t), t],                                # linear
        [np.ones_like(t), t, t**2],                           # quadratic
        [np.ones_like(t), t, np.tanh(t)],                     # tanh
        [np.ones_like(t), t, np.sin(t)],                      # sin
        [np.ones_like(t), t, np.sign(t) * np.sqrt(np.abs(t))] # sqrt
    ]
    best = None
    best_mse = float("inf")
    for X_cols in bases:
        X = np.vstack(X_cols).T
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ coef
            mse = float(np.mean((y - pred)**2))
            if mse < best_mse:
                best_mse = mse
                best = (coef, X_cols)
        except Exception:
            continue

    if best is None:
        return 0.0

    coef, X_cols = best
    # Derivative dy/dt in normalized t units:
    # Only t-dependent columns contribute; approximate at last t
    t0 = t[-1]
    # compute dX/dt for columns
    dcols = []
    for col in X_cols:
        if np.allclose(col, 1.0):
            dcols.append(0.0)
        else:
            # heuristic derivative: fit local slope via gradient
            dcols.append(float(np.gradient(col)[-1]))
    dy_dt_norm = float(np.dot(coef, np.asarray(dcols)))
    # Convert normalized t to index units: dt_norm = 1/std(t_raw)
    std_raw = (np.arange(y.size).std() + 1e-12)
    dy_dt_idx = dy_dt_norm / std_raw
    # Assume 1 index ~= 1 tick. Caller should supply appropriately sampled prices.
    return float(dy_dt_idx)


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(returns: Sequence[float], eps: float = 1e-12) -> float:
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r) + eps)
    return mu / sd * math.sqrt(r.size)


def bootstrap_sharpe(returns: Sequence[float], n: int = 500, seed: int = 7) -> Tuple[float, Tuple[float, float]]:
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    sh = []
    for _ in range(n):
        samp = rng.choice(r, size=r.size, replace=True)
        sh.append(sharpe_ratio(samp))
    sh = np.asarray(sh, dtype=float)
    return float(np.mean(sh)), (float(np.quantile(sh, 0.05)), float(np.quantile(sh, 0.95)))
