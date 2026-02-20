"""
strategy.py — edge calculation + econophysics/ML confluence + AMH adaptation.

Trade idea (short-horizon crypto markets):
- Scan Polymarket Gamma for markets expiring in 5–15 minutes.
- Use Binance spot to estimate probability of resolution (digital option proxy).
- Buy YES when:
    edge >= MIN_EDGE (post cost) AND
    econophysics confluence indicates persistence/stability:
        Hurst > HURST_MIN
        Entropy < ENTROPY_MAX
        LLE <= LLE_MAX
- Symbolic regression provides a micro-drift estimate used to slightly tilt probability.
- AMH adapts required edge multiplier based on recent performance.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import Settings
from exchange import GammaMarket, PolymarketCLOBPublic, OrderBook
from utils import (
    hurst_rs,
    shannon_entropy_returns,
    lyapunov_rosenstein,
    mfdfa_proxy,
    log_returns,
    symbolic_drift,
    transfer_entropy,
    clamp,
)

log = logging.getLogger("strategy")


def _norm_cdf(x: float) -> float:
    # Normal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(slots=True)
class Features:
    p_est: float
    mid_price: float
    edge: float
    hurst: float = 0.5
    entropy: float = 9.9
    lle: float = 0.0
    mfdfa: float = 0.5
    contagion: float = 0.0
    drift: float = 0.0


@dataclass(slots=True)
class Signal:
    market_id: str
    token_id: str
    side: str  # BUY only in this baseline
    limit_price: float
    risk_mult: float
    features: Features
    meta: Dict[str, str] = field(default_factory=dict)


class Strategy:
    def __init__(self, settings: Settings):
        self.s = settings
        self._pnl_hist: List[float] = []  # AMH window of pnl% per trade

    def edge_multiplier(self) -> float:
        """
        Adaptive Market Hypothesis (AMH) knob:
        If the last window underperforms, demand larger edge.
        """
        if not self.s.AMH_ENABLED or not self._pnl_hist:
            return 1.0
        w = self._pnl_hist[-self.s.AMH_WINDOW:]
        mu = float(np.mean(w))
        # If losing, scale up required edge, capped
        if mu < 0:
            mult = 1.0 + min(0.6, abs(mu) * 5.0)
        else:
            mult = 1.0 - min(0.2, mu * 2.0)
        return float(clamp(mult, self.s.AMH_EDGE_MULT_MIN, self.s.AMH_EDGE_MULT_MAX))

    def on_trade_closed(self, pnl_pct: float) -> None:
        self._pnl_hist.append(float(pnl_pct))
        if len(self._pnl_hist) > 500:
            self._pnl_hist = self._pnl_hist[-300:]

    async def evaluate(
        self,
        market: GammaMarket,
        clob: PolymarketCLOBPublic,
        *,
        now_ts: int,
        prices_by_symbol: Dict[str, float],
        history_by_symbol: Dict[str, Sequence[float]],
        other_histories: Dict[str, Sequence[float]] | None = None,
    ) -> Optional[Signal]:
        """
        Returns a BUY signal if market passes all gates.
        """
        if not market.enable_order_book:
            return None
        yes_token = market.yes_token_id
        if not yes_token:
            return None

        tte = market.end_ts - int(now_ts)
        if tte < self.s.MIN_TTE_SEC or tte > self.s.MAX_TTE_SEC:
            return None

        parsed = _parse_crypto_strike(market.question)
        if not parsed:
            return None
        sym, direction, strike = parsed
        spot = float(prices_by_symbol.get(sym, 0.0))
        if spot <= 0:
            return None

        hist = list(history_by_symbol.get(sym, []))
        if len(hist) < 200:
            return None

        mid = await clob.get_midpoint(yes_token)
        if mid is None:
            # fallback to gamma outcomePrices (less precise)
            mid = market.yes_price
        if mid is None or not (0.001 < mid < 0.999):
            return None

        # --- estimate p(resolution) ---
        p_est, drift = self._estimate_probability(
            spot=spot,
            strike=strike,
            direction=direction,
            tte_sec=tte,
            price_series=hist,
        )

        # --- costs in probability units (conservative) ---
        # fee is applied on notional; treat as bps of position value.
        fee_prob = self.s.TAKER_FEE_BPS / 10_000.0
        gas_prob = self.s.GAS_USDC_ESTIMATE / max(self.s.MAX_USDC_PER_TRADE, 1e-9)
        edge = (p_est - mid) - (fee_prob + gas_prob)

        # AMH scales required edge
        req_edge = self.s.MIN_EDGE * self.edge_multiplier()
        if edge < req_edge:
            return None

        feats = Features(p_est=p_est, mid_price=float(mid), edge=float(edge), drift=float(drift))

        # --- econophysics confluence ---
        risk_mult = 1.0
        if self.s.USE_ECONO:
            if self.s.USE_HURST:
                feats.hurst = hurst_rs(hist[-600:], max_lag=20)
                if (not np.isfinite(feats.hurst)) or (feats.hurst < self.s.HURST_MIN):
                    return None
            if self.s.USE_ENTROPY:
                feats.entropy = shannon_entropy_returns(hist[-600:], bins=30)
                if (not np.isfinite(feats.entropy)) or (feats.entropy > self.s.ENTROPY_MAX):
                    return None
            if self.s.USE_LLE:
                feats.lle = lyapunov_rosenstein(log_returns(hist[-800:]), emb_dim=6, tau=1, max_t=18)
                if (not np.isfinite(feats.lle)) or (feats.lle > self.s.LLE_MAX):
                    return None
            if self.s.USE_MFDFA_PROXY:
                feats.mfdfa = mfdfa_proxy(hist[-800:])
                # If too "sticky" volatility, reduce size slightly
                if feats.mfdfa > 0.8:
                    risk_mult *= 0.8

            # contagion: correlations + optional transfer entropy
            if other_histories:
                feats.contagion, contag_rmult = self._contagion_score(sym, hist, other_histories)
                risk_mult *= contag_rmult

        # conservative: buy slightly below midpoint (maker-ish) to reduce fees if you later convert to maker logic
        limit_px = float(clamp(mid, 0.001, 0.999))
        return Signal(
            market_id=market.id,
            token_id=yes_token,
            side="BUY",
            limit_price=limit_px,
            risk_mult=float(clamp(risk_mult, 0.2, 1.0)),
            features=feats,
            meta={"symbol": sym, "direction": direction, "strike": f"{strike:.2f}", "tte_sec": str(tte)},
        )

    def _estimate_probability(self, *, spot: float, strike: float, direction: str, tte_sec: int, price_series: Sequence[float]) -> Tuple[float, float]:
        """
        Digital option proxy under lognormal with realized vol.
        Adds a small drift term from symbolic regression when enabled.
        """
        r = log_returns(price_series[-900:])
        if r.size < 60:
            return 0.5, 0.0

        # Estimate per-tick sigma and scale to horizon (ticks~seconds in this feed)
        sigma = float(np.std(r) + 1e-12)
        # Drift (log units per tick)
        drift = 0.0
        if self.s.USE_SYMBOLIC_REGRESSION:
            drift = float(symbolic_drift(price_series[-self.s.SR_LOOKBACK:], horizon_sec=self.s.SR_HORIZON_SEC, max_terms=self.s.SR_MAX_TERMS))
            # clamp: micro-drift should not dominate
            drift = float(clamp(drift, -3e-4, 3e-4))

        T = float(max(1, tte_sec))  # tick ~ second
        muT = drift * T
        sigT = sigma * math.sqrt(T)

        if sigT <= 1e-9:
            # deterministic
            if direction == "above":
                return (1.0 if spot > strike else 0.0), drift
            else:
                return (1.0 if spot < strike else 0.0), drift

        # P(S_T > K) under lognormal with drift muT in log-space
        d = (math.log(max(spot, 1e-12) / max(strike, 1e-12)) + muT - 0.5 * sigT * sigT) / sigT
        p_above = _norm_cdf(d)
        p = p_above if direction == "above" else (1.0 - p_above)
        return float(clamp(p, 0.0, 1.0)), drift

    def _contagion_score(self, sym: str, hist: Sequence[float], others: Dict[str, Sequence[float]]) -> Tuple[float, float]:
        """
        Combine correlation and optional transfer entropy into a 'contagion score'.
        Higher score => reduce risk.
        """
        r0 = log_returns(hist[-600:])
        if r0.size < 200:
            return 0.0, 1.0
        scores = []
        te_scores = []
        for osym, oh in others.items():
            if osym == sym:
                continue
            r1 = log_returns(oh[-600:])
            n = min(r0.size, r1.size)
            if n < 200:
                continue
            c = float(np.corrcoef(r0[-n:], r1[-n:])[0, 1])
            if math.isfinite(c):
                scores.append(abs(c))
            if self.s.USE_TRANSFER_ENTROPY:
                te_scores.append(transfer_entropy(r1[-n:], r0[-n:], bins=6))  # other -> sym
        corr = float(np.mean(scores)) if scores else 0.0
        te = float(np.mean(te_scores)) if te_scores else 0.0
        contag = 0.7 * corr + 0.3 * te
        if contag > self.s.CONTAGION_TE_MAX:
            # scale down quickly in contagion regimes (AMH risk-off)
            rmult = 0.5
        elif contag > self.s.CONTAGION_TE_MAX * 0.7:
            rmult = 0.75
        else:
            rmult = 1.0
        return float(contag), float(rmult)


_CRYPTO_RE = re.compile(r"\b(BTC|BITCOIN|ETH|ETHEREUM|SOL|SOLANA)\b", re.IGNORECASE)
_ABOVE_RE = re.compile(r"\b(ABOVE|OVER|AT LEAST|>=)\b", re.IGNORECASE)
_BELOW_RE = re.compile(r"\b(BELOW|UNDER|<=|AT MOST)\b", re.IGNORECASE)
_NUM_RE = re.compile(r"(\$?\s*[\d,]+(?:\.\d+)?)")


def _parse_crypto_strike(question: str) -> Optional[Tuple[str, str, float]]:
    """
    Very tolerant parser:
    - Detect underlying (BTC/ETH/SOL) and direction (above/below) and a numeric strike.
    - Assumes Yes means the statement is true.
    """
    q = (question or "").strip()
    m = _CRYPTO_RE.search(q)
    if not m:
        return None
    asset = m.group(1).upper()
    sym = {"BTC": "BTCUSDT", "BITCOIN": "BTCUSDT", "ETH": "ETHUSDT", "ETHEREUM": "ETHUSDT", "SOL": "SOLUSDT", "SOLANA": "SOLUSDT"}[asset]

    direction = "above" if _ABOVE_RE.search(q) else "below" if _BELOW_RE.search(q) else ""
    if not direction:
        # many crypto markets are phrased "reach" or "hit"; treat as above
        if "REACH" in q.upper() or "HIT" in q.upper():
            direction = "above"
        else:
            return None

    nm = _NUM_RE.search(q)
    if not nm:
        return None
    raw = nm.group(1).replace("$", "").replace(",", "").strip()
    try:
        strike = float(raw)
    except Exception:
        return None
    if strike <= 0:
        return None
    return sym, direction, strike


# ──────────────────────────────────────────────────────────────────────────────
# OpenClaw-style market-maker quoting strategy
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class QuotePlan:
    market_id: str
    condition_id: str
    token_id: str
    neg_risk: bool
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    features: Features
    # microstructure alpha diagnostics
    ob_imbalance: float = 0.0
    micro_shift: float = 0.0


class MarketMakerStrategy(Strategy):
    """
    Produces a two-sided quote plan (bid+ask) around a fair price estimate.
    Still uses the econophysics regime layer to avoid unstable/chaotic phases.
    """
    def _orderbook_alpha(self, book: OrderBook) -> Tuple[float, float]:
        """
        Returns (imbalance, micro_shift) where:
          imbalance in [-1,1] using top N depth
          micro_shift is (microprice - mid) in price units
        """
        if not book.bids or not book.asks:
            return 0.0, 0.0
        N = max(1, int(self.s.MM_DEPTH_LEVELS))
        bids = book.bids[:N]
        asks = book.asks[:N]
        bid_vol = float(sum(l.size for l in bids))
        ask_vol = float(sum(l.size for l in asks))
        denom = max(bid_vol + ask_vol, 1e-9)
        imb = (bid_vol - ask_vol) / denom

        bb = book.bids[0]
        ba = book.asks[0]
        mid = 0.5 * (bb.price + ba.price)
        micro = (ba.price * bb.size + bb.price * ba.size) / max(bb.size + ba.size, 1e-9)
        micro_shift = micro - mid
        return float(clamp(imb, -1.0, 1.0)), float(micro_shift)

    async def make_quote_plan(
        self,
        market: GammaMarket,
        book: OrderBook,
        *,
        now_ts: int,
        prices_by_symbol: Dict[str, float],
        history_by_symbol: Dict[str, Sequence[float]],
        other_histories: Dict[str, Sequence[float]] | None = None,
        inventory_usdc: float = 0.0,
    ) -> Optional[QuotePlan]:
        """
        Builds a 2-sided quote plan for the market YES token.

        inventory_usdc is positive when you are long YES (marked at mid),
        negative when you are effectively short.
        """
        if not market.enable_order_book:
            return None
        token_id = market.yes_token_id
        if not token_id:
            return None

        tte = market.end_ts - int(now_ts)
        if tte < self.s.MIN_TTE_SEC or tte > self.s.MAX_TTE_SEC:
            return None

        parsed = _parse_crypto_strike(market.question)
        if not parsed:
            return None
        sym, direction, strike = parsed
        spot = float(prices_by_symbol.get(sym.lower(), 0.0))
        if spot <= 0:
            return None

        series = history_by_symbol.get(sym.lower()) or []
        if len(series) < 60:
            return None

        # Mid from book
        if not book.bids or not book.asks:
            return None
        best_bid = float(book.bids[0].price)
        best_ask = float(book.asks[0].price)
        mid = float(clamp(0.5 * (best_bid + best_ask), 0.0001, 0.9999))

        p_est, drift = self._estimate_probability(spot=spot, strike=strike, direction=direction, tte_sec=tte, price_series=series)

        # Econophysics regime gates
        feats = Features(p_est=p_est, mid_price=mid, edge=(p_est - mid), drift=drift)
        if self.s.USE_ECONO:
            r = log_returns(series[-600:])
            if self.s.USE_HURST:
                feats.hurst = float(hurst_rs(r[-400:])) if r.size >= 120 else float(np.nan)
                if (not np.isfinite(feats.hurst)) or (feats.hurst < self.s.HURST_MIN):
                    return None
            if self.s.USE_ENTROPY:
                feats.entropy = float(shannon_entropy_returns(r[-400:])) if r.size >= 120 else float(np.nan)
                if (not np.isfinite(feats.entropy)) or (feats.entropy > self.s.ENTROPY_MAX):
                    return None
            if self.s.USE_LLE:
                feats.lle = float(lyapunov_rosenstein(series[-400:])) if len(series) >= 250 else float(np.nan)
                if (not np.isfinite(feats.lle)) or (feats.lle > self.s.LLE_MAX):
                    return None
            if self.s.USE_MFDFA_PROXY:
                feats.mfdfa = float(mfdfa_proxy(r[-500:])) if r.size >= 200 else 0.5

            if self.s.USE_TRANSFER_ENTROPY and other_histories:
                tes = []
                for _, other in other_histories.items():
                    try:
                        tes.append(float(transfer_entropy(other[-300:], series[-300:], k=1)))
                    except Exception:
                        pass
                feats.contagion = float(np.mean(tes)) if tes else 0.0
                if feats.contagion > self.s.CONTAGION_TE_MAX:
                    return None

        # Microstructure alpha from the current orderbook
        imb, micro_shift = self._orderbook_alpha(book)

        # Center price: fair estimate + microstructure shifts + inventory skew
        center = float(p_est)
        center += float(self.s.MM_ALPHA_IMB_K) * imb * (book.tick_size * 2.0)
        center += float(self.s.MM_ALPHA_MICRO_K) * micro_shift
        inv_frac = float(clamp(inventory_usdc / max(self.s.MM_MAX_INV_USDC, 1e-9), -1.0, 1.0))
        center -= (self.s.MM_INV_SKEW_BPS / 10_000.0) * inv_frac
        center = float(clamp(center, 0.0001, 0.9999))

        half = max(center * (self.s.MM_HALF_SPREAD_BPS / 10_000.0), book.tick_size * float(self.s.MM_MIN_HALF_SPREAD_TICKS))
        half = float(max(half, book.tick_size))

        bid = center - half
        ask = center + half

        # Enforce post-only: don't cross spread (leave 1 tick safety)
        bid = min(bid, best_ask - book.tick_size)
        ask = max(ask, best_bid + book.tick_size)

        bid = float(clamp(bid, 0.0001, 0.9999))
        ask = float(clamp(ask, 0.0001, 0.9999))
        if bid >= ask:
            return None

        base = float(self.s.MM_BASE_SIZE_SHARES)
        bid_sz = base * float(clamp(1.0 - inv_frac * self.s.MM_INV_SIZE_SKEW, 0.25, 2.0))
        ask_sz = base * float(clamp(1.0 + inv_frac * self.s.MM_INV_SIZE_SKEW, 0.25, 2.0))

        return QuotePlan(
            market_id=market.id,
            condition_id=market.condition_id or market.id,
            token_id=token_id,
            neg_risk=bool(book.neg_risk or market.neg_risk),
            bid_price=bid,
            ask_price=ask,
            bid_size=bid_sz,
            ask_size=ask_sz,
            features=feats,
            ob_imbalance=float(imb),
            micro_shift=float(micro_shift),
        )
