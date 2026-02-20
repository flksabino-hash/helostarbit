"""
main.py — runner (console-first)

Usage:
  python main.py          # paper mode (default)
  python main.py --live   # live mode (still requires LIVE_MODE=True in .env)

Safety:
- LIVE_MODE defaults to False.
- Even with --live, LIVE_MODE must be True to place orders.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Dict, List

import aiohttp

from config import load_settings
from exchange import BinancePriceFeed, PolymarketGammaClient, PolymarketCLOBPublic, PolymarketCLOBTrader
from risk_manager import PaperWallet, RiskManager, PaperPosition
from strategy import Strategy
from microstructure import run_market_maker
from utils import setup_logging, utc_ts, clamp

log = logging.getLogger("main")


def _relevant_market(q: str) -> bool:
    q = (q or "").upper()
    # fast filters: crypto + short horizon phrasing
    return ("BTC" in q or "BITCOIN" in q or "ETH" in q or "ETHEREUM" in q or "SOL" in q or "SOLANA" in q) and (
        "MIN" in q or "MINUTE" in q or "15" in q or "10" in q or "5" in q
    )


async def run(loop_seconds: int = 20, live_flag: bool = False) -> None:
    s = load_settings(".env")
    setup_logging(s.LOG_LEVEL)

    wallet = PaperWallet(start_usdc=s.PAPER_START_USDC)
    risk = RiskManager(s, wallet)
    strat = Strategy(s)

    async with aiohttp.ClientSession(headers={"User-Agent": "PolyScalper/0.1"}) as http:
        feed = BinancePriceFeed(s)
        gamma = PolymarketGammaClient(s, http)
        clob = PolymarketCLOBPublic(s, http)

        trader = None
        live = bool(live_flag and s.LIVE_MODE)
        if live:
            trader = PolymarketCLOBTrader(s, http)
            creds = await trader.create_or_derive_api_creds()
            # Compliance: check geoblock before placing orders (Polymarket docs recommend this)
            try:
                async with http.get("https://polymarket.com/api/geoblock", timeout=10) as r:
                    geo = await r.json()
                if geo.get("blocked"):
                    raise RuntimeError(f"Geoblocked for trading: {geo}")
            except Exception as e:
                raise RuntimeError(f"Geoblock check failed or blocked: {e}")

            log.warning("LIVE MODE ENABLED. Using CLOB apiKey=%s...", str(creds.get("apiKey", ""))[:8])

        await feed.start()

        try:
            while True:
                now = utc_ts()
                prices = feed.prices
                if not prices:
                    await asyncio.sleep(1)
                    continue

                eq = wallet.equity()
                if risk.in_drawdown_stop(eq):
                    log.error("Drawdown cap hit. Equity=%.2f. Halting new trades; still settling positions.", eq)

                # settle expired first
                settled = wallet.settle_expired(now_ts=now, underlying_prices=prices)
                for p in settled:
                    strat.on_trade_closed(p.pnl_pct())
                    log.info("SETTLED %s | pnl=%.2f USDC (%.2f%%) | bal=%.2f", p.market_id, p.pnl_usdc, p.pnl_pct()*100, wallet.balance_usdc)

                # fetch markets
                markets = await gamma.list_markets(active=True, closed=False, limit=160)
                candidates = [m for m in markets if _relevant_market(m.question)]
                # sort by expiry soonest
                candidates.sort(key=lambda m: m.end_ts)

                # Build histories map for strategy
                hist_map: Dict[str, List[float]] = {sym: feed.history_prices(sym, limit=1000) for sym in s.BINANCE_SYMBOLS}

                # Evaluate a handful per cycle to stay within rate limits
                for m in candidates[:30]:
                    if not m.enable_order_book:
                        continue
                    # do not open if at max or cooldown
                    # parse symbol from question in a cheap way
                    q = m.question.upper()
                    sym = "BTCUSDT" if "BTC" in q or "BITCOIN" in q else "ETHUSDT" if "ETH" in q else "SOLUSDT" if "SOL" in q else ""
                    if not sym:
                        continue
                    if risk.is_on_cooldown(m.id, sym, now):
                        continue
                    if not wallet.can_open(s.MAX_POSITIONS):
                        break
                    if risk.in_drawdown_stop(wallet.equity()):
                        break

                    sig = await strat.evaluate(
                        m, clob,
                        now_ts=now,
                        prices_by_symbol=prices,
                        history_by_symbol=hist_map,
                        other_histories=hist_map,
                    )
                    if not sig:
                        continue

                    size = risk.size_usdc(
                        equity=wallet.equity(),
                        symbol_prices=hist_map.get(sig.meta.get("symbol", sym), []),
                        risk_mult=sig.risk_mult,
                    )
                    if size <= 1.0:
                        continue

                    # paper execution
                    strike = float(sig.meta.get("strike", "0"))
                    direction = str(sig.meta.get("direction", "above"))
                    expiry = int(m.end_ts)
                    entry_price = float(sig.limit_price)
                    shares = size / entry_price
                    fees = size * (s.TAKER_FEE_BPS / 10_000.0) + s.GAS_USDC_ESTIMATE
                    pos = PaperPosition(
                        market_id=m.id,
                        token_id=sig.token_id,
                        symbol=str(sig.meta.get("symbol", sym)),
                        direction=direction,
                        strike=strike,
                        expiry_ts=expiry,
                        entry_price=entry_price,
                        shares=shares,
                        usdc_spent=size,
                        fees_usdc=fees,
                        opened_ts=now,
                    )
                    wallet.open_position(pos)
                    risk.set_cooldown(m.id, pos.symbol, now)
                    risk.note_slippage(limit_price=entry_price, mid_price=sig.features.mid_price)

                    log.info(
                        "OPEN %s %s | size=%.2f @%.3f | edge=%.3f | H=%.2f E=%.2f LLE=%.4f contag=%.2f rmult=%.2f | bal=%.2f",
                        pos.symbol, m.id, size, entry_price, sig.features.edge,
                        sig.features.hurst, sig.features.entropy, sig.features.lle, sig.features.contagion, sig.risk_mult,
                        wallet.balance_usdc
                    )

                    # live execution (optional)
                    if live and trader:
                        try:
                            exp = 0  # expiration unused for GTC; use GTD in MM loop
                            order = trader.sign_limit_order(
                                token_id=sig.token_id,
                                side="BUY",
                                price=entry_price,
                                size_shares=(size / max(entry_price, 1e-6)),
                                expiration_ts=exp,
                                fee_rate_bps=s.TAKER_FEE_BPS,
                                nonce=0,
                                neg_risk=False,
                            )
                            res = await trader.post_order(order, order_type="GTC", defer_exec=False)
                            log.warning("LIVE order result: %s", res)
                        except Exception as e:
                            log.error("LIVE order failed: %s", e)


                # write lightweight status snapshot for optional mobile UI
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    snap = {
                        "ts": now,
                        "equity": wallet.equity(),
                        "balance": wallet.balance_usdc,
                        "open_positions": len(wallet.positions),
                        "prices": prices,
                        "last_closed": (wallet.closed[-1].__dict__ if wallet.closed else None),
                    }
                    _Path("data").mkdir(exist_ok=True)
                    _Path("data/status.json").write_text(_json.dumps(snap, indent=2))
                except Exception:
                    pass

                await asyncio.sleep(loop_seconds)

        except asyncio.CancelledError:
            pass
        finally:
            await feed.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Attempt live mode (also requires LIVE_MODE=True in .env)")
    ap.add_argument("--mm", action="store_true", help="Run OpenClaw-style market-maker loop (post-only quotes)")
    ap.add_argument("--arm", action="store_true", help="Required for LIVE arming ceremony (MM/live only)")
    ap.add_argument("--loop", type=int, default=20, help="Market scan loop seconds")
    args = ap.parse_args()
    if args.mm:
        asyncio.run(run_market_maker(live_flag=args.live, arm_flag=args.arm))
    else:
        asyncio.run(run(loop_seconds=args.loop, live_flag=args.live))


if __name__ == "__main__":
    main()
