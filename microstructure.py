"""
microstructure.py — OpenClaw-style market-maker loop (hardened)

Adds on top of the earlier MM loop:
- **Batch cancel/replace** using /orders endpoints (reduces API load).
- **User-channel reconciliation** (fills/cancels update inventory from real events).
- **OrderRouter**: single choke-point so strategies cannot bypass risk gates.
- **LIVE arming ceremony**: LIVE requires runtime action (not just .env edits).

Safety:
- Paper trading by default.
- Live requires: LIVE_MODE=true in .env AND CLI --live AND (if LIVE_ARM_REQUIRED) CLI --arm interactive confirmation.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

from config import load_settings, Settings
from exchange import (
    BinancePriceFeed,
    PolymarketGammaClient,
    PolymarketCLOBPublic,
    PolymarketCLOBTrader,
    PolymarketMarketWS,
    PolymarketUserWS,
)
from order_router import OrderRouter, OrderIntent
from risk_manager import PaperWallet, RiskManager
from strategy import MarketMakerStrategy, QuotePlan
from utils import setup_logging, utc_ts, tick_round, clamp

log = logging.getLogger("mm")


@dataclass(slots=True)
class QuoteState:
    bid_order_id: str = ""
    ask_order_id: str = ""
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    bid_ts: int = 0
    ask_ts: int = 0


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _arm_live(s: Settings) -> bool:
    """
    Runtime arming step to prevent "LIVE by file tampering".
    Requires an interactive terminal.
    """
    if not s.LIVE_ARM_REQUIRED:
        return True

    if not sys.stdin or not sys.stdin.isatty():
        log.error("LIVE_ARM_REQUIRED=true but no interactive TTY is available. Refusing LIVE.")
        return False

    code = secrets.token_hex(3).upper()  # 6 hex chars
    log.warning("══════════════════════════════════════════════════════════════")
    log.warning("LIVE MODE ARMING REQUIRED.")
    log.warning("This will enable REAL order placement/cancellation.")
    log.warning("Type this code within %ss to arm: %s", s.LIVE_ARM_TIMEOUT_SEC, code)
    log.warning("══════════════════════════════════════════════════════════════")

    try:
        start = time.time()
        typed = input("ARM CODE> ").strip().upper()
        if typed != code:
            log.error("Arming failed: code mismatch.")
            return False
        if time.time() - start > float(s.LIVE_ARM_TIMEOUT_SEC):
            log.error("Arming failed: timeout.")
            return False
        log.warning("LIVE MODE ARMED. Use Ctrl+C as kill switch.")
        return True
    except KeyboardInterrupt:
        return False
    except Exception as e:
        log.error("Arming failed: %s", e)
        return False


class MarketMakerEngine:
    """
    Maintains quote state per token and drives a cancel/replace loop.
    All live order actions go through OrderRouter.
    """
    def __init__(
        self,
        s: Settings,
        wallet: PaperWallet,
        risk: RiskManager,
        strat: MarketMakerStrategy,
        *,
        http: aiohttp.ClientSession,
        feed: BinancePriceFeed,
        gamma: PolymarketGammaClient,
        public: PolymarketCLOBPublic,
        live_flag: bool,
        armed: bool,
    ):
        self.s = s
        self.wallet = wallet
        self.risk = risk
        self.strat = strat
        self.http = http
        self.feed = feed
        self.gamma = gamma
        self.public = public
        self.router = OrderRouter(s, wallet, risk)

        self.live_enabled = bool(live_flag and s.LIVE_MODE and armed)
        self.trader: Optional[PolymarketCLOBTrader] = None

        self.wss: Optional[PolymarketMarketWS] = None
        self.user_wss: Optional[PolymarketUserWS] = None
        self._user_q: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=5000)

        self.state: Dict[str, QuoteState] = {}
        self.order_index: Dict[str, Tuple[str, str]] = {}  # order_id -> (token_id, "bid"/"ask")

        self._hb_task: Optional[asyncio.Task] = None
        self._hb_id: str = ""

    async def _ensure_live(self) -> None:
        if not self.live_enabled or self.trader:
            return
        if not self.s.POLY_PRIVATE_KEY:
            raise RuntimeError("LIVE requested but POLY_PRIVATE_KEY is missing")

        # basic RPC scheme hardening (defense-in-depth)
        rpc = (self.s.POLYGON_RPC_URL or "").strip()
        if self.s.RPC_REQUIRE_HTTPS and rpc.startswith("http://") and "localhost" not in rpc:
            raise RuntimeError("Refusing http:// RPC in LIVE mode (set RPC_REQUIRE_HTTPS=false to override)")

        self.trader = PolymarketCLOBTrader(self.s, self.http)

        # Validate chainId if web3 is available
        try:
            cid = int(self.trader.w3.eth.chain_id)
            if cid != int(self.s.POLYGON_CHAIN_ID):
                raise RuntimeError(f"RPC chain_id mismatch: got {cid}, expected {self.s.POLYGON_CHAIN_ID}")
        except Exception as e:
            raise RuntimeError(f"RPC validation failed: {e}")

        await self.trader.init_signer()

        # heartbeat keepalive (required for resting orders)
        async def heartbeat_loop():
            assert self.trader is not None
            while True:
                try:
                    resp = await self.trader.post_heartbeat(self._hb_id)
                    self._hb_id = str(resp.get("heartbeat_id") or self._hb_id)
                except Exception as e:
                    log.warning("Heartbeat failed: %s", e)
                await asyncio.sleep(max(2, int(self.s.MM_HEARTBEAT_SEC)))

        self._hb_task = asyncio.create_task(heartbeat_loop())

    def _on_user_event(self, e: Dict[str, Any]) -> None:
        try:
            self._user_q.put_nowait(e)
        except Exception:
            # drop on overload; safest response is to reconcile later
            self.router.needs_reconcile = True

    async def _start_user_stream(self, condition_ids: Sequence[str]) -> None:
        if not self.live_enabled:
            return
        await self._ensure_live()
        assert self.trader is not None
        if self.user_wss:
            return
        self.user_wss = PolymarketUserWS(
            self.s,
            self.http,
            api_key=self.trader.api_key,
            api_secret=self.trader.api_secret,
            api_passphrase=self.trader.api_passphrase,
            on_event=self._on_user_event,
        )
        await self.user_wss.start(condition_ids)

    async def _drain_user_events(self) -> None:
        """
        Apply fills/cancels to wallet + internal state.
        This is conservative (partial fills handled as "clear & reconcile").
        """
        while True:
            try:
                e = self._user_q.get_nowait()
            except asyncio.QueueEmpty:
                break

            et = str(e.get("event_type") or "").lower()
            if et == "trade":
                try:
                    fee_bps = int(float(e.get("fee_rate_bps") or 0))
                except Exception:
                    fee_bps = 0
                makers = e.get("maker_orders") or []
                for mo in makers if isinstance(makers, list) else []:
                    oid = str(mo.get("order_id") or "")
                    if not oid or oid not in self.order_index:
                        continue
                    token_id = str(mo.get("asset_id") or e.get("asset_id") or "")
                    side = str(mo.get("side") or "").upper()
                    try:
                        px = float(mo.get("price") or e.get("price") or 0.0)
                        sh = float(mo.get("matched_amount") or 0.0)
                    except Exception:
                        continue
                    if token_id:
                        self.wallet.apply_external_fill(
                            order_id=oid,
                            token_id=token_id,
                            side=side,
                            price=px,
                            size_shares=sh,
                            fee_rate_bps=fee_bps,
                            ts=utc_ts(),
                        )
                    # clear state side
                    tok, which = self.order_index.pop(oid, ("", ""))
                    qs = self.state.get(tok)
                    if qs:
                        if which == "bid" and qs.bid_order_id == oid:
                            qs.bid_order_id = ""
                        if which == "ask" and qs.ask_order_id == oid:
                            qs.ask_order_id = ""
            elif et == "order":
                oid = str(e.get("id") or "")
                status = str(e.get("status") or "").upper()
                otype = str(e.get("type") or "").upper()
                if oid and (otype in ("CANCELLATION", "CANCEL") or "CANCEL" in status or "MATCHED" in status or "CLOSED" in status):
                    tok, which = self.order_index.pop(oid, ("", ""))
                    qs = self.state.get(tok)
                    if qs:
                        if which == "bid" and qs.bid_order_id == oid:
                            qs.bid_order_id = ""
                        if which == "ask" and qs.ask_order_id == oid:
                            qs.ask_order_id = ""
            else:
                continue

    async def _reconcile_open_orders(self, token_ids: Sequence[str]) -> None:
        """
        Pull open orders and rebuild internal state. Use sparingly.
        """
        if not self.live_enabled:
            self.router.needs_reconcile = False
            return
        await self._ensure_live()
        assert self.trader is not None

        try:
            j = await self.trader.get_open_orders(limit=200)
            data = j.get("data") or []
        except Exception as e:
            log.warning("Reconcile failed: %s", e)
            return

        # reset then rehydrate
        self.order_index.clear()
        for tid in token_ids:
            self.state.setdefault(str(tid), QuoteState())
            self.state[tid].bid_order_id = ""
            self.state[tid].ask_order_id = ""

        for o in data if isinstance(data, list) else []:
            oid = str(o.get("id") or "")
            tid = str(o.get("asset_id") or o.get("token_id") or "")
            side = str(o.get("side") or "").upper()
            if not oid or tid not in self.state:
                continue
            px = float(o.get("price") or 0.0)
            sz = float(o.get("original_size") or o.get("size") or 0.0)
            qs = self.state[tid]
            # best-effort: classify BUY as bid, SELL as ask
            if side == "BUY":
                qs.bid_order_id, qs.bid_price, qs.bid_size, qs.bid_ts = oid, px, sz, utc_ts()
                self.order_index[oid] = (tid, "bid")
            elif side == "SELL":
                qs.ask_order_id, qs.ask_price, qs.ask_size, qs.ask_ts = oid, px, sz, utc_ts()
                self.order_index[oid] = (tid, "ask")

        self.router.needs_reconcile = False

    def _needs_replace(self, *, qs: QuoteState, side: str, desired_px: float, desired_sz: float, tick: float, now: int) -> bool:
        side = side.lower()
        if side == "bid":
            oid, opx, osz, ots = qs.bid_order_id, qs.bid_price, qs.bid_size, qs.bid_ts
        else:
            oid, opx, osz, ots = qs.ask_order_id, qs.ask_price, qs.ask_size, qs.ask_ts

        if not oid:
            return desired_sz > 0

        # if desired size is zero -> cancel
        if desired_sz <= 0:
            return True

        # TTL rotation
        if ots and (now - ots) >= int(self.s.MM_ORDER_TTL_SEC):
            return True

        # price drift: > 0.5 tick
        if abs(float(desired_px) - float(opx)) >= 0.5 * float(tick):
            return True

        # size drift: > 10% or > 1 share
        if osz <= 1e-9:
            return True
        if abs(float(desired_sz) - float(osz)) >= max(1.0, 0.10 * float(osz)):
            return True

        return False

    async def _sync_quotes_batch(
        self,
        *,
        plans: Sequence[QuotePlan],
        books: Dict[str, Any],
    ) -> None:
        now = utc_ts()

        # mark prices for equity
        marks: Dict[str, float] = {}
        for tid, book in books.items():
            if book and getattr(book, "mid", 0.0):
                marks[tid] = float(book.mid)
        equity = float(self.wallet.equity(marks))

        if not self.risk.allowed_to_trade(equity):
            log.error("Risk governor halted trading: cancelling all local quotes.")
            await self.cancel_all_quotes(list(books.keys()))
            return

        cancels: List[str] = []
        intents: List[OrderIntent] = []
        intent_meta: List[Tuple[str, str]] = []  # (token_id, "bid"/"ask") for mapping return ids

        for p in plans:
            tid = str(p.token_id)
            book = books.get(tid)
            if not book or not getattr(book, "bids", None) or not getattr(book, "asks", None):
                continue

            tick = float(getattr(book, "tick_size", 0.01) or 0.01)
            best_bid = float(book.bids[0].price)
            best_ask = float(book.asks[0].price)
            inv_usdc = float(self.wallet.inventory_value_usdc(tid, float(book.mid)))

            qs = self.state.setdefault(tid, QuoteState())

            # Build bid/ask intents (post-only enforced locally in router)
            bid_int = OrderIntent(
                token_id=tid,
                side="BUY",
                price=float(clamp(p.bid_price, 0.0001, 0.9999)),
                size_shares=float(max(0.0, p.bid_size)),
                tick_size=tick,
                neg_risk=bool(p.neg_risk),
                order_type=str(self.s.MM_ORDER_TYPE),
                post_only=bool(self.s.MM_POST_ONLY),
            )
            ask_int = OrderIntent(
                token_id=tid,
                side="SELL",
                price=float(clamp(p.ask_price, 0.0001, 0.9999)),
                size_shares=float(max(0.0, p.ask_size)),
                tick_size=tick,
                neg_risk=bool(p.neg_risk),
                order_type=str(self.s.MM_ORDER_TYPE),
                post_only=bool(self.s.MM_POST_ONLY),
            )

            # enforce post-only locally and risk caps
            bid_int = self.router._enforce_post_only_local(bid_int, best_bid=best_bid, best_ask=best_ask)
            ask_int = self.router._enforce_post_only_local(ask_int, best_bid=best_bid, best_ask=best_ask)

            bid_int = self.router.cap_size(bid_int, equity=equity, inventory_usdc=inv_usdc)
            ask_int = self.router.cap_size(ask_int, equity=equity, inventory_usdc=inv_usdc)

            # Decide replace per side
            if self._needs_replace(qs=qs, side="bid", desired_px=bid_int.price, desired_sz=bid_int.size_shares, tick=tick, now=now):
                if qs.bid_order_id:
                    cancels.append(qs.bid_order_id)
                    self.order_index.pop(qs.bid_order_id, None)
                    qs.bid_order_id = ""
                if bid_int.size_shares > 0:
                    intents.append(bid_int)
                    intent_meta.append((tid, "bid"))
                    qs.bid_price, qs.bid_size, qs.bid_ts = bid_int.price, bid_int.size_shares, now

            if self._needs_replace(qs=qs, side="ask", desired_px=ask_int.price, desired_sz=ask_int.size_shares, tick=tick, now=now):
                if qs.ask_order_id:
                    cancels.append(qs.ask_order_id)
                    self.order_index.pop(qs.ask_order_id, None)
                    qs.ask_order_id = ""
                if ask_int.size_shares > 0:
                    intents.append(ask_int)
                    intent_meta.append((tid, "ask"))
                    qs.ask_price, qs.ask_size, qs.ask_ts = ask_int.price, ask_int.size_shares, now

        # Execute cancels then posts
        await self._ensure_live()
        trader = self.trader

        await self.router.cancel_many(trader, cancels, live=self.live_enabled)  # type: ignore[arg-type]

        # Chunk posts to batch limit
        if intents:
            placed_ids: List[str] = []
            chunk = int(max(1, self.s.MM_BATCH_MAX_ORDERS))
            for i in range(0, len(intents), chunk):
                ids = await self.router.place_many(trader, intents[i:i+chunk], live=self.live_enabled)  # type: ignore[arg-type]
                placed_ids.extend(ids)

            for oid, (tid, which) in zip(placed_ids, intent_meta):
                if not oid:
                    continue
                qs = self.state.setdefault(tid, QuoteState())
                if which == "bid":
                    qs.bid_order_id = oid
                else:
                    qs.ask_order_id = oid
                self.order_index[oid] = (tid, which)

    async def cancel_all_quotes(self, token_ids: Sequence[str]) -> None:
        cancels: List[str] = []
        for tid in token_ids:
            qs = self.state.get(str(tid))
            if not qs:
                continue
            if qs.bid_order_id:
                cancels.append(qs.bid_order_id)
                qs.bid_order_id = ""
            if qs.ask_order_id:
                cancels.append(qs.ask_order_id)
                qs.ask_order_id = ""
        await self._ensure_live()
        await self.router.cancel_many(self.trader, cancels, live=self.live_enabled)  # type: ignore[arg-type]

    async def run(self) -> None:
        log.warning("MM loop starting. LIVE=%s", self.live_enabled)

        # Select a small universe of near-term markets
        markets = await self.gamma.list_markets(max_markets=200)
        near = self.strat.select_mm_markets(markets)[: int(self.s.MM_MAX_ASSETS)]
        if not near:
            log.error("No MM markets found. Exiting.")
            return

        token_ids: List[str] = []
        condition_ids: List[str] = []
        for m in near:
            for tid in m.clob_token_ids[:2]:
                token_ids.append(str(tid))
            if m.condition_id:
                condition_ids.append(str(m.condition_id))

        token_ids = list(dict.fromkeys(token_ids))
        condition_ids = list(dict.fromkeys(condition_ids))

        # Start market WS
        self.wss = PolymarketMarketWS(self.s, self.http)
        await self.wss.start(token_ids)

        # Start user WS (live only)
        await self._start_user_stream(condition_ids)

        last = _now_ms()
        while True:
            try:
                # user events first (fills/cancels)
                await self._drain_user_events()

                # reconcile if we had ambiguous API outcomes
                if self.router.needs_reconcile and self.live_enabled:
                    await self._reconcile_open_orders(token_ids)

                # snapshot books
                books: Dict[str, Any] = {}
                for tid in token_ids:
                    b = self.wss.get_book(tid) if self.wss else None
                    if not b:
                        b = await self.public.get_order_book(tid)
                    books[tid] = b

                # build quote plans
                plans: List[QuotePlan] = []
                for m in near:
                    for tid in m.clob_token_ids[:2]:
                        b = books.get(str(tid))
                        if not b:
                            continue
                        p = self.strat.make_mm_quote_plan(m, str(tid), b)
                        if p:
                            plans.append(p)

                await self._sync_quotes_batch(plans=plans, books=books)

                # Paper fill simulator when not live
                if not self.live_enabled:
                    tob: Dict[str, Tuple[float, float]] = {}
                    for tid, b in books.items():
                        if b and b.bids and b.asks:
                            tob[tid] = (float(b.bids[0].price), float(b.asks[0].price))
                    self.wallet.simulate_mm_fills(best_bid_ask=tob, ts=utc_ts(), maker_fee_bps=int(self.s.MAKER_FEE_BPS))

                # pace loop
                dt = _now_ms() - last
                last = _now_ms()
                sleep_s = float(max(0.0, float(self.s.MM_REFRESH_SEC) - dt / 1000.0))
                await asyncio.sleep(sleep_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("MM loop error: %s", e)
                await asyncio.sleep(2)


async def run_market_maker(live_flag: bool = False, arm_flag: bool = False) -> None:
    s = load_settings(".env")
    setup_logging(s.LOG_LEVEL)

    armed = False
    if live_flag and s.LIVE_MODE and arm_flag:
        armed = await _arm_live(s)
    elif live_flag and s.LIVE_MODE and s.LIVE_ARM_REQUIRED:
        log.error("LIVE requested but not armed. Re-run with --arm to complete arming ceremony.")

    wallet = PaperWallet(start_usdc=s.PAPER_START_USDC)
    risk = RiskManager(s, wallet)
    strat = MarketMakerStrategy(s)

    async with aiohttp.ClientSession(headers={"User-Agent": "PolyScalper/0.1"}) as http:
        feed = BinancePriceFeed(s)
        gamma = PolymarketGammaClient(s, http)
        public = PolymarketCLOBPublic(s, http)

        eng = MarketMakerEngine(
            s,
            wallet,
            risk,
            strat,
            http=http,
            feed=feed,
            gamma=gamma,
            public=public,
            live_flag=live_flag,
            armed=armed,
        )
        try:
            await eng.run()
        finally:
            await feed.close()
