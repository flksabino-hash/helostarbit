"""
exchange.py — market-data + execution wrappers

Components:
- BinancePriceFeed: async websocket feed for BTC/ETH/SOL spot trades (public).
- PolymarketGammaClient: scans Gamma /markets (public) for candidate short-dated markets.
- PolymarketCLOBPublic: reads orderbooks/midpoints/spreads (public).
- PolymarketMarketWS: orderbook/trade stream (public WebSocket).
- PolymarketUserWS: order/trade stream (authenticated WebSocket; optional).
- PolymarketCLOBTrader: optional live trading (requires L1/L2 auth + EIP-712 order signing).

Safety:
- LIVE_MODE defaults to False (paper trading only).
- Even with LIVE_MODE=True, this project is educational — test with tiny sizes and comply with local laws.
"""
from __future__ import annotations

import contextlib
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import aiohttp

from config import Settings
from utils import (
    safe_json_dumps,
    utc_ts,
    quantize_down,
    clamp,
)

# web3 is only needed for live order signing / auth
try:
    from web3 import Web3
    from eth_account import Account
    from eth_account.messages import encode_typed_data
    HAS_WEB3 = True
except Exception:  # pragma: no cover
    HAS_WEB3 = False


log = logging.getLogger("exchange")


# ──────────────────────────────────────────────────────────────────────────────
# Binance: spot prices via websocket (public)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Tick:
    ts: float
    price: float


class BinancePriceFeed:
    """
    Minimal Binance websocket price feed (trade stream).
    Not a trading connector — just a market-data feed for our probability estimator.
    """
    def __init__(self, settings: Settings):
        self.s = settings
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._prices: Dict[str, float] = {}
        self._history: Dict[str, List[Tick]] = {sym: [] for sym in settings.BINANCE_SYMBOLS}

    @property
    def prices(self) -> Dict[str, float]:
        return dict(self._prices)

    def history_prices(self, symbol: str, limit: int = 600) -> List[float]:
        h = self._history.get(symbol, [])
        return [t.price for t in h[-limit:]]

    async def start(self) -> None:
        if self._task:
            return
        self._session = aiohttp.ClientSession(headers={"User-Agent": "PolyScalper/0.2"})
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(Exception):
                await self._task
        self._task = None
        if self._session:
            await self._session.close()
        self._session = None

    async def _run(self) -> None:
        assert self._session is not None
        # Binance combined stream: e.g. btcusdt@trade/ethusdt@trade
        streams = "/".join([f"{sym.lower()}@trade" for sym in self.s.BINANCE_SYMBOLS])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        while True:
            try:
                async with self._session.ws_connect(url, heartbeat=30) as ws:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        ev = data.get("data", {})
                        sym = str(ev.get("s", "")).lower()
                        price = float(ev.get("p"))
                        self._prices[sym] = price
                        h = self._history.setdefault(sym, [])
                        h.append(Tick(ts=float(ev.get("T", 0)) / 1000.0, price=price))
                        # cap history
                        if len(h) > self.s.BINANCE_HISTORY_LIMIT:
                            del h[: len(h) - self.s.BINANCE_HISTORY_LIMIT]
            except Exception as e:
                log.warning("Binance WS disconnected (%s). Reconnecting...", e)
                await asyncio.sleep(1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Polymarket Gamma: market discovery (public REST)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_iso_to_ts(s: Any) -> int:
    if not s:
        return 0
    if isinstance(s, (int, float)):
        return int(s)
    try:
        # Gamma typically uses ISO8601
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return 0


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_json_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except Exception:
            return [p.strip() for p in s.split(",") if p.strip()]
    return []


@dataclass(slots=True)
class GammaMarket:
    """
    Unified view of Gamma market entries with the fields we need.
    NOTE: Gamma schemas evolve; keep this parser tolerant.
    """
    id: str                          # Gamma market id (often condition id, but not guaranteed)
    question: str
    end_ts: int
    outcomes: List[str]
    outcome_prices: List[float]
    clob_token_ids: List[str]
    enable_order_book: bool

    # optional / best-effort fields
    condition_id: str = ""
    neg_risk: bool = False
    order_price_min_tick_size: Optional[float] = None
    order_min_size: Optional[float] = None

    @property
    def yes_price(self) -> Optional[float]:
        for i, o in enumerate(self.outcomes):
            if o.strip().lower() == "yes" and i < len(self.outcome_prices):
                return float(self.outcome_prices[i])
        return None

    @property
    def yes_token_id(self) -> Optional[str]:
        for i, o in enumerate(self.outcomes):
            if o.strip().lower() == "yes" and i < len(self.clob_token_ids):
                return str(self.clob_token_ids[i])
        return None


class PolymarketGammaClient:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session

    async def list_markets(self, *, active: bool = True, closed: bool = False, limit: int = 100) -> List[GammaMarket]:
        params = {"active": str(active).lower(), "closed": str(closed).lower(), "limit": str(limit)}
        url = f"{self.s.POLY_GAMMA_BASE}/markets"
        async with self.http.get(url, params=params, timeout=12) as r:
            r.raise_for_status()
            data = await r.json()

        arr = data.get("markets") if isinstance(data, dict) else data
        if not isinstance(arr, list):
            return []

        out: List[GammaMarket] = []
        for m in arr:
            try:
                mid = str(m.get("id", "")).strip()
                q = str(m.get("question", "")).strip()
                if not mid or not q:
                    continue
                end_date = m.get("endDate") or m.get("end_date") or m.get("endDateIso") or m.get("end_date_iso")
                end_ts = _parse_iso_to_ts(end_date)

                outcomes = _parse_json_list(m.get("outcomes"))
                prices = [float(x) for x in _parse_json_list(m.get("outcomePrices"))]
                token_ids = [str(x) for x in _parse_json_list(m.get("clobTokenIds"))]
                enable = bool(m.get("enableOrderBook", False))

                cond = str(m.get("conditionId") or m.get("condition_id") or m.get("conditionID") or "").strip()
                neg_risk = bool(m.get("negRisk") or m.get("neg_risk") or m.get("isNegRisk") or False)

                out.append(GammaMarket(
                    id=mid,
                    question=q,
                    end_ts=end_ts,
                    outcomes=outcomes,
                    outcome_prices=prices,
                    clob_token_ids=token_ids,
                    enable_order_book=enable,
                    condition_id=cond or mid,
                    neg_risk=neg_risk,
                    order_price_min_tick_size=_safe_float(m.get("orderPriceMinTickSize") or m.get("order_price_min_tick_size")),
                    order_min_size=_safe_float(m.get("orderMinSize") or m.get("order_min_size")),
                ))
            except Exception:
                continue
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Polymarket CLOB public endpoints (public REST)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(slots=True)
class OrderBook:
    token_id: str
    market: str
    timestamp: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    min_order_size: float
    tick_size: float
    neg_risk: bool = False
    hash: str = ""


class PolymarketCLOBPublic:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session

    async def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
        url = f"{self.s.POLY_CLOB_BASE}/book"
        async with self.http.get(url, params={"token_id": token_id}, timeout=10) as r:
            if r.status != 200:
                return None
            j = await r.json()
        try:
            bids = [OrderBookLevel(price=float(x["price"]), size=float(x["size"])) for x in j.get("bids", [])]
            asks = [OrderBookLevel(price=float(x["price"]), size=float(x["size"])) for x in j.get("asks", [])]
            return OrderBook(
                token_id=str(j.get("asset_id") or token_id),
                market=str(j.get("market") or ""),
                timestamp=str(j.get("timestamp") or ""),
                bids=bids,
                asks=asks,
                min_order_size=float(j.get("min_order_size") or 0.0),
                tick_size=float(j.get("tick_size") or 0.01),
                neg_risk=bool(j.get("neg_risk") or False),
                hash=str(j.get("hash") or ""),
            )
        except Exception:
            return None

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        url = f"{self.s.POLY_CLOB_BASE}/midpoint"
        async with self.http.get(url, params={"token_id": token_id}, timeout=10) as r:
            if r.status != 200:
                return None
            j = await r.json()
        try:
            return float(j.get("mid_price"))
        except Exception:
            return None

    async def get_spread(self, token_id: str) -> Optional[float]:
        url = f"{self.s.POLY_CLOB_BASE}/spread"
        async with self.http.get(url, params={"token_id": token_id}, timeout=10) as r:
            if r.status != 200:
                return None
            j = await r.json()
        try:
            return float(j.get("spread"))
        except Exception:
            return None

    async def get_tick_size(self, token_id: Optional[str] = None) -> Optional[float]:
        url = f"{self.s.POLY_CLOB_BASE}/tick-size"
        params = {"token_id": token_id} if token_id else None
        async with self.http.get(url, params=params, timeout=10) as r:
            if r.status != 200:
                return None
            j = await r.json()
        try:
            return float(j.get("minimum_tick_size"))
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Polymarket WebSocket market/user channels (optional but recommended for MM)
# ──────────────────────────────────────────────────────────────────────────────

class PolymarketMarketWS:
    """
    Public market channel stream.

    Maintains an in-memory orderbook cache per asset_id (token_id). This is a best-effort
    incremental book that is "good enough" for quoting and alpha features.

    Docs: market channel streams book snapshots + price_change updates.
    """
    def __init__(
        self,
        settings: Settings,
        session: aiohttp.ClientSession,
        *,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.s = settings
        self.http = session
        self.on_event = on_event
        self._task: Optional[asyncio.Task] = None
        self._books: Dict[str, OrderBook] = {}
        # internal dicts for incremental updates (price->size)
        self._bids: Dict[str, Dict[float, float]] = {}
        self._asks: Dict[str, Dict[float, float]] = {}

    def get_book(self, token_id: str) -> Optional[OrderBook]:
        return self._books.get(token_id)

    def top_of_book(self, token_id: str) -> Optional[Tuple[OrderBookLevel, OrderBookLevel, float]]:
        b = self._books.get(token_id)
        if not b or not b.bids or not b.asks:
            return None
        best_bid = b.bids[0]
        best_ask = b.asks[0]
        mid = 0.5 * (best_bid.price + best_ask.price)
        return best_bid, best_ask, mid

    async def start(self, token_ids: Sequence[str]) -> None:
        if self._task:
            return
        token_ids = list(dict.fromkeys([str(t) for t in token_ids if t]))
        if not token_ids:
            raise ValueError("No token_ids to subscribe")
        self._task = asyncio.create_task(self._run(token_ids))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(Exception):
                await self._task
        self._task = None

    async def _run(self, token_ids: Sequence[str]) -> None:
        sub_msg = {
            "type": "market",
            "assets_ids": list(token_ids),
            "custom_feature_enabled": True,
        }
        url = self.s.POLY_WSS_MARKET
        while True:
            try:
                async with self.http.ws_connect(url, heartbeat=0) as ws:
                    await ws.send_json(auth_msg)
                    await ws.send_json(sub_msg)
                    # heartbeat: send "PING" every 10 seconds
                    async def pinger():
                        while True:
                            await asyncio.sleep(10)
                            await ws.send_json({})
                    ping_task = asyncio.create_task(pinger())
                    try:
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                if msg.data in ("PONG", "PING", "{}", ""):
                                    continue
                                data = json.loads(msg.data)
                                self._handle_market_event(data)
                                if self.on_event:
                                    self.on_event(data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                    finally:
                        ping_task.cancel()
                        with contextlib.suppress(Exception):
                            await ping_task
            except Exception as e:
                log.warning("Polymarket market WS disconnected (%s). Reconnecting...", e)
                await asyncio.sleep(1.5)

    def _handle_market_event(self, ev: Dict[str, Any]) -> None:
        et = str(ev.get("event_type") or ev.get("type") or "").lower()
        asset = str(ev.get("asset_id") or ev.get("asset_id") or ev.get("assetId") or "")
        if not asset:
            # Some events carry asset_id inside payload; ignore for now.
            asset = str(ev.get("asset") or "")
        if not asset:
            # try for book snapshot
            asset = str(ev.get("asset_id") or ev.get("assetId") or "")
        # book snapshot
        if et == "book":
            token_id = str(ev.get("asset_id") or ev.get("assetId") or "")
            if not token_id:
                return
            bids = [OrderBookLevel(float(x["price"]), float(x["size"])) for x in ev.get("bids", [])]
            asks = [OrderBookLevel(float(x["price"]), float(x["size"])) for x in ev.get("asks", [])]
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            tick = float(ev.get("tick_size") or ev.get("tickSize") or 0.01)
            mn = float(ev.get("min_order_size") or ev.get("minOrderSize") or 0.0)
            ob = OrderBook(
                token_id=token_id,
                market=str(ev.get("market") or ""),
                timestamp=str(ev.get("timestamp") or ""),
                bids=bids,
                asks=asks,
                min_order_size=mn,
                tick_size=tick,
                neg_risk=bool(ev.get("neg_risk") or False),
                hash=str(ev.get("hash") or ""),
            )
            self._books[token_id] = ob
            self._bids[token_id] = {lvl.price: lvl.size for lvl in bids}
            self._asks[token_id] = {lvl.price: lvl.size for lvl in asks}
            return

        # incremental price updates
        if et == "price_change":
            token_id = str(ev.get("asset_id") or ev.get("assetId") or "")
            if not token_id:
                return
            bidmap = self._bids.setdefault(token_id, {})
            askmap = self._asks.setdefault(token_id, {})
            for ch in ev.get("price_changes", []) or ev.get("priceChanges", []) or []:
                side = str(ch.get("side") or "").upper()
                price = float(ch.get("price"))
                size = float(ch.get("size"))
                if side == "BUY":
                    if size <= 0:
                        bidmap.pop(price, None)
                    else:
                        bidmap[price] = size
                elif side == "SELL":
                    if size <= 0:
                        askmap.pop(price, None)
                    else:
                        askmap[price] = size
            # rebuild top levels (cap to N to keep it cheap)
            bids = [OrderBookLevel(p, s) for p, s in bidmap.items()]
            asks = [OrderBookLevel(p, s) for p, s in askmap.items()]
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            old = self._books.get(token_id)
            tick = old.tick_size if old else float(ev.get("tick_size") or 0.01)
            mn = old.min_order_size if old else 0.0
            self._books[token_id] = OrderBook(
                token_id=token_id,
                market=old.market if old else "",
                timestamp=str(ev.get("timestamp") or ""),
                bids=bids[:200],
                asks=asks[:200],
                min_order_size=mn,
                tick_size=tick,
                neg_risk=old.neg_risk if old else False,
                hash=str(ev.get("hash") or old.hash if old else ""),
            )
            return


class PolymarketUserWS:
    """
    Authenticated user channel stream.

    Only used in LIVE mode to track fills and order state.
    Subscribes by condition IDs (market identifiers), not asset IDs.
    """
    def __init__(
        self,
        settings: Settings,
        session: aiohttp.ClientSession,
        *,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.s = settings
        self.http = session
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.on_event = on_event
        self._task: Optional[asyncio.Task] = None

    async def start(self, condition_ids: Sequence[str]) -> None:
        if self._task:
            return
        condition_ids = list(dict.fromkeys([str(x) for x in condition_ids if x]))
        auth_msg = {
            "type": "user",
            "auth": {"apiKey": self.api_key, "secret": self.api_secret, "passphrase": self.api_passphrase},
        }
        sub_msg = {"operation": "subscribe", "markets": condition_ids}
        self._task = asyncio.create_task(self._run(auth_msg, sub_msg))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(Exception):
                await self._task
        self._task = None

    async def _run(self, auth_msg: Dict[str, Any], sub_msg: Dict[str, Any]) -> None:
        url = self.s.POLY_WSS_USER
        while True:
            try:
                async with self.http.ws_connect(url, heartbeat=0) as ws:
                    await ws.send_json(auth_msg)
                    await ws.send_json(sub_msg)

                    async def pinger():
                        while True:
                            await asyncio.sleep(10)
                            await ws.send_json({})
                    ping_task = asyncio.create_task(pinger())
                    try:
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                if msg.data in ("PONG", "PING", "{}", ""):
                                    continue
                                data = json.loads(msg.data)
                                if self.on_event:
                                    self.on_event(data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                    finally:
                        ping_task.cancel()
                        with contextlib.suppress(Exception):
                            await ping_task
            except Exception as e:
                log.warning("Polymarket user WS disconnected (%s). Reconnecting...", e)
                await asyncio.sleep(1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Polymarket CLOB trading (L1 + L2 auth + EIP712 orders)
# ──────────────────────────────────────────────────────────────────────────────

# Domain constants from Polymarket open-source order-utils
PROTOCOL_NAME = "Polymarket CTF Exchange"
PROTOCOL_VERSION = "1"

# CLOB auth message from docs
CLOB_AUTH_MSG = "This message attests that I control the given wallet"


class PolymarketCLOBTrader:
    """
    Auth flow:
    - L1: EIP-712 signature over ClobAuth struct to create/derive API key.
    - L2: HMAC over request to authenticated endpoints.
    - Orders themselves must be EIP-712 signed payloads (EIP712Domain = exchange contract).

    NOTE: Polymarket uses "limit order" primitives for everything.
    For GTC/GTD (limit orders), `size` is expressed in SHARES (not USDC).
    """
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session
        if not HAS_WEB3:
            raise RuntimeError("web3/eth-account not installed; install web3 to use live mode")
        if not self.s.POLY_PRIVATE_KEY:
            raise RuntimeError("POLY_PRIVATE_KEY missing")
        self.w3 = Web3(Web3.HTTPProvider(self.s.POLYGON_RPC_URL, request_kwargs={"timeout": 15}))
        self.acct = Account.from_key(self.s.POLY_PRIVATE_KEY)
        self.signer = Web3.to_checksum_address(self.acct.address)
        self.funder = Web3.to_checksum_address(self.s.POLY_FUNDER_ADDRESS) if self.s.POLY_FUNDER_ADDRESS else self.signer

        self.api_key = self.s.POLY_API_KEY
        self.api_secret = self.s.POLY_API_SECRET
        self.api_passphrase = self.s.POLY_API_PASSPHRASE

    async def create_or_derive_api_creds(self) -> Dict[str, str]:
        ts = str(utc_ts())
        nonce = 0
        sig = self._sign_clob_auth(ts=ts, nonce=nonce)
        headers = {
            "POLY_ADDRESS": self.signer,
            "POLY_SIGNATURE": sig,
            "POLY_TIMESTAMP": ts,
            "POLY_NONCE": str(nonce),
        }
        derive_url = f"{self.s.POLY_CLOB_BASE}/auth/derive-api-key"
        create_url = f"{self.s.POLY_CLOB_BASE}/auth/api-key"

        async with self.http.get(derive_url, headers=headers, timeout=12) as r:
            if r.status == 200:
                creds = await r.json()
                self._set_creds(creds)
                return creds

        async with self.http.post(create_url, headers=headers, timeout=12) as r:
            r.raise_for_status()
            creds = await r.json()
            self._set_creds(creds)
            return creds

    def _set_creds(self, creds: Dict[str, Any]) -> None:
        self.api_key = str(creds.get("apiKey", "")).strip()
        self.api_secret = str(creds.get("secret", "")).strip()
        self.api_passphrase = str(creds.get("passphrase", "")).strip()

    def _sign_clob_auth(self, ts: str, nonce: int) -> str:
        typed = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "ClobAuth": [
                    {"name": "address", "type": "address"},
                    {"name": "timestamp", "type": "string"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "message", "type": "string"},
                ],
            },
            "primaryType": "ClobAuth",
            "domain": {
                "name": "Polymarket",
                "version": "1",
                "chainId": self.s.POLY_CHAIN_ID,
                "verifyingContract": "0x0000000000000000000000000000000000000000",
            },
            "message": {
                "address": self.signer,
                "timestamp": str(ts),
                "nonce": int(nonce),
                "message": CLOB_AUTH_MSG,
            },
        }
        msg = encode_typed_data(full_message=typed)
        signed = self.acct.sign_message(msg)
        return signed.signature.hex()

    def _l2_headers(self, method: str, path: str, body: Any) -> Dict[str, str]:
        if not (self.api_key and self.api_secret and self.api_passphrase):
            raise RuntimeError("Missing L2 creds. Call create_or_derive_api_creds() first.")
        ts = str(utc_ts())
        if body is None or body == "":
            body_str = ""
        elif isinstance(body, str):
            body_str = body
        else:
            body_str = safe_json_dumps(body)
        payload = ts + method.upper() + path + body_str
        secret = base64.urlsafe_b64decode(self.api_secret + "==")
        sig = base64.urlsafe_b64encode(hmac.new(secret, payload.encode(), hashlib.sha256).digest()).decode().rstrip("=")
        return {
            "POLY_ADDRESS": self.signer,
            "POLY_API_KEY": self.api_key,
            "POLY_PASSPHRASE": self.api_passphrase,
            "POLY_TIMESTAMP": ts,
            "POLY_SIGNATURE": sig,
        }

    def sign_limit_order(
        self,
        *,
        token_id: str,
        side: str,
        price: float,
        size_shares: float,
        expiration_ts: int = 0,
        fee_rate_bps: int = 0,
        nonce: int = 0,
        neg_risk: bool = False,
    ) -> Dict[str, Any]:
        """
        Create + EIP712-sign a LIMIT order payload suitable for POST /order with GTC/GTD.

        For limit orders on Polymarket:
          - BUY size is shares.
          - SELL size is shares.
          - USDC amounts are derived as size * price.

        maker/taker mapping (docs):
          BUY: makerAmount=USDC spent, takerAmount=shares received
          SELL: makerAmount=shares sold, takerAmount=USDC received
        """
        side_u = side.strip().upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        price = float(clamp(price, 0.0001, 0.9999))
        size_shares = float(max(0.0, size_shares))

        shares_amt = int(quantize_down(size_shares, 1e-6) * 1_000_000)
        usdc_amt = int(quantize_down(size_shares * price, 1e-6) * 1_000_000)

        if side_u == "BUY":
            maker_amt = usdc_amt
            taker_amt = shares_amt
            side_num = 0
        else:
            maker_amt = shares_amt
            taker_amt = usdc_amt
            side_num = 1

        salt = int.from_bytes(os.urandom(8), "big", signed=False)
        verifying = Web3.to_checksum_address(self.s.POLY_NEG_RISK_EXCHANGE if neg_risk else self.s.POLY_CTF_EXCHANGE)
        token_int = int(str(token_id), 0)  # supports "0x.." or decimal

        order_msg = {
            "salt": int(salt),
            "maker": self.funder,
            "signer": self.signer,
            "taker": "0x0000000000000000000000000000000000000000",
            "tokenId": int(token_int),
            "makerAmount": int(maker_amt),
            "takerAmount": int(taker_amt),
            "expiration": int(expiration_ts),
            "nonce": int(nonce),
            "feeRateBps": int(fee_rate_bps),
            "side": int(side_num),
            "signatureType": int(self.s.POLY_SIGNATURE_TYPE),
        }

        typed = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Order": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
            },
            "primaryType": "Order",
            "domain": {
                "name": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "chainId": self.s.POLY_CHAIN_ID,
                "verifyingContract": verifying,
            },
            "message": order_msg,
        }
        msg = encode_typed_data(full_message=typed)
        signed = self.acct.sign_message(msg)

        payload = {
            "maker": order_msg["maker"],
            "signer": order_msg["signer"],
            "taker": order_msg["taker"],
            "tokenId": str(token_id),
            "makerAmount": str(order_msg["makerAmount"]),
            "takerAmount": str(order_msg["takerAmount"]),
            "side": "BUY" if side_num == 0 else "SELL",
            "expiration": str(order_msg["expiration"]),
            "nonce": str(order_msg["nonce"]),
            "feeRateBps": str(order_msg["feeRateBps"]),
            "signature": signed.signature.hex(),
            "salt": int(order_msg["salt"]),
            "signatureType": int(order_msg["signatureType"]),
        }
        return payload

    async def post_order(
        self,
        order: Dict[str, Any],
        *,
        order_type: str = "GTC",
        post_only: bool = False,
        defer_exec: bool = False,
        neg_risk: bool = False,
    ) -> Dict[str, Any]:
        """
        POST /order. Requires L2 headers + signed order payload.

        Polymarket supports a post-only flag for GTC/GTD (rejected if order would cross). (See docs.)
        """
        path = "/order"
        body = {
            "order": order,
            "owner": self.api_key,
            "orderType": order_type,
            "deferExec": bool(defer_exec),
        }
        # Some API variants accept postOnly and negRisk as order options.
        if post_only:
            body["postOnly"] = True
        if neg_risk:
            body["negRisk"] = True

        headers = self._l2_headers("POST", path, body)
        headers["Content-Type"] = "application/json"
        url = f"{self.s.POLY_CLOB_BASE}{path}"
        async with self.http.post(url, headers=headers, json=body, timeout=15) as r:
            j = await r.json()
            if r.status != 200:
                raise RuntimeError(f"post_order failed {r.status}: {j}")
            return j


    async def post_orders(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        POST /orders (batch). Maximum 15 orders per request; orders processed in parallel.
        Each element should match the API reference shape:
          { "order": <signedOrder>, "owner": <uuid>, "orderType": "GTC|GTD|FOK|FAK", "deferExec": false }
        """
        if not orders:
            return []
        if len(orders) > int(self.s.MM_BATCH_MAX_ORDERS):
            raise ValueError(f"batch too large: {len(orders)} > {self.s.MM_BATCH_MAX_ORDERS}")
        path = "/orders"
        body = orders
        headers = self._l2_headers("POST", path, body)
        headers["Content-Type"] = "application/json"
        url = f"{self.s.POLY_CLOB_BASE}{path}"
        async with self.http.post(url, headers=headers, json=body, timeout=20) as r:
            j = await r.json()
            if r.status != 200:
                raise RuntimeError(f"post_orders failed {r.status}: {j}")
            return list(j)

    async def cancel_orders(self, order_ids: Sequence[str]) -> Dict[str, Any]:
        """
        DELETE /orders (cancel multiple). Body is JSON array of order IDs (max 3000).
        Returns { canceled: [...], not_canceled: {id: reason} }.
        """
        ids = [str(x) for x in order_ids if x]
        if not ids:
            return {"canceled": [], "not_canceled": {}}
        path = "/orders"
        body = ids
        headers = self._l2_headers("DELETE", path, body)
        headers["Content-Type"] = "application/json"
        url = f"{self.s.POLY_CLOB_BASE}{path}"
        async with self.http.request("DELETE", url, headers=headers, json=body, timeout=20) as r:
            j = await r.json()
            if r.status != 200:
                raise RuntimeError(f"cancel_orders failed {r.status}: {j}")
            return j

    async def cancel_all(self) -> Dict[str, Any]:
        """DELETE /cancel-all — cancel all open orders for the user."""
        path = "/cancel-all"
        headers = self._l2_headers("DELETE", path, "")
        url = f"{self.s.POLY_CLOB_BASE}{path}"
        async with self.http.request("DELETE", url, headers=headers, timeout=20) as r:
            j = await r.json()
            if r.status != 200:
                raise RuntimeError(f"cancel_all failed {r.status}: {j}")
            return j

    async def cancel_market_orders(self, *, market: Optional[str] = None, token_id: Optional[str] = None) -> Dict[str, Any]:
        """DELETE /cancel-market-orders — cancel orders by market and optionally token."""
        path = "/cancel-market-orders"
        params: Dict[str, str] = {}
        if market:
            params["market"] = str(market)
        if token_id:
            params["asset_id"] = str(token_id)
        headers = self._l2_headers("DELETE", path, "")
        url = f"{self.s.POLY_CLOB_BASE}{path}"
        async with self.http.request("DELETE", url, headers=headers, params=params, timeout=20) as r:
            j = await r.json()
            if r.status != 200:
                raise RuntimeError(f"cancel_market_orders failed {r.status}: {j}")
            return j

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        DELETE /order with JSON body {"orderID": "..."}.
        """
        path = "/order"
        body = {"orderID": str(order_id)}
        headers = self._l2_headers("DELETE", path, body)
        headers["Content-Type"] = "application/json"
        url = f"{self.s.POLY_CLOB_BASE}{path}"
        async with self.http.request("DELETE", url, headers=headers, json=body, timeout=15) as r:
            j = await r.json()
            if r.status != 200:
                raise RuntimeError(f"cancel_order failed {r.status}: {j}")
            return j

    async def get_open_orders(self, *, market: Optional[str] = None, token_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """
        GET /orders (open orders for user). Use for reconciliation in LIVE mode.
        """
        path = "/orders"
        params: Dict[str, str] = {"limit": str(limit)}
        if market:
            params["market"] = market
        if token_id:
            params["token_id"] = token_id
        headers = self._l2_headers("GET", path, "")
        url = f"{self.s.POLY_CLOB_BASE}{path}"
        async with self.http.get(url, headers=headers, params=params, timeout=15) as r:
            j = await r.json()
            if r.status != 200:
                raise RuntimeError(f"get_open_orders failed {r.status}: {j}")
            return j

    async def post_heartbeat(self, heartbeat_id: str = "") -> Dict[str, Any]:
        """
        POST /heartbeat — keep resting orders alive. If heartbeats stop, open orders may be canceled.
        """
        path = "/heartbeat"
        body = {"heartbeat_id": heartbeat_id or ""}
        headers = self._l2_headers("POST", path, body)
        headers["Content-Type"] = "application/json"
        url = f"{self.s.POLY_CLOB_BASE}{path}"
        async with self.http.post(url, headers=headers, json=body, timeout=10) as r:
            j = await r.json()
            if r.status != 200:
                raise RuntimeError(f"heartbeat failed {r.status}: {j}")
            return j
