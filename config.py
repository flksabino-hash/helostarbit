"""
config.py — central configuration and .env loading.

Safe defaults:
- LIVE_MODE is False (paper trading only).
- Requires explicit LIVE_MODE=True AND POLY_PRIVATE_KEY set to trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Sequence

from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else int(default)
    except Exception:
        return int(default)


@dataclass(slots=True)
class Settings:
    # --- Mode ---
    LIVE_MODE: bool = False                # must be explicitly enabled
    LIVE_ARM_REQUIRED: bool = True         # require runtime arming ceremony (prevents .env tampering only)
    LIVE_ARM_TIMEOUT_SEC: int = 60         # seconds to enter arming code
    POLYGON_CHAIN_ID: int = 137            # validate RPC chain id in LIVE mode
    RPC_REQUIRE_HTTPS: bool = True         # refuse http:// RPC in LIVE mode

    PAPER_START_USDC: float = 1000.0
    LOG_LEVEL: str = "INFO"

    # Backtest diagnostics
    BACKTEST_SYNTH_N: int = 5000
    BACKTEST_DEBUG_GATES: bool = True
    BACKTEST_RUN_SANITY: bool = True
    BACKTEST_METRIC_STRIDE: int = 5
    BACKTEST_FORCE_CLOSE_EOD: bool = True
    BACKTEST_RUN_ABLATION: bool = False
    BACKTEST_SIGNAL_THRESHOLD: float = 0.5
    BACKTEST_AUTO_CALIBRATE: bool = False
    BACKTEST_CALIB_HURST_Q: float = 0.60
    BACKTEST_CALIB_ENTROPY_Q: float = 0.80
    BACKTEST_CALIB_LLE_Q: float = 0.80
    BACKTEST_CALIB_ON_SIGNAL_ONLY: bool = True
    BACKTEST_CALIB_MIN_SAMPLES: int = 300
    BACKTEST_PRECOMPUTE_METRICS: bool = True
    BACKTEST_RUN_WALKFORWARD: bool = False
    BACKTEST_WF_CALIB_FRAC: float = 0.40
    BACKTEST_SAVE_REPORT: str = ""
    BACKTEST_DEBUG_INDICATOR_STATS: bool = True
    BACKTEST_DEBUG_SAMPLE_MAX: int = 50000

    # --- Binance feed ---
    BINANCE_SYMBOLS: Sequence[str] = field(default_factory=lambda: ("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    BINANCE_WS_URL: str = "wss://stream.binance.com:9443/stream"
    BINANCE_REST_URL: str = "https://api.binance.com"
    HISTORY_MAXLEN: int = 2400             # ~40 minutes at 1s ticks

    # --- Polymarket endpoints ---
    POLY_GAMMA_BASE: str = "https://gamma-api.polymarket.com"
    POLY_CLOB_BASE: str = "https://clob.polymarket.com"

    # WebSocket endpoints (market/user channels)
    POLY_WSS_MARKET: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    POLY_WSS_USER: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


    # --- Polymarket chain (Polygon mainnet) ---
    POLY_CHAIN_ID: int = 137
    POLY_CTF_EXCHANGE: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    POLY_NEG_RISK_EXCHANGE: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    POLY_USDC_E: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    POLY_CTF: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

    # RPC (use a rate-limited public endpoint for dry-run; switch to paid RPC for live)
    POLYGON_RPC_URL: str = "https://polygon-rpc.com"

    # --- Credentials (keep in .env; optionally encrypted at rest) ---
    POLY_PRIVATE_KEY: str = ""             # required for L1 auth + order signing
    POLY_FUNDER_ADDRESS: str = ""          # if empty, derived from private key
    POLY_SIGNATURE_TYPE: int = 0           # 0=EOA (see docs)

    # Optional: store derived API creds instead of deriving each run
    POLY_API_KEY: str = ""
    POLY_API_SECRET: str = ""              # base64url string
    POLY_API_PASSPHRASE: str = ""

    # --- Strategy core knobs ---
    MIN_EDGE: float = 0.05                 # 5% post-cost edge
    MIN_TTE_SEC: int = 300                 # 5 minutes
    MAX_TTE_SEC: int = 900                 # 15 minutes
    MAX_POSITIONS: int = 3
    COOLDOWN_SEC: int = 180                # per-market cooldown

    # --- OpenClaw-style Market Making (microstructure loop) ---
    MARKET_MAKER_MODE: bool = False        # enable maker/post-only quoting loop
    MM_USE_WSS: bool = True               # use Polymarket market websocket for books
    MM_MAX_ASSETS: int = 3
    MM_USE_BATCH: bool = True              # use /orders batch endpoints for cancel/replace
    MM_BATCH_MAX_ORDERS: int = 15          # Polymarket batch limit
    MM_MAX_TOKEN_IDS: int = 10           # cap WS subscriptions
    MM_REFRESH_SEC: float = 2.0           # cancel/replace cadence
    MM_ORDER_TTL_SEC: int = 25            # cancel if older than this
    MM_POST_ONLY: bool = True             # maker only (reject if crosses)
    MM_ORDER_TYPE: str = "GTC"            # GTC or GTD
    MM_GTD_LIFETIME_SEC: int = 180        # if using GTD, effective lifetime (plus 60s buffer per docs)

    # Quoting parameters (prices are in probability dollars 0..1)
    MM_HALF_SPREAD_BPS: float = 25.0      # half-spread around fair, in bps of price (25 bps = 0.25%)
    MM_MIN_HALF_SPREAD_TICKS: int = 1     # enforce >= N ticks from center
    MM_ALPHA_IMB_K: float = 0.15          # orderbook imbalance -> price shift scale
    MM_ALPHA_MICRO_K: float = 0.75        # microprice-mid -> price shift scale
    MM_DEPTH_LEVELS: int = 10             # levels used for imbalance

    # Inventory & risk constraints
    MM_BASE_SIZE_SHARES: float = 40.0     # base quote size per side, in shares
    MM_MAX_USDC_PER_ORDER: float = 50.0   # notional cap per order
    MM_MAX_INV_USDC: float = 200.0        # soft inventory cap (value at mid)
    MM_INV_SKEW_BPS: float = 75.0         # inventory skew (bps shift on center per full cap)
    MM_INV_SIZE_SKEW: float = 0.8         # size skew factor

    # Live-only: heartbeat maintenance for resting orders
    MM_HEARTBEAT_SEC: float = 5.0         # send heartbeat every ~5s

    # --- Fees/costs (conservative; tune in backtest) ---
    # CLOB fee rates can be fetched; these are fallbacks.
    MAKER_FEE_BPS: int = 0
    TAKER_FEE_BPS: int = 30
    GAS_USDC_ESTIMATE: float = 0.10        # "all-in" per trade, conservative

    # --- Risk management ---
    RISK_PER_TRADE: float = 0.01           # 1% equity
    DRAWDOWN_CAP: float = 0.10             # 10% max peak-to-trough; stop trading
    MAX_USDC_PER_TRADE: float = 50.0
    SLIPPAGE_BPS_EST: float = 10.0         # assumed bps impact in live
    LEVY_ALPHA_MIN: float = 1.2            # if estimated alpha < this, reduce risk sharply

    # --- Econophysics flags ---
    USE_ECONO: bool = True
    USE_HURST: bool = True
    USE_ENTROPY: bool = True
    USE_LLE: bool = True
    USE_TRANSFER_ENTROPY: bool = False     # heavier compute; enable once stable
    USE_MFDFA_PROXY: bool = True

    HURST_MIN: float = 0.55                # trend persistence gate
    ENTROPY_MAX: float = 2.8               # lower = more stable (depends on bins)
    LLE_MAX: float = -0.001                # require <= (negative / small) to avoid chaos
    CONTAGION_TE_MAX: float = 0.15         # if TE above, scale risk down

    # --- ML / symbolic regression ---
    USE_SYMBOLIC_REGRESSION: bool = True
    SR_MAX_TERMS: int = 4
    SR_LOOKBACK: int = 120                 # points
    SR_HORIZON_SEC: int = 120              # forecast horizon

    # --- AMH adaptation ---
    AMH_ENABLED: bool = True
    AMH_WINDOW: int = 30                   # last N trades
    AMH_EDGE_MULT_MIN: float = 0.8
    AMH_EDGE_MULT_MAX: float = 1.6

    # --- Backtest ---
    BACKTEST_CSV: str = ""                 # optional path
    BACKTEST_SEED: int = 7


def load_settings(dotenv_path: str | None = ".env") -> Settings:
    if dotenv_path:
        p = Path(dotenv_path)
        if p.exists():
            load_dotenv(p)

    s = Settings()
    s.LIVE_MODE = _env_bool("LIVE_MODE", s.LIVE_MODE)
    s.LIVE_ARM_REQUIRED = _env_bool("LIVE_ARM_REQUIRED", s.LIVE_ARM_REQUIRED)
    s.LIVE_ARM_TIMEOUT_SEC = _env_int("LIVE_ARM_TIMEOUT_SEC", s.LIVE_ARM_TIMEOUT_SEC)
    s.POLYGON_CHAIN_ID = _env_int("POLYGON_CHAIN_ID", s.POLYGON_CHAIN_ID)
    s.RPC_REQUIRE_HTTPS = _env_bool("RPC_REQUIRE_HTTPS", s.RPC_REQUIRE_HTTPS)
    s.PAPER_START_USDC = _env_float("PAPER_START_USDC", s.PAPER_START_USDC)
    s.LOG_LEVEL = os.getenv("LOG_LEVEL", s.LOG_LEVEL)

    s.POLYGON_RPC_URL = os.getenv("POLYGON_RPC_URL", s.POLYGON_RPC_URL)
    s.POLY_WSS_MARKET = os.getenv("POLY_WSS_MARKET", s.POLY_WSS_MARKET)
    s.POLY_WSS_USER = os.getenv("POLY_WSS_USER", s.POLY_WSS_USER)

    s.POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", s.POLY_PRIVATE_KEY).strip()
    s.POLY_FUNDER_ADDRESS = os.getenv("POLY_FUNDER_ADDRESS", s.POLY_FUNDER_ADDRESS).strip()
    s.POLY_SIGNATURE_TYPE = _env_int("POLY_SIGNATURE_TYPE", s.POLY_SIGNATURE_TYPE)

    s.POLY_API_KEY = os.getenv("POLY_API_KEY", s.POLY_API_KEY).strip()
    s.POLY_API_SECRET = os.getenv("POLY_API_SECRET", s.POLY_API_SECRET).strip()
    s.POLY_API_PASSPHRASE = os.getenv("POLY_API_PASSPHRASE", s.POLY_API_PASSPHRASE).strip()

    s.MIN_EDGE = _env_float("MIN_EDGE", s.MIN_EDGE)
    s.MIN_TTE_SEC = _env_int("MIN_TTE_SEC", s.MIN_TTE_SEC)
    s.MAX_TTE_SEC = _env_int("MAX_TTE_SEC", s.MAX_TTE_SEC)
    s.MAX_POSITIONS = _env_int("MAX_POSITIONS", s.MAX_POSITIONS)
    s.COOLDOWN_SEC = _env_int("COOLDOWN_SEC", s.COOLDOWN_SEC)

    s.MARKET_MAKER_MODE = _env_bool("MARKET_MAKER_MODE", s.MARKET_MAKER_MODE)
    s.MM_USE_WSS = _env_bool("MM_USE_WSS", s.MM_USE_WSS)
    s.MM_USE_BATCH = _env_bool("MM_USE_BATCH", s.MM_USE_BATCH)
    s.MM_BATCH_MAX_ORDERS = _env_int("MM_BATCH_MAX_ORDERS", s.MM_BATCH_MAX_ORDERS)
    s.MM_MAX_ASSETS = _env_int("MM_MAX_ASSETS", s.MM_MAX_ASSETS)
    s.MM_REFRESH_SEC = _env_float("MM_REFRESH_SEC", s.MM_REFRESH_SEC)
    s.MM_ORDER_TTL_SEC = _env_int("MM_ORDER_TTL_SEC", s.MM_ORDER_TTL_SEC)
    s.MM_POST_ONLY = _env_bool("MM_POST_ONLY", s.MM_POST_ONLY)
    s.MM_ORDER_TYPE = os.getenv("MM_ORDER_TYPE", s.MM_ORDER_TYPE)
    s.MM_GTD_LIFETIME_SEC = _env_int("MM_GTD_LIFETIME_SEC", s.MM_GTD_LIFETIME_SEC)
    s.MM_HALF_SPREAD_BPS = _env_float("MM_HALF_SPREAD_BPS", s.MM_HALF_SPREAD_BPS)
    s.MM_MIN_HALF_SPREAD_TICKS = _env_int("MM_MIN_HALF_SPREAD_TICKS", s.MM_MIN_HALF_SPREAD_TICKS)
    s.MM_ALPHA_IMB_K = _env_float("MM_ALPHA_IMB_K", s.MM_ALPHA_IMB_K)
    s.MM_ALPHA_MICRO_K = _env_float("MM_ALPHA_MICRO_K", s.MM_ALPHA_MICRO_K)
    s.MM_DEPTH_LEVELS = _env_int("MM_DEPTH_LEVELS", s.MM_DEPTH_LEVELS)
    s.MM_BASE_SIZE_SHARES = _env_float("MM_BASE_SIZE_SHARES", s.MM_BASE_SIZE_SHARES)
    s.MM_MAX_USDC_PER_ORDER = _env_float("MM_MAX_USDC_PER_ORDER", s.MM_MAX_USDC_PER_ORDER)
    s.MM_MAX_INV_USDC = _env_float("MM_MAX_INV_USDC", s.MM_MAX_INV_USDC)
    s.MM_INV_SKEW_BPS = _env_float("MM_INV_SKEW_BPS", s.MM_INV_SKEW_BPS)
    s.MM_INV_SIZE_SKEW = _env_float("MM_INV_SIZE_SKEW", s.MM_INV_SIZE_SKEW)
    s.MM_HEARTBEAT_SEC = _env_float("MM_HEARTBEAT_SEC", s.MM_HEARTBEAT_SEC)


    s.MAKER_FEE_BPS = _env_int("MAKER_FEE_BPS", s.MAKER_FEE_BPS)
    s.TAKER_FEE_BPS = _env_int("TAKER_FEE_BPS", s.TAKER_FEE_BPS)
    s.GAS_USDC_ESTIMATE = _env_float("GAS_USDC_ESTIMATE", s.GAS_USDC_ESTIMATE)

    s.RISK_PER_TRADE = _env_float("RISK_PER_TRADE", s.RISK_PER_TRADE)
    s.DRAWDOWN_CAP = _env_float("DRAWDOWN_CAP", s.DRAWDOWN_CAP)
    s.MAX_USDC_PER_TRADE = _env_float("MAX_USDC_PER_TRADE", s.MAX_USDC_PER_TRADE)
    s.SLIPPAGE_BPS_EST = _env_float("SLIPPAGE_BPS_EST", s.SLIPPAGE_BPS_EST)
    s.LEVY_ALPHA_MIN = _env_float("LEVY_ALPHA_MIN", s.LEVY_ALPHA_MIN)

    s.USE_ECONO = _env_bool("USE_ECONO", s.USE_ECONO)
    s.USE_HURST = _env_bool("USE_HURST", s.USE_HURST)
    s.USE_ENTROPY = _env_bool("USE_ENTROPY", s.USE_ENTROPY)
    s.USE_LLE = _env_bool("USE_LLE", s.USE_LLE)
    s.USE_TRANSFER_ENTROPY = _env_bool("USE_TRANSFER_ENTROPY", s.USE_TRANSFER_ENTROPY)
    s.USE_MFDFA_PROXY = _env_bool("USE_MFDFA_PROXY", s.USE_MFDFA_PROXY)

    s.HURST_MIN = _env_float("HURST_MIN", s.HURST_MIN)
    s.ENTROPY_MAX = _env_float("ENTROPY_MAX", s.ENTROPY_MAX)
    s.LLE_MAX = _env_float("LLE_MAX", s.LLE_MAX)
    s.CONTAGION_TE_MAX = _env_float("CONTAGION_TE_MAX", s.CONTAGION_TE_MAX)

    s.USE_SYMBOLIC_REGRESSION = _env_bool("USE_SYMBOLIC_REGRESSION", s.USE_SYMBOLIC_REGRESSION)
    s.SR_MAX_TERMS = _env_int("SR_MAX_TERMS", s.SR_MAX_TERMS)
    s.SR_LOOKBACK = _env_int("SR_LOOKBACK", s.SR_LOOKBACK)
    s.SR_HORIZON_SEC = _env_int("SR_HORIZON_SEC", s.SR_HORIZON_SEC)

    s.AMH_ENABLED = _env_bool("AMH_ENABLED", s.AMH_ENABLED)
    s.AMH_WINDOW = _env_int("AMH_WINDOW", s.AMH_WINDOW)
    s.AMH_EDGE_MULT_MIN = _env_float("AMH_EDGE_MULT_MIN", s.AMH_EDGE_MULT_MIN)
    s.AMH_EDGE_MULT_MAX = _env_float("AMH_EDGE_MULT_MAX", s.AMH_EDGE_MULT_MAX)

    s.BACKTEST_CSV = os.getenv("BACKTEST_CSV", s.BACKTEST_CSV).strip()
    s.BACKTEST_SEED = _env_int("BACKTEST_SEED", s.BACKTEST_SEED)
    s.BACKTEST_SYNTH_N = _env_int("BACKTEST_SYNTH_N", s.BACKTEST_SYNTH_N)
    s.BACKTEST_DEBUG_GATES = _env_bool("BACKTEST_DEBUG_GATES", s.BACKTEST_DEBUG_GATES)
    s.BACKTEST_RUN_SANITY = _env_bool("BACKTEST_RUN_SANITY", s.BACKTEST_RUN_SANITY)
    s.BACKTEST_METRIC_STRIDE = _env_int("BACKTEST_METRIC_STRIDE", s.BACKTEST_METRIC_STRIDE)
    s.BACKTEST_FORCE_CLOSE_EOD = _env_bool("BACKTEST_FORCE_CLOSE_EOD", s.BACKTEST_FORCE_CLOSE_EOD)
    s.BACKTEST_RUN_ABLATION = _env_bool("BACKTEST_RUN_ABLATION", s.BACKTEST_RUN_ABLATION)
    s.BACKTEST_SIGNAL_THRESHOLD = _env_float("BACKTEST_SIGNAL_THRESHOLD", s.BACKTEST_SIGNAL_THRESHOLD)
    s.BACKTEST_AUTO_CALIBRATE = _env_bool("BACKTEST_AUTO_CALIBRATE", s.BACKTEST_AUTO_CALIBRATE)
    s.BACKTEST_CALIB_HURST_Q = _env_float("BACKTEST_CALIB_HURST_Q", s.BACKTEST_CALIB_HURST_Q)
    s.BACKTEST_CALIB_ENTROPY_Q = _env_float("BACKTEST_CALIB_ENTROPY_Q", s.BACKTEST_CALIB_ENTROPY_Q)
    s.BACKTEST_CALIB_LLE_Q = _env_float("BACKTEST_CALIB_LLE_Q", s.BACKTEST_CALIB_LLE_Q)
    s.BACKTEST_CALIB_ON_SIGNAL_ONLY = _env_bool("BACKTEST_CALIB_ON_SIGNAL_ONLY", s.BACKTEST_CALIB_ON_SIGNAL_ONLY)
    s.BACKTEST_CALIB_MIN_SAMPLES = _env_int("BACKTEST_CALIB_MIN_SAMPLES", s.BACKTEST_CALIB_MIN_SAMPLES)
    s.BACKTEST_PRECOMPUTE_METRICS = _env_bool("BACKTEST_PRECOMPUTE_METRICS", s.BACKTEST_PRECOMPUTE_METRICS)
    s.BACKTEST_RUN_WALKFORWARD = _env_bool("BACKTEST_RUN_WALKFORWARD", s.BACKTEST_RUN_WALKFORWARD)
    s.BACKTEST_WF_CALIB_FRAC = _env_float("BACKTEST_WF_CALIB_FRAC", s.BACKTEST_WF_CALIB_FRAC)
    s.BACKTEST_SAVE_REPORT = os.getenv("BACKTEST_SAVE_REPORT", s.BACKTEST_SAVE_REPORT)
    s.BACKTEST_DEBUG_INDICATOR_STATS = _env_bool("BACKTEST_DEBUG_INDICATOR_STATS", s.BACKTEST_DEBUG_INDICATOR_STATS)
    s.BACKTEST_DEBUG_SAMPLE_MAX = _env_int("BACKTEST_DEBUG_SAMPLE_MAX", s.BACKTEST_DEBUG_SAMPLE_MAX)

    return s
