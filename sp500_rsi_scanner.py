"""
S&P 500 RSI(4) Multi-Timeframe Scanner
Detects overbought/oversold conditions and bullish/bearish divergences.

Data sources (priority order):
  1. yfinance        — Yahoo Finance, no API key needed
  2. --demo          — synthetic data, no internet required

Watch mode refresh schedule (--watch):
  5min  → every 5 minutes
  15min → every 15 minutes
  1h    → every 1 hour
"""

import bisect
import csv
import argparse
import random
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich import box

warnings.filterwarnings("ignore")

console = Console()
ET = ZoneInfo("America/New_York")

# Default: scan top 50 S&P 500 tickers (used when no --tickers / --top given)
DEFAULT_TOP = 50

# ─── S&P 500 tickers (top 100 by market cap) ─────────────────────────────────
SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "JPM","TSLA","UNH","XOM","V","MA","PG","COST","HD","WMT",
    "NFLX","JNJ","ABBV","BAC","KO","CRM","CVX","MRK","AMD","ORCL",
    "PEP","TMO","ACN","ADBE","MCD","LIN","ABT","DHR","CSCO","WFC",
    "TXN","NKE","PM","AMGN","NEE","RTX","HON","QCOM","IBM","UNP",
    "LOW","SPGI","CAT","GS","MS","AXP","ISRG","INTU","BLK","AMAT",
    "SYK","ELV","MDLZ","ADI","DE","GE","NOW","BKNG","PLD","MMC",
    "TJX","VRTX","CI","CB","SO","DUK","MO","BSX","REGN","ZTS",
    "AON","CME","SHW","MCO","CL","EOG","PANW","PGR","SNPS","CDNS",
    "ITW","ETN","PSA","BDX","HUM","LRCX","APD","GD","ICE","NSC",
]

# ─── Timeframe config ─────────────────────────────────────────────────────────
# Intraday system: 1h = direction, 15min = confirmation, 5min = entry
TIMEFRAMES = {
    "5min":  {"interval": "5m",  "period": "5d"},
    "15min": {"interval": "15m", "period": "5d"},
    "1h":    {"interval": "1h",  "period": "30d"},
}

# Refresh schedule per timeframe
TF_SCHEDULE = {
    "5min":  {"type": "interval", "seconds": 5 * 60},        # every 5 min
    "15min": {"type": "interval", "seconds": 15 * 60},       # every 15 min
    "1h":    {"type": "interval", "seconds": 60 * 60},       # every 1 hour
}

RSI_PERIOD   = 4   # default (used for display label)
DIV_LOOKBACK = 20

# Divergence quality filters — weak/stale divergences are ignored
DIV_MIN_RSI_GAP   = 5.0     # min RSI points between the two extremes
DIV_MIN_PRICE_PCT = 0.0015  # min price gap between extremes (0.15%)
DIV_MAX_AGE       = 6       # most recent extreme within N bars

# RSI period per timeframe — entry sensitive, direction stable
RSI_PERIODS = {"5min": 4, "15min": 7, "1h": 14}

# Overbought/oversold thresholds per timeframe
OVERBOUGHT = {"5min": 80, "15min": 75, "1h": 70}
OVERSOLD   = {"5min": 20, "15min": 25, "1h": 30}

# Direction filters
EMA_PERIOD  = 50   # EMA on 1h closes — macro trend filter

# Relative volume thresholds — signal requires vol_ratio >= threshold
VOL_THRESHOLDS: dict[str, float] = {
    "NVDA": 1.2, "TSLA": 1.2,
    "AAPL": 1.2, "AMZN": 1.2, "AMD": 1.2,
}
VOL_THRESHOLD_DEFAULT = 1.2

# Stop loss base — se ajusta por ATR del ticker en backtest
STOP_LOSS_PCT    = 0.50   # fallback fijo si ATR no disponible
ATR_STOP_MULT    = 1.5    # multiplicador ATR por defecto
# Multiplicador ATR por ticker — los monstruos volátiles necesitan más aire
ATR_STOP_MULT_BY_TICKER: dict[str, float] = {
    "NVDA": 2.0, "TSLA": 2.0, "AMD": 1.8,
}
ATR_STOP_MIN_PCT = 0.20   # stop mínimo (tickers muy tranquilos)
ATR_STOP_MAX_PCT = 1.20   # stop máximo (subido por los multiplicadores altos)
# Trailing stop — se activa tras alcanzar +TRAIL_ACTIVATION_PCT;
# si el precio retrocede TRAIL_STOP_PCT desde el pico, cierra.
# Take profit escalonado: 50% sale en TRAIL_ACTIVATION_PCT, 50% corre con
# trailing hasta TRADE_END (30 min antes del cierre, evita gaps overnight).
TRAIL_ACTIVATION_PCT = 0.40
TRAIL_STOP_PCT       = 0.25
# Filtro VWAP del SPY (Opción A — banda muerta):
# solo bloquea si el SPY está claramente lejos del VWAP. Pegado al VWAP no filtra.
SPY_VWAP_BAND_PCT = 0.15
# Umbral del RSI 1H para confirmar tendencia mayor (más estricto = señal más limpia)
RSI_1H_LONG_MIN  = 60   # antes 55 — solo LONG cuando la tendencia hourly es clara
RSI_1H_SHORT_MAX = 40   # antes 45 — solo SHORT cuando la tendencia hourly es clara
# Régimen de mercado — se filtra usando el rango del SPY en la 1ª hora
REGIME_RANGE_MIN = 0.45   # mejor resultado en backtest: +8.04% ROI vs +6.40% con 0.50%
# Circuit breaker diario: si el ROI acumulado del día llega a este nivel, no más trades
CIRCUIT_BREAKER_PCT = -1.5
# Filtro de earnings — no operar el día del reporte ni 1 día antes
EARNINGS_FILTER = False

# Cache de datos — evita re-descargar en cada backtest
CACHE_DIR = Path("cache")
# El cache se invalida automáticamente después de las 20:00 ET (mercado cerrado)
CACHE_MAX_AGE_HOURS = 4   # durante el día, refresca cada 4h como máximo

# Trading hours filter (ET) — no signals outside this window
TRADE_START = (9, 50)   # ignore first 20 min of session (noise)
TRADE_END   = (15, 30)  # no new entries in last 30 min

LOG_FILE = Path("signal_log.csv")
LOG_FIELDS = [
    "timestamp", "ticker", "señal", "market_open",
    "rsi_5m", "vol_5m",
    "rsi_15m", "vol_15m",
    "rsi_1h", "vol_1h",
    "tendencia_ema", "vwap",
    "precio",
]


# ═════════════════════════════════════════════════════════════════════════════
# RSI
# ═════════════════════════════════════════════════════════════════════════════
def calc_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ═════════════════════════════════════════════════════════════════════════════
# Divergence
# ═════════════════════════════════════════════════════════════════════════════
def _local_extremes(arr: np.ndarray, order: int = 3):
    highs, lows = [], []
    for i in range(order, len(arr) - order):
        w = arr[i - order: i + order + 1]
        if arr[i] == w.max():
            highs.append(i)
        if arr[i] == w.min():
            lows.append(i)
    return highs, lows


def detect_divergence(price: pd.Series, rsi: pd.Series,
                      lookback: int = DIV_LOOKBACK) -> str:
    """
    Bullish : price lower low  + RSI higher low  → BUY setup
    Bearish : price higher high + RSI lower high → SELL setup

    Quality filters (avoid weak/stale divergences):
      - RSI gap between the two extremes must be >= DIV_MIN_RSI_GAP
      - price gap must be >= DIV_MIN_PRICE_PCT
      - the most recent extreme must be within DIV_MAX_AGE bars
    """
    if len(price) < lookback + 5:
        return ""
    p = price.iloc[-lookback:].values
    r = rsi.iloc[-lookback:].values
    ph, pl = _local_extremes(p)
    rh, rl = _local_extremes(r)

    last_bar = len(p) - 1

    if len(ph) >= 2 and len(rh) >= 2:
        price_gap = (p[ph[-1]] - p[ph[-2]]) / p[ph[-2]]
        rsi_gap   = r[rh[-2]] - r[rh[-1]]
        recent    = (last_bar - ph[-1]) <= DIV_MAX_AGE
        if (p[ph[-1]] > p[ph[-2]] and r[rh[-1]] < r[rh[-2]]
                and price_gap >= DIV_MIN_PRICE_PCT
                and rsi_gap   >= DIV_MIN_RSI_GAP
                and recent):
            return "bearish"

    if len(pl) >= 2 and len(rl) >= 2:
        price_gap = (p[pl[-2]] - p[pl[-1]]) / p[pl[-2]]
        rsi_gap   = r[rl[-1]] - r[rl[-2]]
        recent    = (last_bar - pl[-1]) <= DIV_MAX_AGE
        if (p[pl[-1]] < p[pl[-2]] and r[rl[-1]] > r[rl[-2]]
                and price_gap >= DIV_MIN_PRICE_PCT
                and rsi_gap   >= DIV_MIN_RSI_GAP
                and recent):
            return "bullish"
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# Signal resolver
# ═════════════════════════════════════════════════════════════════════════════
def resolve_signal(condition: str, divergence: str) -> str:
    if condition == "oversold"   and divergence == "bullish":  return "🟢 BUY"
    if condition == "overbought" and divergence == "bearish":  return "🔴 SELL"
    if condition == "oversold"   and divergence == "bearish":  return "⚠️  CONT ↓"
    if condition == "overbought" and divergence == "bullish":  return "⚠️  CONT ↑"
    return "—"


# ═════════════════════════════════════════════════════════════════════════════
# Relative volume
# ═════════════════════════════════════════════════════════════════════════════
def calc_rel_volume(df: pd.DataFrame) -> float | None:
    """Current bar volume / average volume of the previous 20 bars.
    Returns None when market is closed — volume data is not meaningful."""
    if not _market_open():
        return None
    if df is None or "Volume" not in df.columns or len(df) < 2:
        return 0.0
    vol = df["Volume"].dropna().astype(float)
    if len(vol) < 2:
        return 0.0
    current  = vol.iloc[-1]
    avg20    = vol.iloc[-21:-1].mean() if len(vol) >= 21 else vol.iloc[:-1].mean()
    if avg20 == 0:
        return 0.0
    return round(current / avg20, 2)


# ═════════════════════════════════════════════════════════════════════════════
# Direction filters: EMA, VWAP, VWMA
# ═════════════════════════════════════════════════════════════════════════════
def calc_above_ema(close: pd.Series, period: int = EMA_PERIOD) -> bool | None:
    """True if last close is above EMA(period). None if not enough data."""
    if len(close) < period:
        return None
    ema = close.ewm(span=period, adjust=False).mean()
    return bool(close.iloc[-1] > ema.iloc[-1])


def calc_above_vwap(df: pd.DataFrame) -> bool | None:
    """True if last close is above today's session VWAP (intraday bars)."""
    if df is None or "Volume" not in df.columns or len(df) < 2:
        return None
    idx = pd.to_datetime(df.index)
    last_day = idx[-1].date()
    day = df[idx.date == last_day]
    vol = day["Volume"].astype(float)
    if len(day) < 2 or vol.sum() == 0:
        return None
    typical = (day["High"] + day["Low"] + day["Close"]) / 3
    vwap = (typical * vol).cumsum() / vol.cumsum()
    return bool(day["Close"].iloc[-1] > vwap.iloc[-1])


# ═════════════════════════════════════════════════════════════════════════════
# Multi-TF alignment signal
# ═════════════════════════════════════════════════════════════════════════════
def calc_alignment(ticker: str, tf_data: dict) -> Text:
    """
    Intraday system — 1h direction, 15min confirmation, 5min entry.

    Priority order:
      1. BLOQ    — divergence contrary to trade direction blocks the trade
      2. LONG    — 1H RSI>RSI_1H_LONG_MIN + precio>EMA50(1H) + precio>VWAP
                   + 15M RSI>50 + 5M RSI 20-35 (reversal) + vol≥threshold
      3. SHORT   — all inverted (5M RSI 65-80)
      4. ESPERAR — no clear setup yet
    """
    # Evaluate preliminary direction from 1H RSI before checking divergences
    d5_pre  = tf_data.get("5min",  {})
    d15_pre = tf_data.get("15min", {})
    d1h_pre = tf_data.get("1h",    {})
    rsi_1h_pre  = d1h_pre.get("rsi", 50)
    rsi_15m_pre = d15_pre.get("rsi", 50)
    rsi_5m_pre  = d5_pre.get("rsi", 50)
    # Tentative direction: LONG bias if 1H bullish, SHORT bias if bearish
    bias = "long" if rsi_1h_pre > RSI_1H_LONG_MIN else "short" if rsi_1h_pre < RSI_1H_SHORT_MAX else ""

    # BLOQ: only block when divergence is CONTRARY to the trade direction
    for info in tf_data.values():
        div  = info.get("divergence", "")
        cond = info.get("condition", "")
        if not div:
            continue
        # bearish divergence blocks a LONG; bullish divergence blocks a SHORT
        contrary = (div == "bearish" and bias == "long") or \
                   (div == "bullish" and bias == "short")
        if contrary:
            t = Text()
            if "CONT" in resolve_signal(cond, div):
                t.append("BLOQ CONT", style="bold yellow")
            elif div == "bearish":
                t.append("BLOQ ↓div", style="bold red")
            else:
                t.append("BLOQ ↑div", style="bold green")
            return t

    d5  = tf_data.get("5min",  {})
    d15 = tf_data.get("15min", {})
    d1h = tf_data.get("1h",    {})

    rsi_5m  = d5.get("rsi")
    rsi_15m = d15.get("rsi")
    rsi_1h  = d1h.get("rsi")
    vol_5m  = d5.get("vol_ratio", 0.0)
    threshold = VOL_THRESHOLDS.get(ticker, VOL_THRESHOLD_DEFAULT)

    if None in (rsi_5m, rsi_15m, rsi_1h):
        return Text("—", style="dim")

    vol_ok    = vol_5m is not None and vol_5m >= threshold
    above_ema  = d1h.get("above_ema")
    above_vwap = d5.get("above_vwap")

    long_dir  = above_ema is True  and above_vwap is True
    short_dir = above_ema is False and above_vwap is False

    # Volume check on the reversal candle (RSI bouncing from extreme, not at extreme)
    in_long_reversal  = 20 <= rsi_5m <= 35
    in_short_reversal = 65 <= rsi_5m <= 80

    t = Text()
    if (rsi_1h > RSI_1H_LONG_MIN and long_dir and rsi_15m > 50
            and in_long_reversal and vol_ok):
        if _market_tradeable():
            t.append("LONG", style="bold green")
        else:
            t.append("ESPERAR", style="dim")
    elif (rsi_1h < RSI_1H_SHORT_MAX and short_dir and rsi_15m < 50
            and in_short_reversal and vol_ok):
        if _market_tradeable():
            t.append("SHORT", style="bold red")
        else:
            t.append("ESPERAR", style="dim")
    else:
        t.append("ESPERAR", style="dim")
    return t


# ═════════════════════════════════════════════════════════════════════════════
# Schedule helpers
# ═════════════════════════════════════════════════════════════════════════════
_FORCE_OPEN = False  # backtest replays session bars as if the market were open


def _market_open() -> bool:
    """True if within regular NYSE session (Mon–Fri 09:30–16:00 ET)."""
    if _FORCE_OPEN:
        return True
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    hm = (now.hour, now.minute)
    return (9, 30) <= hm < (16, 0)


def _market_tradeable() -> bool:
    """True only during the tradeable window (09:50–15:30 ET).
    Excludes the noisy opening 20 min and the illiquid last 30 min."""
    if _FORCE_OPEN:
        return True
    if not _market_open():
        return False
    now = datetime.now(ET)
    hm  = (now.hour, now.minute)
    return TRADE_START <= hm < TRADE_END


def _seconds_until(dt: datetime) -> float:
    now = datetime.now(dt.tzinfo or ET)
    return max(0.0, (dt - now).total_seconds())


def _fmt_countdown(seconds: float) -> str:
    if seconds <= 0:
        return "ahora"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    if s < 86400:
        h, rem = divmod(s, 3600)
        return f"{h}h {rem // 60:02d}m"
    d, rem = divmod(s, 86400)
    return f"{d}d {rem // 3600:02d}h"


# ═════════════════════════════════════════════════════════════════════════════
# Data providers
# ═════════════════════════════════════════════════════════════════════════════
def fetch_yfinance(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, interval=interval, period=period,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception:
        return None


# ─── Demo / synthetic data ────────────────────────────────────────────────────
def _synthetic_price(n: int, start: float, vol: float = 0.01) -> np.ndarray:
    rng   = np.random.default_rng(abs(hash(str(n) + str(start))) % (2**32))
    steps = rng.normal(0, vol, n)
    return start * np.exp(np.cumsum(steps))


def _force_extreme(close: np.ndarray, condition: str) -> np.ndarray:
    close = close.copy()
    factor = 1.012 if condition == "overbought" else 0.988
    for i in range(-RSI_PERIOD * 3, 0):
        close[i] *= factor
    return close


def _inject_divergence(close: np.ndarray, condition: str, div_type: str) -> np.ndarray:
    close = close.copy()
    lb = DIV_LOOKBACK + 5
    if len(close) < lb:
        return close
    i1, i2 = -lb, -(lb // 2)
    if div_type == "bullish" and condition == "oversold":
        close[i1] *= 0.985
        close[i2] *= 0.980
    elif div_type == "bearish" and condition == "overbought":
        close[i1] *= 1.015
        close[i2] *= 1.020
        for j in range(i2 + 1, 0):
            close[j] *= 0.998
    return close


def fetch_demo(ticker: str, interval: str, period: str,
               seed_offset: int = 0) -> pd.DataFrame | None:
    """
    Synthetic OHLCV.  seed_offset lets watch mode produce different data
    on each refresh cycle (simulates market movement).
    """
    n_bars  = {"5m": 390, "15m": 260, "1h": 200, "1d": 130}.get(interval, 150)
    rng     = random.Random(hash(ticker + interval) + seed_offset)
    start_p = rng.uniform(20, 800)
    close   = _synthetic_price(n_bars, start_p)

    scenario = rng.random()
    if scenario < 0.15:
        cond  = "overbought"
        dtype = rng.choice(["bearish", "bullish", None, None])
        close = _force_extreme(close, cond)
        if dtype:
            close = _inject_divergence(close, cond, dtype)
    elif scenario < 0.30:
        cond  = "oversold"
        dtype = rng.choice(["bullish", "bearish", None, None])
        close = _force_extreme(close, cond)
        if dtype:
            close = _inject_divergence(close, cond, dtype)

    spread   = np.abs(np.diff(close, prepend=close[0])) * 0.5 + start_p * 0.002
    open_    = np.roll(close, 1); open_[0] = close[0]
    volume   = np.abs(
        np.random.default_rng(42).normal(1_000_000, 300_000, n_bars)
    ).astype(int)

    freq = {"5m": "5min", "15m": "15min", "1h": "h", "1d": "D"}.get(interval, "h")
    idx  = pd.date_range(end=datetime.now(), periods=n_bars, freq=freq)

    return pd.DataFrame({
        "Open": open_, "High": close + spread, "Low": close - spread,
        "Close": close, "Volume": volume,
    }, index=idx)


# ═════════════════════════════════════════════════════════════════════════════
# Cache de datos en disco
# ═════════════════════════════════════════════════════════════════════════════
def _cache_path(interval: str, period: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{interval}_{period}.pkl"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age_h = (datetime.now() - mtime).total_seconds() / 3600
    if age_h > CACHE_MAX_AGE_HOURS:
        return False
    return True


def _cache_save(path: Path, data: dict[str, pd.DataFrame]) -> None:
    try:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def _cache_load(path: Path) -> dict[str, pd.DataFrame] | None:
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def fetch_batch_cached(tickers: list[str], interval: str, period: str,
                       force_fresh: bool = False) -> dict[str, pd.DataFrame]:
    """Batch download con cache en disco. Reutiliza datos si tienen < CACHE_MAX_AGE_HOURS."""
    path = _cache_path(interval, period)
    if not force_fresh and _cache_valid(path):
        cached = _cache_load(path)
        if cached is not None:
            # Devuelve solo los tickers pedidos que estén en el cache
            result = {t: cached[t] for t in tickers if t in cached}
            if result:
                return result
    batch = fetch_yfinance_batch(tickers, interval, period)
    if batch:
        # Fusiona con cache existente para no perder tickers previos
        existing = _cache_load(path) or {}
        existing.update(batch)
        _cache_save(path, existing)
    return batch


def fetch_ohlcv(ticker: str, interval: str, period: str,
                use_demo: bool = False, seed_offset: int = 0) -> pd.DataFrame | None:
    if use_demo:
        return fetch_demo(ticker, interval, period, seed_offset)
    return fetch_yfinance(ticker, interval, period)


# ═════════════════════════════════════════════════════════════════════════════
# Per-timeframe scanner
# ═════════════════════════════════════════════════════════════════════════════
def fetch_yfinance_batch(tickers: list[str], interval: str,
                         period: str) -> dict[str, pd.DataFrame]:
    """Download all tickers in one yfinance call — much faster than one-by-one."""
    try:
        import yfinance as yf
        df = yf.download(tickers, interval=interval, period=period,
                         auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
        if df is None or df.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                if t in df.columns.get_level_values(0):
                    sub = df[t].dropna(how="all")
                    if not sub.empty:
                        out[t] = sub
        elif len(tickers) == 1:
            out[tickers[0]] = df
        return out
    except Exception:
        return {}


def scan_timeframe(tickers: list[str], tf_name: str,
                   use_demo: bool = False, seed_offset: int = 0) -> dict:
    """Returns {ticker: {rsi, condition, divergence}} for one timeframe."""
    cfg     = TIMEFRAMES[tf_name]
    results = {}

    # Batch download (real data only) — single API call for all tickers
    batch: dict[str, pd.DataFrame] = {}
    if not use_demo:
        batch = fetch_yfinance_batch(tickers, cfg["interval"], cfg["period"])

    for ticker in tickers:
        df = batch.get(ticker)
        if df is None:
            df = fetch_ohlcv(ticker, cfg["interval"], cfg["period"],
                             use_demo, seed_offset)
        if df is None:
            continue
        info = _analyze_df(df, tf_name)
        if info is not None:
            results[ticker] = info
    return results


def _analyze_df(df: pd.DataFrame, tf_name: str) -> dict | None:
    """Compute RSI, condition, divergence and direction filters for one df."""
    if df is None or len(df) < RSI_PERIOD + 10:
        return None
    vol_ratio  = calc_rel_volume(df)
    close      = df["Close"].squeeze()
    rsi_period = RSI_PERIODS.get(tf_name, RSI_PERIOD)
    rsi        = calc_rsi(close, rsi_period)
    last       = float(rsi.iloc[-1])
    if np.isnan(last):
        return None
    ob = OVERBOUGHT.get(tf_name, 80)
    os = OVERSOLD.get(tf_name, 20)
    if last >= ob:
        cond = "overbought"
    elif last <= os:
        cond = "oversold"
    else:
        cond = ""
    info = {
        "rsi":       round(last, 1),
        "condition": cond,
        "divergence": detect_divergence(close, rsi) if cond else "",
        "vol_ratio": vol_ratio,
        "price":     round(float(close.iloc[-1]), 4),
    }
    # Direction filters
    if tf_name == "1h":
        info["above_ema"] = calc_above_ema(close)
    elif tf_name == "5min":
        info["above_vwap"] = calc_above_vwap(df)
    return info


def full_scan(tickers: list[str], use_demo: bool = False,
              seed_offset: int = 0) -> dict:
    """
    Returns {ticker: {tf_name: {rsi, condition, divergence}}} for ALL timeframes.
    Only tickers with at least one extreme condition are included.
    """
    tf_results: dict[str, dict] = {}
    for tf_name in TIMEFRAMES:
        tf_results[tf_name] = scan_timeframe(tickers, tf_name,
                                             use_demo, seed_offset)

    scan_data: dict = {}
    for tf_name, by_ticker in tf_results.items():
        for ticker, info in by_ticker.items():
            scan_data.setdefault(ticker, {})[tf_name] = info

    return scan_data


# ═════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ═════════════════════════════════════════════════════════════════════════════
def _vol_text(vol_ratio: float | None) -> Text:
    t = Text()
    if vol_ratio is None:
        t.append(" VOL:--", style="dim")
        return t
    if vol_ratio <= 0:
        return t
    label = "ALTO" if vol_ratio >= 1.3 else "BAJO"
    color = "cyan" if vol_ratio >= 1.3 else "dim"
    t.append(f" VOL:{vol_ratio}× {label}", style=color)
    return t


def _cell(info: dict) -> Text:
    rsi_val   = info["rsi"]
    cond      = info["condition"]
    div       = info["divergence"]
    vol_ratio = info.get("vol_ratio", 0.0)
    cell      = Text()
    if cond:
        signal = resolve_signal(cond, div)
        color  = "red" if cond == "overbought" else "green"
        icon   = "🔺" if cond == "overbought" else "🔻"
        cell.append(f"RSI {rsi_val} {icon}", style=f"bold {color}")
        if div == "bullish":
            cell.append("  ↑div", style="bold green")
        elif div == "bearish":
            cell.append("  ↓div", style="bold red")
        cell.append_text(_vol_text(vol_ratio))
        if signal != "—":
            cell.append(f"\n{signal}")
    else:
        cell.append(f"RSI {rsi_val}", style="dim")
        cell.append_text(_vol_text(vol_ratio))
    return cell


_SIGNAL_CLEAN = {
    "LONG":       "LONG",
    "SHORT":      "SHORT",
    "ESPERAR":    "ESPERAR",
    "BLOQ CONT":  "BLOQ_CONT",
    "BLOQ ↓div":  "BLOQ_DIV_BAJISTA",
    "BLOQ ↑div":  "BLOQ_DIV_ALCISTA",
}

def _clean_signal(signal: str) -> str:
    for key, val in _SIGNAL_CLEAN.items():
        if key in signal:
            return val
    return signal.strip() or "—"


def log_signal(ticker: str, signal: str, tf_data: dict):
    """Append one row per ticker per render cycle to signal_log.csv."""
    write_header = not LOG_FILE.exists()
    d5  = tf_data.get("5min",  {})
    d15 = tf_data.get("15min", {})
    d1  = tf_data.get("1h",    {})

    def _dir(val, up: str, down: str) -> str:
        if val is True:  return up
        if val is False: return down
        return ""

    row = {
        "timestamp":   datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
        "ticker":      ticker,
        "señal":       _clean_signal(signal),
        "market_open": _market_open(),
        "rsi_5m":      d5.get("rsi", ""),
        "vol_5m":      d5.get("vol_ratio", ""),
        "rsi_15m":     d15.get("rsi", ""),
        "vol_15m":     d15.get("vol_ratio", ""),
        "rsi_1h":      d1.get("rsi", ""),
        "vol_1h":      d1.get("vol_ratio", ""),
        "tendencia_ema": _dir(d1.get("above_ema"),  "ALCISTA", "BAJISTA"),
        "vwap":          _dir(d5.get("above_vwap"), "ARRIBA",  "ABAJO"),
        "precio":      d5.get("price", ""),
    }
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def build_table(scan_data: dict, last_refresh: dict[str, datetime],
                next_refresh: dict[str, datetime],
                status: str = "", demo: bool = False) -> Table:
    tf_cols  = list(TIMEFRAMES.keys())
    now      = datetime.now(ET)
    mode_tag = "  [dim yellow][DEMO][/dim yellow]" if demo else ""

    table = Table(
        title=(
            f"[bold cyan]S&P 500 · RSI(4/7/14) Scanner[/bold cyan]"
            f"{mode_tag}  "
            f"[dim]{now.strftime('%Y-%m-%d  %H:%M:%S ET')}[/dim]"
            + (f"  [dim]{status}[/dim]" if status else "")
        ),
        box=box.SIMPLE_HEAD,
        show_lines=False,
        pad_edge=False,
        expand=True,
    )

    table.add_column("Ticker", style="bold white", width=8)
    table.add_column("SEÑAL", justify="center", width=12)
    for tf in tf_cols:
        last  = last_refresh.get(tf)
        nxt   = next_refresh.get(tf)
        sched = TF_SCHEDULE[tf]

        mins    = sched["seconds"] // 60
        cadence = f"/{mins}min" if mins < 60 else f"/{mins//60}h"

        refresh_info = ""
        if last:
            refresh_info += f"[dim]↺ {last.strftime('%H:%M')}[/dim]"
        if nxt:
            secs_left = _seconds_until(nxt)
            refresh_info += f"  [dim cyan]→ {_fmt_countdown(secs_left)}[/dim cyan]"

        rsi_p  = RSI_PERIODS.get(tf, RSI_PERIOD)
        header = f"[bold]{tf}[/bold] [dim]{cadence} RSI({rsi_p})[/dim]\n{refresh_info}"
        table.add_column(header, justify="center", width=24)

    if not scan_data:
        table.add_row(
            "[dim]—[/dim]", "[dim]—[/dim]",
            *["[dim]sin datos[/dim]"] * len(tf_cols)
        )
    else:
        hidden = 0
        for ticker, tf_data in sorted(scan_data.items()):
            alignment = calc_alignment(ticker, tf_data)

            # Show only tickers with something relevant: active alignment
            # signal, extreme RSI condition, or divergence in any TF.
            interesting = alignment.plain not in ("ESPERAR", "—") or any(
                info.get("condition") or info.get("divergence")
                for info in tf_data.values()
            )
            if not interesting:
                hidden += 1
                continue

            row: list = [ticker, alignment]
            for tf in tf_cols:
                if tf not in tf_data:
                    row.append(Text("·", style="dim"))
                else:
                    row.append(_cell(tf_data[tf]))
            table.add_row(*row)

        if hidden:
            table.add_row(
                Text(f"+{hidden}", style="dim"),
                Text("sin setup", style="dim"),
                *[Text("·", style="dim")] * len(tf_cols),
            )

    return table


def build_legend() -> Text:
    t = Text()
    t.append("Leyenda  ", style="bold")
    t.append("🔺 Sobrecompra 5M≥80 15M≥75 1H≥70  🔻 Sobreventa 5M≤20 15M≤25 1H≤30  ")
    t.append("↑div", style="bold green")
    t.append(" div alcista  ")
    t.append("↓div", style="bold red")
    t.append(" div bajista  ")
    t.append("🟢 BUY", style="bold green")
    t.append(" sob.venta+alcista  ")
    t.append("🔴 SELL", style="bold red")
    t.append(" sob.compra+bajista")
    return t


def build_summary(scan_data: dict, demo: bool = False) -> Text:
    buy_s = sell_s = active = 0
    for td in scan_data.values():
        has_extreme = False
        for info in td.values():
            if info["condition"]:
                has_extreme = True
                sig = resolve_signal(info["condition"], info["divergence"])
                if sig == "🟢 BUY":   buy_s  += 1
                elif sig == "🔴 SELL": sell_s += 1
        if has_extreme:
            active += 1
    t = Text()
    t.append("Monitoreados: ", style="bold")
    t.append(str(len(scan_data)), style="cyan")
    t.append("  Con señal: ", style="bold")
    t.append(str(active), style="cyan")
    t.append("  BUY: ", style="bold")
    t.append(str(buy_s),  style="bold green")
    t.append("  SELL: ", style="bold")
    t.append(str(sell_s), style="bold red")
    if demo:
        t.append("  [DEMO]", style="bold red")
    return t


# ═════════════════════════════════════════════════════════════════════════════
# Watch mode
# ═════════════════════════════════════════════════════════════════════════════
class WatchState:
    """Shared mutable state between the scheduler threads and the render loop."""

    def __init__(self, tickers: list[str], use_demo: bool):
        self.tickers   = tickers
        self.use_demo  = use_demo
        self.lock      = threading.Lock()

        # scan_data[ticker][tf] = {rsi, condition, divergence}
        self.scan_data: dict = {}

        # per-timeframe timestamps
        self.last_refresh: dict[str, datetime] = {}
        self.next_refresh: dict[str, datetime] = {}

        self.status      = "iniciando…"
        self.seed_offset = 0   # incremented on each demo refresh

        # Compute initial "next refresh" for each timeframe
        now = datetime.now(ET)
        for tf_name, sched in TF_SCHEDULE.items():
            self.next_refresh[tf_name] = now + timedelta(seconds=sched["seconds"])

    def refresh_tf(self, tf_name: str):
        with self.lock:
            self.status = f"actualizando {tf_name}…"

        new_tf_data = scan_timeframe(
            self.tickers, tf_name,
            self.use_demo, self.seed_offset
        )

        now = datetime.now(ET)
        sched = TF_SCHEDULE[tf_name]

        with self.lock:
            # Merge: remove old data for this tf, add new
            for ticker in list(self.scan_data.keys()):
                self.scan_data[ticker].pop(tf_name, None)
                if not self.scan_data[ticker]:
                    del self.scan_data[ticker]

            for ticker, info in new_tf_data.items():
                self.scan_data.setdefault(ticker, {})[tf_name] = info

            self.last_refresh[tf_name] = now
            self.next_refresh[tf_name] = now + timedelta(seconds=sched["seconds"])

            self.seed_offset += 1
            self.status = ""
            # Snapshot for logging (outside lock below)
            snapshot = {t: dict(d) for t, d in self.scan_data.items()}

        # Write CSV immediately after scan — don't wait for render
        for ticker, tf_data in snapshot.items():
            alignment = calc_alignment(ticker, tf_data)
            log_signal(ticker, alignment.plain, tf_data)


def _tf_worker(tf_name: str, state: WatchState, stop_event: threading.Event):
    """Background thread: refreshes one timeframe on its schedule."""
    # Initial scan immediately
    state.refresh_tf(tf_name)

    sched = TF_SCHEDULE[tf_name]

    while not stop_event.is_set():
        with state.lock:
            nxt = state.next_refresh.get(tf_name)

        secs = _seconds_until(nxt) if nxt else 60
        # Sleep in small chunks so we can react to stop_event quickly
        chunk = min(secs, 10.0)
        while chunk > 0 and not stop_event.is_set():
            time.sleep(min(chunk, 1.0))
            chunk -= 1.0
            with state.lock:
                nxt = state.next_refresh.get(tf_name)
            remaining = _seconds_until(nxt) if nxt else 0
            if remaining <= 0:
                break

        if stop_event.is_set():
            break

        state.refresh_tf(tf_name)


def run_watch(tickers: list[str], use_demo: bool):
    state      = WatchState(tickers, use_demo)
    stop_event = threading.Event()

    # Launch one background thread per timeframe
    threads = []
    for tf_name in TIMEFRAMES:
        t = threading.Thread(
            target=_tf_worker,
            args=(tf_name, state, stop_event),
            daemon=True,
            name=f"tf-{tf_name}",
        )
        t.start()
        threads.append(t)

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                with state.lock:
                    scan_data    = dict(state.scan_data)
                    last_refresh = dict(state.last_refresh)
                    next_refresh = dict(state.next_refresh)
                    status       = state.status

                table   = build_table(scan_data, last_refresh, next_refresh,
                                      status=status, demo=use_demo)
                legend  = build_legend()
                summary = build_summary(scan_data, demo=use_demo)

                help_text = Text(
                    "\n  Ctrl+C para salir", style="dim"
                )

                live.update(
                    Panel(
                        Columns([table]),
                        subtitle=Text.assemble(legend, "\n", summary, help_text),
                        border_style="dim cyan",
                        padding=(0, 1),
                    )
                )
                time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        console.print("\n[dim]Scanner detenido.[/dim]")


# ═════════════════════════════════════════════════════════════════════════════
# Backtest helpers — ATR stop, regime filter, earnings filter
# ═════════════════════════════════════════════════════════════════════════════

def _atr_stop_pct(df5: pd.DataFrame, as_of_ts, ticker: str = "") -> float:
    """ATR(14) en 5min hasta as_of_ts → stop en % con multiplicador por ticker."""
    pos    = df5.index.searchsorted(as_of_ts, side="right")
    sliced = df5.iloc[max(0, pos - 30):pos]
    if len(sliced) < 5:
        return STOP_LOSS_PCT
    high = sliced["High"]
    low  = sliced["Low"]
    prev_close = sliced["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr  = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    px   = float(sliced["Close"].iloc[-1])
    mult = ATR_STOP_MULT_BY_TICKER.get(ticker, ATR_STOP_MULT)
    pct  = (atr / px * 100) * mult
    return round(max(ATR_STOP_MIN_PCT, min(ATR_STOP_MAX_PCT, pct)), 3)


def _spy_vwap_series(spy_df5: pd.DataFrame, session_date) -> pd.Series:
    """VWAP acumulado del SPY barra a barra para la sesión dada."""
    idx      = pd.to_datetime(spy_df5.index)
    day_bars = spy_df5[idx.date == session_date].copy()
    if day_bars.empty or "Volume" not in day_bars.columns:
        return pd.Series(dtype=float)
    typical          = (day_bars["High"] + day_bars["Low"] + day_bars["Close"]) / 3
    day_bars["_tpv"] = typical * day_bars["Volume"]
    day_bars["_vwap"] = day_bars["_tpv"].cumsum() / day_bars["Volume"].cumsum()
    return day_bars["_vwap"]


def _spy_vwap_dict(spy_df5: pd.DataFrame | None,
                   session_date) -> tuple[dict, list]:
    """Devuelve (dict {ts: (close, vwap)}, sorted_keys) para lookup O(log n)."""
    if spy_df5 is None:
        return {}, []
    series = _spy_vwap_series(spy_df5, session_date)
    if series.empty:
        return {}, []
    closes = spy_df5["Close"]
    result = {}
    for ts, vwap in series.items():
        result[ts] = (float(closes.get(ts, vwap)), float(vwap))
    return result, sorted(result.keys())


def _spy_regime(spy_df5: pd.DataFrame | None, session_date) -> bool:
    """True = día con tendencia (operable). False = día choppy, no operar.
    Criterio: rango de SPY en barras 9:30-10:30 ET >= REGIME_RANGE_MIN %."""
    if spy_df5 is None:
        return True  # sin datos SPY, no filtramos
    idx = pd.to_datetime(spy_df5.index)
    day_bars = spy_df5[(idx.date == session_date) &
                       (idx.hour == 9) | (idx.hour == 10)]
    # Más preciso: barras entre 9:30 y 10:30
    day_bars = spy_df5[
        (pd.to_datetime(spy_df5.index).date == session_date) &
        (pd.to_datetime(spy_df5.index).hour.isin([9, 10]))
    ]
    if len(day_bars) < 4:
        return True
    rng = (day_bars["High"].max() - day_bars["Low"].min())
    open_px = float(day_bars["Open"].iloc[0])
    rng_pct = rng / open_px * 100
    return rng_pct >= REGIME_RANGE_MIN


def _build_earnings_set(tickers: list[str]) -> set[tuple]:
    """Descarga calendario de earnings de yfinance.
    Retorna set de (ticker, date) donde NO se debe operar
    (día del reporte y día anterior)."""
    blocked: set[tuple] = set()
    if not EARNINGS_FILTER:
        return blocked
    for tk in tickers:
        try:
            cal = yf.Ticker(tk).calendar
            if cal is None:
                continue
            # calendar puede ser dict o DataFrame según versión de yfinance
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed is None:
                    continue
                dates = [ed] if not hasattr(ed, "__iter__") else list(ed)
            elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                dates = list(cal["Earnings Date"])
            elif hasattr(cal, "index") and "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"]
                dates = list(val) if hasattr(val, "__iter__") else [val]
            else:
                continue
            for d in dates:
                try:
                    dt = pd.to_datetime(d).date()
                    blocked.add((tk, dt))
                    blocked.add((tk, dt - pd.Timedelta(days=1)))
                except Exception:
                    pass
        except Exception:
            pass
    return blocked


# ═════════════════════════════════════════════════════════════════════════════
# Backtest — replay today's session bar by bar
# ═════════════════════════════════════════════════════════════════════════════
def run_backtest(tickers: list[str], use_demo: bool,
                 all_hours: bool = False, days: int = 1,
                 force_fresh: bool = False):
    global _FORCE_OPEN

    # For multi-day we need more history — use max allowed by yfinance for 5min (60d)
    bt_periods = {
        "5min":  "60d" if days > 5 else "5d",
        "15min": "60d" if days > 5 else "5d",
        "1h":    "60d",
    }
    bt_intervals = {"5min": "5m", "15min": "15m", "1h": "1h"}

    console.print(
        f"\n[bold cyan]Backtest intradía — {len(tickers)} acciones  "
        f"{'último mes' if days >= 20 else f'últimos {days} día(s)'}[/bold cyan]\n"
        f"[dim]Reproduce cada sesión vela a vela (5min).[/dim]\n"
    )

    # Download full data — usa cache si está disponible
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for tf_name in TIMEFRAMES:
        interval = bt_intervals[tf_name]
        period   = bt_periods[tf_name]
        batch: dict[str, pd.DataFrame] = {}
        if not use_demo:
            path = _cache_path(interval, period)
            if not force_fresh and _cache_valid(path):
                console.print(f"[dim]{tf_name} — cargando desde cache…[/dim]")
            else:
                console.print(f"[dim]{tf_name} — descargando…[/dim]")
            batch = fetch_batch_cached(tickers, interval, period, force_fresh)
        for t in tickers:
            df = batch.get(t)
            if df is None:
                df = fetch_ohlcv(t, interval, period, use_demo)
            if df is not None and not df.empty:
                data.setdefault(t, {})[tf_name] = df

    # SPY para filtro de régimen — también cacheado
    spy_df5: pd.DataFrame | None = None
    if not use_demo:
        spy_period = "60d" if days > 5 else "5d"
        spy_path   = _cache_path("5m", spy_period)
        if not force_fresh and _cache_valid(spy_path):
            console.print("[dim]SPY — cargando desde cache…[/dim]")
        else:
            console.print("[dim]SPY — descargando para filtro de régimen…[/dim]")
        spy_batch = fetch_batch_cached(["SPY"], "5m", spy_period, force_fresh)
        _spy = spy_batch.get("SPY")
        if _spy is None or _spy.empty:
            _spy = data.get("SPY", {}).get("5min")
        spy_df5 = _spy

    # Earnings bloqueados — cacheados en disco 24h (cambian poco)
    earnings_blocked: set[tuple] = set()
    if EARNINGS_FILTER and not use_demo:
        earn_cache = CACHE_DIR / "earnings.pkl"
        if not force_fresh and earn_cache.exists():
            age_h = (datetime.now() - datetime.fromtimestamp(
                earn_cache.stat().st_mtime)).total_seconds() / 3600
            if age_h < 24:
                _ec = _cache_load(earn_cache)
                if _ec is not None:
                    earnings_blocked = _ec
        if not earnings_blocked:
            console.print("[dim]Descargando calendario de earnings…[/dim]")
            earnings_blocked = _build_earnings_set(tickers)
            _cache_save(earn_cache, earnings_blocked)
        else:
            console.print("[dim]Earnings — cargando desde cache…[/dim]")

    # Collect all trading days available in the 5min data
    all_days: list = []
    for tfs in data.values():
        df5 = tfs.get("5min")
        if df5 is not None:
            idx = pd.to_datetime(df5.index)
            all_days = sorted(set(idx.date))
            break
    session_days = all_days[-days:] if days < len(all_days) else all_days

    # Per-day results
    day_summary: list[tuple] = []   # (date, wins, losses, bloqs, roi, skipped_regime)
    # Atribución de pérdidas: (date, ticker, sig, roi, motivo_salida)
    loss_records: list[tuple] = []

    _FORCE_OPEN = True
    try:
        for session_date in session_days:
            # ── Rec 1: Filtro de régimen (umbral subido a 0.55%) ─────────────
            if not _spy_regime(spy_df5, session_date):
                day_summary.append((session_date, 0, 0, 0, 0.0, "—", True))
                continue

            # VWAP del SPY precalculado para lookup O(log n) en el loop
            spy_vwap_d, spy_vwap_keys = _spy_vwap_dict(spy_df5, session_date)

            events: list[tuple] = []

            # Precomputa rango de fechas del día — detecta timezone del índice
            _sample_idx = next(
                (tfs["5min"].index for tfs in data.values() if "5min" in tfs), None)
            _tz = getattr(_sample_idx, "tz", None) if _sample_idx is not None else None
            day_start = pd.Timestamp(session_date, tz=_tz)
            day_end   = day_start + pd.Timedelta(days=1)

            for ticker, tfs in sorted(data.items()):
                df5 = tfs.get("5min")
                if df5 is None or len(tfs) < 3:
                    continue
                if (ticker, session_date) in earnings_blocked:
                    continue

                # searchsorted en lugar de máscara booleana para encontrar el día
                i0 = df5.index.searchsorted(day_start, side="left")
                i1 = df5.index.searchsorted(day_end,   side="left")
                session = df5.index[i0:i1]
                if len(session) < 10:
                    continue

                prev_sig = ""
                for ts in session:
                    tf_data: dict = {}
                    for tf_name, df in tfs.items():
                        pos_tf = df.index.searchsorted(ts, side="right")
                        sliced = df.iloc[:pos_tf]
                        info   = _analyze_df(sliced, tf_name)
                        if info is not None:
                            tf_data[tf_name] = info
                    if len(tf_data) < 3:
                        continue
                    sig   = calc_alignment(ticker, tf_data).plain
                    ts_dt = pd.to_datetime(ts)
                    hm_ts = (ts_dt.hour, ts_dt.minute)
                    in_window = all_hours or (TRADE_START <= hm_ts < TRADE_END)
                    if in_window and (
                        (sig != prev_sig and sig in ("LONG", "SHORT")) or
                        (sig.startswith("BLOQ") and not prev_sig.startswith("BLOQ"))
                    ):
                        # ── Filtro VWAP SPY (Opción A — banda muerta) ──
                        # Solo bloquea si el SPY está claramente lejos del VWAP.
                        # Pegado al VWAP (±SPY_VWAP_BAND_PCT) no filtra: día indeciso,
                        # deja que la señal del ticker decida.
                        if sig in ("LONG", "SHORT") and spy_vwap_keys:
                            spy_entry = spy_vwap_d.get(ts)
                            if spy_entry is None:
                                i = bisect.bisect_right(spy_vwap_keys, ts) - 1
                                if i >= 0:
                                    spy_entry = spy_vwap_d[spy_vwap_keys[i]]
                            if spy_entry is not None:
                                spy_close_val, spy_vwap_val = spy_entry
                                # distancia del SPY a su VWAP en %
                                spy_dist = (spy_close_val - spy_vwap_val) / spy_vwap_val * 100
                                # LONG bloqueado solo si SPY claramente bajo VWAP
                                if sig == "LONG" and spy_dist < -SPY_VWAP_BAND_PCT:
                                    prev_sig = sig
                                    continue
                                # SHORT bloqueado solo si SPY claramente sobre VWAP
                                if sig == "SHORT" and spy_dist > SPY_VWAP_BAND_PCT:
                                    prev_sig = sig
                                    continue

                        # Simula el trade barra a barra:
                        #   stop  = ATR(14)×mult por ticker, fill al open siguiente
                        #   TP    = 50% sale en +0.40%
                        #   resto = trailing stop hasta TRADE_END (15:30 ET) — sin
                        #           gaps overnight. roi60 = ROI realizado del trade.
                        roi30 = roi60 = None
                        if sig in ("LONG", "SHORT"):
                            px            = tf_data["5min"]["price"]
                            pos           = df5.index.get_loc(ts)
                            stop_pct      = _atr_stop_pct(df5, ts, ticker)
                            peak          = 0.0
                            exit_roi      = None
                            exit_bar      = None
                            exit_reason   = None  # STOP / TRAIL / FORZADO / TIEMPO
                            half_exit_roi = None  # primera mitad TP en +0.40%

                            for k in range(pos + 1, i1):  # i1 = fin de sesión
                                k_dt = pd.to_datetime(df5.index[k])
                                if k_dt.date() != session_date:
                                    break
                                hi    = float(df5["High"].iloc[k])
                                lo    = float(df5["Low"].iloc[k])
                                cl    = float(df5["Close"].iloc[k])
                                bar_n = k - pos
                                forced_close = (k_dt.hour, k_dt.minute) >= TRADE_END

                                if sig == "LONG":
                                    adverse   = (px - lo) / px * 100
                                    favorable = (hi - px) / px * 100
                                    close_roi = (cl - px) / px * 100
                                else:
                                    adverse   = (hi - px) / px * 100
                                    favorable = (px - lo) / px * 100
                                    close_roi = (px - cl) / px * 100

                                # Stop: si TP ya tomado, solo afecta la mitad restante
                                if adverse >= stop_pct:
                                    if k + 1 < len(df5.index):
                                        fill = float(df5["Open"].iloc[k + 1])
                                        stop_exit = round(
                                            ((fill - px) / px * 100) if sig == "LONG"
                                            else ((px - fill) / px * 100), 2)
                                    else:
                                        stop_exit = -stop_pct
                                    exit_roi = round(
                                        (half_exit_roi + stop_exit) / 2, 2
                                    ) if half_exit_roi is not None else stop_exit
                                    exit_bar = bar_n
                                    exit_reason = "STOP"
                                    break

                                peak = max(peak, favorable)

                                # TP 50% al superar la activación
                                if half_exit_roi is None and close_roi >= TRAIL_ACTIVATION_PCT:
                                    half_exit_roi = TRAIL_ACTIVATION_PCT

                                # Trailing del resto (solo activo si superó activación)
                                if (peak >= TRAIL_ACTIVATION_PCT and
                                        (peak - favorable) >= TRAIL_STOP_PCT):
                                    second = round(close_roi, 2)
                                    exit_roi = round(
                                        (half_exit_roi + second) / 2, 2
                                    ) if half_exit_roi is not None else second
                                    exit_bar = bar_n
                                    exit_reason = "TRAIL"
                                    break

                                # Cierre forzado 30 min antes del cierre de bolsa
                                if forced_close:
                                    second = round(close_roi, 2)
                                    exit_roi = round(
                                        (half_exit_roi + second) / 2, 2
                                    ) if half_exit_roi is not None else second
                                    exit_bar = bar_n
                                    exit_reason = "FORZADO"
                                    break

                                if bar_n == 6:
                                    roi30 = round(close_roi, 2)

                            if exit_roi is not None:
                                if exit_bar is not None and exit_bar <= 6:
                                    roi30 = exit_roi
                                roi60 = exit_roi
                            else:
                                # Sin salida hasta el final de los datos disponibles
                                j = min(pos + 12, len(df5.index) - 1)
                                if pd.to_datetime(df5.index[j]).date() == session_date:
                                    cl = float(df5["Close"].iloc[j])
                                    time_exit = round(
                                        ((cl - px) / px * 100) if sig == "LONG"
                                        else ((px - cl) / px * 100), 2)
                                    roi60 = round(
                                        (half_exit_roi + time_exit) / 2, 2
                                    ) if half_exit_roi is not None else time_exit
                                    exit_reason = "TIEMPO"

                        events.append((
                            ts_dt.strftime("%H:%M"), ticker, sig,
                            tf_data["5min"]["price"],
                            roi30, roi60,
                            tf_data["5min"]["rsi"],
                            tf_data.get("15min", {}).get("rsi", ""),
                            tf_data.get("1h", {}).get("rsi", ""),
                            exit_reason if sig in ("LONG", "SHORT") else "",
                        ))
                    prev_sig = sig

            # Classify events en orden cronológico:
            # circuit breaker -1.5% acumulado → cerrar día
            wins = losses = bloqs = 0
            day_roi      = 0.0
            ev_rows: list[tuple] = []
            circuit_open = True

            for ev in sorted(events):
                hora, tic, sig, px, roi30, roi60, r5, r15, r1h, motivo = ev

                # Circuit breaker activo: no más trades el resto del día
                if sig in ("LONG", "SHORT") and not circuit_open:
                    bloqs += 1
                    continue

                if sig in ("LONG", "SHORT"):
                    roi_eval = roi60 if roi60 is not None else roi30
                    if roi_eval is not None:
                        day_roi += roi_eval
                        if roi_eval >= 0:
                            wins += 1
                        else:
                            losses += 1
                            loss_records.append(
                                (session_date, tic, sig, roi_eval, motivo or "?"))
                        if day_roi <= CIRCUIT_BREAKER_PCT:
                            circuit_open = False
                    ev_rows.append((hora, tic, sig, px,
                                    roi30, roi60, r5, r15, r1h, motivo))
                else:
                    bloqs += 1

            # Print per-day table only if single day; otherwise just summary
            if days == 1:
                _print_day_table(session_date, ev_rows, wins, losses,
                                 bloqs, day_roi)
            else:
                traded = wins + losses
                wr = f"{wins/traded*100:.0f}%" if traded else "—"
                day_summary.append((session_date, wins, losses, bloqs,
                                    round(day_roi, 2), wr, False))
    finally:
        _FORCE_OPEN = False

    if days > 1:
        _print_month_summary(day_summary)
        _print_loss_attribution(day_summary, loss_records)


def _print_day_table(session_date, ev_rows, wins, losses, bloqs, day_roi):
    traded = wins + losses
    wr  = f"{wins/traded*100:.0f}%" if traded else "—"
    avg = f"{day_roi/traded:+.2f}%" if traded else "—"

    table = Table(
        title=f"[bold cyan]Backtest — sesión {session_date}[/bold cyan]",
        box=box.SIMPLE_HEAD,
    )
    for col in ("Hora ET", "Ticker", "Señal", "Entrada",
                "ROI+30m", "ROI+60m", "RSI 5m", "RSI 15m", "RSI 1h", "Salida"):
        table.add_column(col, justify="center")

    def _roi_cell(roi):
        if roi is None:
            return Text("—", style="dim")
        return Text(f"+{roi:.2f}%" if roi >= 0 else f"{roi:.2f}%",
                    style="bold green" if roi >= 0 else "bold red")

    for hora, tic, sig, px, roi30, roi60, r5, r15, r1h, motivo in ev_rows:
        sig_style = ("bold green" if sig == "LONG"
                     else "bold red" if sig == "SHORT" else "yellow")
        table.add_row(hora, tic, Text(sig, style=sig_style),
                      str(px), _roi_cell(roi30), _roi_cell(roi60),
                      str(r5), str(r15), str(r1h), str(motivo or "—"))
    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] [green]{wins+losses} trades[/green]  "
        f"[green]{wins}W[/green] [red]{losses}L[/red]  "
        f"[yellow]{bloqs} BLOQ[/yellow]  "
        f"│  Win rate: [cyan]{wr}[/cyan]  "
        f"ROI@60m acum: [cyan]{day_roi:+.2f}%[/cyan]  "
        f"ROI@60m promedio: [cyan]{avg}[/cyan]\n"
    )
    _trader_advice_single(ev_rows, wins, losses, bloqs, day_roi)


def _print_month_summary(day_summary: list[tuple]):
    table = Table(
        title="[bold cyan]Backtest — resumen mensual[/bold cyan]",
        box=box.SIMPLE_HEAD,
    )
    for col in ("Fecha", "Trades", "W", "L",
                "Win rate", "ROI@60m día", "ROI@60m acum", "Régimen"):
        table.add_column(col, justify="center")

    cum_roi = 0.0
    total_w = total_l = total_b = 0
    choppy_days = 0
    for row in day_summary:
        date, wins, losses, bloqs, day_roi, wr = row[:6]
        choppy = row[6] if len(row) > 6 else False
        cum_roi += day_roi
        total_w += wins
        total_l += losses
        total_b += bloqs
        if choppy:
            choppy_days += 1
        traded    = wins + losses
        roi_style = "bold green" if day_roi >= 0 else "bold red"
        cum_style = "bold green" if cum_roi >= 0 else "bold red"
        regime_label = Text("CHOPPY", style="dim") if choppy else Text("OK", style="dim green")
        table.add_row(
            str(date),
            str(traded),
            str(wins), str(losses),
            wr,
            Text(f"{day_roi:+.2f}%", style=roi_style),
            Text(f"{cum_roi:+.2f}%", style=cum_style),
            regime_label,
        )

    console.print(table)
    total_traded = total_w + total_l
    active_days  = len(day_summary) - choppy_days
    global_wr    = f"{total_w/total_traded*100:.0f}%" if total_traded else "—"
    avg_day      = f"{cum_roi/active_days:+.2f}%" if active_days else "—"
    console.print(
        f"\n[bold]TOTAL:[/bold]  {total_traded} trades  "
        f"[green]{total_w}W[/green] [red]{total_l}L[/red]  "
        f"│  Win rate global: [cyan]{global_wr}[/cyan]  "
        f"ROI acumulado: [cyan]{cum_roi:+.2f}%[/cyan]  "
        f"Promedio/día activo: [cyan]{avg_day}[/cyan]  "
        f"[dim]Días choppy filtrados: {choppy_days}[/dim]\n"
    )
    _trader_advice_monthly(day_summary, total_w, total_l, total_b, cum_roi)


def _print_loss_attribution(day_summary, loss_records):
    """Diagnóstico: ¿de dónde vienen las pérdidas?
    Desglosa los días negativos, el ticker que más sangró y el motivo de
    salida (STOP / TRAIL / FORZADO / TIEMPO) para detectar patrones."""
    if not loss_records:
        return

    # Días negativos peores (los que más arrastran el ROI total)
    neg_days = sorted(
        [(r[0], r[4]) for r in day_summary if len(r) > 4 and r[4] < 0],
        key=lambda x: x[1],
    )
    if neg_days:
        t = Table(
            title="[bold red]Atribución de pérdidas — días negativos[/bold red]",
            box=box.SIMPLE_HEAD,
        )
        for col in ("Fecha", "ROI día", "Trades perdedores",
                    "Tickers que sangraron", "Motivos de salida"):
            t.add_column(col, justify="left")
        for date, roi in neg_days:
            day_losses = [r for r in loss_records if r[0] == date]
            tick_pl: dict[str, float] = {}
            reason_ct: dict[str, int] = {}
            for _, tic, _sig, rl, mot in day_losses:
                tick_pl[tic]   = tick_pl.get(tic, 0.0) + rl
                reason_ct[mot] = reason_ct.get(mot, 0) + 1
            tick_str = ", ".join(
                f"{k} ({v:+.2f}%)"
                for k, v in sorted(tick_pl.items(), key=lambda x: x[1])[:4])
            reason_str = ", ".join(f"{k}×{v}" for k, v in
                                   sorted(reason_ct.items(), key=lambda x: -x[1]))
            t.add_row(str(date),
                      Text(f"{roi:+.2f}%", style="bold red"),
                      str(len(day_losses)), tick_str, reason_str)
        console.print(t)

    # Agregado global: ticker y motivo de salida que más pesan
    tick_total: dict[str, float] = {}
    reason_total: dict[str, list] = {}
    for _, tic, _sig, rl, mot in loss_records:
        tick_total[tic] = tick_total.get(tic, 0.0) + rl
        reason_total.setdefault(mot, [0, 0.0])
        reason_total[mot][0] += 1
        reason_total[mot][1] += rl

    worst_ticks = sorted(tick_total.items(), key=lambda x: x[1])[:6]
    tick_line = "  ".join(f"{k} {v:+.2f}%" for k, v in worst_ticks)
    reason_line = "  ".join(
        f"{k}: {c} trades ({s:+.2f}%)"
        for k, (c, s) in sorted(reason_total.items(), key=lambda x: x[1][1]))
    console.print(
        f"\n[bold]Tickers que más pérdida acumulan:[/bold] [red]{tick_line}[/red]")
    console.print(
        f"[bold]Pérdida por motivo de salida:[/bold] [red]{reason_line}[/red]\n")


# ═════════════════════════════════════════════════════════════════════════════
# Trader algorítmico — análisis experto contextual
# ═════════════════════════════════════════════════════════════════════════════
def _trader_advice_single(ev_rows, wins, losses, bloqs, day_roi):
    traded = wins + losses
    if traded == 0:
        return
    wr = wins / traded

    # e = (hora, ticker, señal, entrada, roi30, roi60, rsi5, rsi15, rsi1h)
    long_rois  = [e[5] for e in ev_rows if e[2] == "LONG"  and e[5] is not None]
    short_rois = [e[5] for e in ev_rows if e[2] == "SHORT" and e[5] is not None]
    long_wr    = sum(1 for r in long_rois  if r > 0) / len(long_rois)  if long_rois  else None
    short_wr   = sum(1 for r in short_rois if r > 0) / len(short_rois) if short_rois else None
    avg_win    = sum(r for r in long_rois + short_rois if r > 0) / max(wins, 1)
    avg_loss   = sum(r for r in long_rois + short_rois if r < 0) / max(losses, 1)
    rr         = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    late_rois  = [e[5] for e in ev_rows
                  if e[5] is not None and e[0] >= "15:00"]
    early_rois = [e[5] for e in ev_rows
                  if e[5] is not None and e[0] < "11:00"]

    lines: list[str] = []

    # Win rate
    if wr >= 0.75:
        lines.append(f"✅  Win rate de {wr*100:.0f}% — sistema alineado con el mercado hoy. "
                     "Mantén la disciplina, no amplíes el tamaño de posición por euforia.")
    elif wr >= 0.55:
        lines.append(f"⚠️  Win rate de {wr*100:.0f}% — aceptable pero mejorable. "
                     "Revisa si las pérdidas se concentran en un horario o en LONGs/SHORTs específicos.")
    else:
        lines.append(f"🔴  Win rate de {wr*100:.0f}% — por debajo del umbral mínimo viable (55%). "
                     "Este día el sistema operó contra la tendencia dominante. No escales capital.")

    # Long vs Short sesgo
    if long_wr is not None and short_wr is not None:
        if long_wr < 0.40 and short_wr > 0.65:
            lines.append("📉  Los LONGs fallaron consistentemente mientras los SHORTs ganaron. "
                         "El mercado tenía sesgo bajista — considera agregar un filtro de tendencia "
                         "macro (SPY/QQQ por encima o debajo de su EMA diaria) antes de permitir LONGs.")
        elif short_wr < 0.40 and long_wr > 0.65:
            lines.append("📈  Los SHORTs fallaron consistentemente mientras los LONGs ganaron. "
                         "Mercado con sesgo alcista fuerte — el filtro de VWAP está subvalorando el momentum.")
        elif long_wr < 0.50 and short_wr < 0.50:
            lines.append("⚠️  Ni LONGs ni SHORTs funcionaron bien hoy — día lateral o volátil sin dirección. "
                         "En este contexto el sistema genera ruido. Considera pausar si SPY cae <0.3% "
                         "y sube <0.3% en las primeras 2 horas.")

    # Risk/Reward
    if rr < 1.0 and losses > 0:
        lines.append(f"🔺  Risk/Reward implícito: {rr:.2f}R — estás ganando menos de lo que pierdes por trade. "
                     "Con stop en 1R y objetivo en 2R este sistema sería rentable incluso con 40% win rate. "
                     "Es la mejora de mayor impacto pendiente.")
    elif rr >= 1.5:
        lines.append(f"✅  R/R implícito de {rr:.2f}R — las ganancias superan las pérdidas. "
                     "Cuando implementes gestión de riesgo, un objetivo de 1.5R–2R es consistente con lo visto.")

    # Señales tardías
    if late_rois and sum(late_rois) < 0:
        lines.append("🕒  Las señales después de las 15:00 ET tuvieron ROI negativo acumulado. "
                     "Con gestión de riesgo real (stop) muchas de estas serían stop out antes del cierre. "
                     "Considera adelantar el TRADE_END a 15:00 y evaluar el impacto.")

    # Señales tempranas
    if early_rois and len(early_rois) >= 2:
        early_wr = sum(1 for r in early_rois if r > 0) / len(early_rois)
        if early_wr < 0.40:
            lines.append("⏰  Las primeras señales del día (antes de las 11:00 ET) tuvieron bajo win rate. "
                         "La liquidez y el price discovery de apertura generan falsos positivos — "
                         "considera retrasar TRADE_START a 10:00 ET.")

    # BLOQs
    if bloqs > traded:
        lines.append(f"🚫  {bloqs} BLOQs vs {traded} trades operados — el detector de divergencias "
                     "sigue siendo conservador. Si los BLOQs corresponden a setups que habrían ganado, "
                     "considera reducir DIV_MIN_RSI_GAP de 5 a 4.")

    _render_advice(lines)


def _trader_advice_monthly(day_summary, total_w, total_l, total_b, cum_roi):
    if not day_summary:
        return
    total_traded  = total_w + total_l
    if total_traded == 0:
        return

    global_wr     = total_w / total_traded
    # day_summary row: (date, wins, losses, bloqs, day_roi, wr, choppy)
    days_positive = sum(1 for r in day_summary if float(r[4]) > 0)
    days_negative = len(day_summary) - days_positive
    avg_day_roi   = cum_roi / len(day_summary)
    worst_day     = min(day_summary, key=lambda x: x[4])
    best_day      = max(day_summary, key=lambda x: x[4])
    consec_losses = 0
    max_consec    = 0
    streak        = 0
    for r in day_summary:
        if float(r[4]) < 0:
            streak += 1
            max_consec = max(max_consec, streak)
        else:
            streak = 0

    lines: list[str] = []

    # ROI acumulado
    if cum_roi > 0:
        lines.append(f"✅  ROI acumulado mensual: {cum_roi:+.2f}% con {global_wr*100:.0f}% win rate global. "
                     f"Sistema rentable en muestra de {len(day_summary)} sesiones — estadísticamente significativo.")
    else:
        lines.append(f"🔴  ROI mensual negativo ({cum_roi:+.2f}%). "
                     "Antes de operar capital real necesitas al menos 3 meses con ROI positivo consistente.")

    # Consistencia diaria
    if days_positive / len(day_summary) >= 0.65:
        lines.append(f"✅  {days_positive}/{len(day_summary)} días positivos — alta consistencia diaria. "
                     "El sistema no depende de un solo día extraordinario.")
    elif days_negative > days_positive:
        lines.append(f"⚠️  Más días negativos ({days_negative}) que positivos ({days_positive}). "
                     "El ROI positivo viene de pocos días con muchos trades ganadores — "
                     "distribución frágil. Investiga qué condiciones macro diferenciaron los días positivos.")

    # Peor día
    if worst_day[4] < -3.0:
        lines.append(f"🔺  Peor día: {worst_day[0]} con {worst_day[4]:+.2f}%. "
                     "Un stop diario de -2% habría limitado ese daño y mejorado el ROI mensual. "
                     "Regla estándar: si el día llega a -2% acumulado, parar.")

    # Racha perdedora
    if max_consec >= 3:
        lines.append(f"⚠️  Racha máxima de {max_consec} días consecutivos negativos. "
                     "Con 3+ días en rojo seguidos el sistema probablemente está operando "
                     "en condiciones de mercado que no son su entorno óptimo (alta volatilidad macro, "
                     "earnings season, etc.). Considera un circuit breaker automático.")

    # Promedio diario
    if avg_day_roi > 0.5:
        lines.append(f"📈  Promedio de {avg_day_roi:+.2f}% por día — con gestión de riesgo "
                     "y compounding conservador (1% de capital por trade) esto proyecta "
                     "retornos mensuales significativos sin riesgo de ruina.")
    elif 0 < avg_day_roi <= 0.5:
        lines.append(f"📊  Promedio de {avg_day_roi:+.2f}% por día — marginal sin gestión de riesgo. "
                     "La implementación de stop loss transformaría esta métrica al cortar pérdidas "
                     "que actualmente arrastran el promedio al final de sesión.")

    _render_advice(lines)


def _render_advice(lines: list[str]):
    if not lines:
        return
    body = Text()
    for i, line in enumerate(lines):
        body.append(line)
        if i < len(lines) - 1:
            body.append("\n\n")
    console.print(
        Panel(
            body,
            title="[bold yellow]⚡ Trader Algorítmico[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


# ═════════════════════════════════════════════════════════════════════════════
# One-shot scan (no watch)
# ═════════════════════════════════════════════════════════════════════════════
def run_once(tickers: list[str], use_demo: bool):
    console.print(
        f"\n[bold cyan]Escaneando {len(tickers)} acciones del S&P 500…[/bold cyan]\n"
        f"RSI(4/7/14)  Sobrecompra [bold red]80/75/70[/bold red]  "
        f"Sobreventa [bold green]20/25/30[/bold green]  "
        f"Timeframes: [italic]{', '.join(TIMEFRAMES)}[/italic]\n"
    )

    total = len(tickers)
    for i, t in enumerate(tickers, 1):
        console.print(f"[dim]({i:>3}/{total}) {t:<8}[/dim]", end="\r")

    scan_data = full_scan(tickers, use_demo)
    console.print(" " * 60, end="\r")

    if not scan_data:
        console.print("[yellow]No hay señales activas en este momento.[/yellow]")
        return

    for ticker, tf_data in scan_data.items():
        log_signal(ticker, calc_alignment(ticker, tf_data).plain, tf_data)

    now = datetime.now(ET)
    empty_ts: dict = {}
    table = build_table(scan_data, empty_ts, empty_ts, demo=use_demo)
    console.print(table)

    console.print()
    console.print(build_legend())
    console.print()
    console.print(build_summary(scan_data, demo=use_demo))
    console.print()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def _yfinance_available() -> bool:
    try:
        import yfinance as yf
        df = yf.download("SPY", interval="1d", period="5d",
                         auto_adjust=True, progress=False)
        return df is not None and not df.empty
    except Exception:
        return False


def parse_args():
    p = argparse.ArgumentParser(
        description="S&P 500 RSI(4) multi-timeframe scanner with divergence detection."
    )
    p.add_argument("--watch",   action="store_true",
                   help="Live mode: auto-refresh each timeframe on its schedule.")
    p.add_argument("--demo",    action="store_true",
                   help="Use synthetic data (no internet required).")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Specific tickers, e.g. --tickers AAPL MSFT NVDA TSLA AMZN")
    p.add_argument("--top",     type=int, default=None,
                   help="Scan first N tickers from the built-in list.")
    p.add_argument("--backtest", action="store_true",
                   help="Replay today's session bar by bar and list the "
                        "LONG/SHORT signals it would have generated.")
    p.add_argument("--all-hours", action="store_true",
                   help="With --backtest: include signals outside 09:50–15:30 window.")
    p.add_argument("--days", type=int, default=1,
                   help="With --backtest: number of sessions to replay (max ~42). "
                        "Use --days 22 for ~1 month.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignora el cache y descarga datos nuevos de yfinance.")
    return p.parse_args()


def main():
    args    = parse_args()
    tickers = args.tickers or SP500_TICKERS[:(args.top or DEFAULT_TOP)]
    demo = args.demo or not _yfinance_available()

    if demo and not args.demo:
        console.print(
            "[yellow]⚠  yfinance no disponible — "
            "usando datos sintéticos (--demo).[/yellow]\n"
        )

    if args.backtest:
        run_backtest(tickers, demo, all_hours=args.all_hours, days=args.days,
                     force_fresh=args.fresh)
    elif args.watch:
        # Print schedule info before entering live screen
        console.print(
            f"\n[bold cyan]Watch mode — {len(tickers)} acciones[/bold cyan]\n"
            f"  [bold]5min[/bold]  → refresco cada [cyan]5 minutos[/cyan]\n"
            f"  [bold]15min[/bold] → refresco cada [cyan]15 minutos[/cyan]\n"
            f"  [bold]1h[/bold]    → refresco cada [cyan]1 hora[/cyan]\n"
        )
        time.sleep(1.5)
        run_watch(tickers, demo)
    else:
        run_once(tickers, demo)


if __name__ == "__main__":
    main()
