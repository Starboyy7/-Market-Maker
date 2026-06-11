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
    "NVDA": 1.3, "TSLA": 1.3,
    "AAPL": 1.5, "AMZN": 1.5, "AMD": 1.5,
}
VOL_THRESHOLD_DEFAULT = 1.5

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
      2. LONG    — 1H RSI>55 + precio>EMA50(1H) + precio>VWAP
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
    bias = "long" if rsi_1h_pre > 55 else "short" if rsi_1h_pre < 45 else ""

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
    if (rsi_1h > 55 and long_dir and rsi_15m > 50
            and in_long_reversal and vol_ok):
        if _market_tradeable():
            t.append("LONG", style="bold green")
        else:
            t.append("ESPERAR", style="dim")
    elif (rsi_1h < 45 and short_dir and rsi_15m < 50
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
# Backtest — replay today's session bar by bar
# ═════════════════════════════════════════════════════════════════════════════
def run_backtest(tickers: list[str], use_demo: bool):
    global _FORCE_OPEN

    console.print(
        f"\n[bold cyan]Backtest intradía — {len(tickers)} acciones[/bold cyan]\n"
        f"[dim]Reproduce la última sesión vela a vela (5min) con la lógica "
        f"actual de señales.[/dim]\n"
    )

    # Download full data once per timeframe
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for tf_name, cfg in TIMEFRAMES.items():
        console.print(f"[dim]Descargando {tf_name}…[/dim]")
        batch: dict[str, pd.DataFrame] = {}
        if not use_demo:
            batch = fetch_yfinance_batch(tickers, cfg["interval"], cfg["period"])
        for t in tickers:
            df = batch.get(t)
            if df is None:
                df = fetch_ohlcv(t, cfg["interval"], cfg["period"], use_demo)
            if df is not None and not df.empty:
                data.setdefault(t, {})[tf_name] = df

    events: list[tuple] = []
    _FORCE_OPEN = True
    try:
        for ticker, tfs in sorted(data.items()):
            df5 = tfs.get("5min")
            if df5 is None or "5min" not in tfs or len(tfs) < 3:
                continue
            idx      = pd.to_datetime(df5.index)
            last_day = idx[-1].date()
            session  = df5.index[idx.date == last_day]

            prev_sig = ""
            for ts in session:
                tf_data: dict = {}
                for tf_name, df in tfs.items():
                    sliced = df[df.index <= ts]
                    info   = _analyze_df(sliced, tf_name)
                    if info is not None:
                        tf_data[tf_name] = info
                if len(tf_data) < 3:
                    continue
                sig = calc_alignment(ticker, tf_data).plain
                # Apply trading-hours filter: only include events 09:50–15:30 ET
                ts_dt = pd.to_datetime(ts)
                hm_ts = (ts_dt.hour, ts_dt.minute)
                in_window = TRADE_START <= hm_ts < TRADE_END
                if in_window and (
                    (sig != prev_sig and sig in ("LONG", "SHORT")) or
                    (sig.startswith("BLOQ") and not prev_sig.startswith("BLOQ"))
                ):
                    events.append((
                        ts_dt.strftime("%H:%M"),
                        ticker, sig,
                        tf_data["5min"]["price"],
                        tf_data["5min"]["rsi"],
                        tf_data.get("15min", {}).get("rsi", ""),
                        tf_data.get("1h", {}).get("rsi", ""),
                    ))
                prev_sig = sig
    finally:
        _FORCE_OPEN = False

    if not events:
        console.print(
            f"\n[yellow]Sin señales LONG/SHORT en la sesión del "
            f"{last_day if data else '—'}.[/yellow]\n"
            f"[dim]La alineación completa (RSI + EMA50 + VWAP + "
            f"volumen) no se dio en ningún momento.[/dim]\n"
        )
        return

    table = Table(
        title=f"[bold cyan]Señales del backtest — sesión {last_day}[/bold cyan]",
        box=box.SIMPLE_HEAD,
    )
    for col in ("Hora ET", "Ticker", "Señal", "Precio",
                "RSI 5m", "RSI 15m", "RSI 1h"):
        table.add_column(col, justify="center")
    for ev in sorted(events):
        hora, tic, sig, px, r5, r15, r1h = ev
        style = ("bold green" if sig == "LONG"
                 else "bold red" if sig == "SHORT" else "yellow")
        table.add_row(hora, tic, Text(sig, style=style),
                      str(px), str(r5), str(r15), str(r1h))
    console.print(table)
    longs  = sum(1 for e in events if e[2] == "LONG")
    shorts = sum(1 for e in events if e[2] == "SHORT")
    bloqs  = len(events) - longs - shorts
    console.print(
        f"\n[bold]Total:[/bold] [green]{longs} LONG[/green]  "
        f"[red]{shorts} SHORT[/red]  [yellow]{bloqs} BLOQ[/yellow]\n"
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
        run_backtest(tickers, demo)
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
