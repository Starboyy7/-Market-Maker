"""
S&P 500 RSI(4) Multi-Timeframe Scanner
Detects overbought/oversold conditions and bullish/bearish divergences.

Data sources (priority order):
  1. yfinance        — Yahoo Finance, no API key needed
  2. Alpha Vantage   — set env ALPHAVANTAGE_API_KEY
  3. --demo          — synthetic data, no internet required

Watch mode refresh schedule (--watch):
  5min  → every 15 minutes
  1h    → every 5 hours
  4h    → every 4 hours
"""

import os
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
TIMEFRAMES = {
    "5min": {"interval": "5m",  "period": "5d"},
    "1h":   {"interval": "1h",  "period": "30d"},
    "4h":   {"interval": "1h",  "period": "60d"},   # resampled to 4h
}

# Refresh schedule per timeframe
TF_SCHEDULE = {
    "5min": {"type": "interval", "seconds": 15 * 60},           # every 15 min
    "1h":   {"type": "interval", "seconds": 5 * 60 * 60},       # every 5 hours
    "4h":   {"type": "interval", "seconds": 4 * 60 * 60},       # every 4 hours
}

RSI_PERIOD   = 4   # default (used for display label)
DIV_LOOKBACK = 20

# RSI period per timeframe — entry sensitive, context selective
RSI_PERIODS = {"5min": 4, "1h": 7, "4h": 14}

# Overbought/oversold thresholds per timeframe
OVERBOUGHT = {"5min": 80, "1h": 70, "4h": 65}
OVERSOLD   = {"5min": 20, "1h": 30, "4h": 35}

# Relative volume thresholds — signal requires vol_ratio >= threshold
VOL_THRESHOLDS: dict[str, float] = {
    "NVDA": 1.3, "TSLA": 1.3,
    "AAPL": 1.5, "AMZN": 1.5, "AMD": 1.5,
}
VOL_THRESHOLD_DEFAULT = 1.5

# Slope lookback per timeframe (candles)
SLOPE_LOOKBACK = {"5min": 5, "1h": 8, "4h": 10}

LOG_FILE = Path("signal_log.csv")
LOG_FIELDS = [
    "timestamp", "ticker", "señal", "market_open",
    "rsi_5m", "slope_5m", "vol_5m",
    "rsi_1h", "slope_1h", "vol_1h",
    "rsi_4h", "slope_4h", "vol_4h",
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
    """
    if len(price) < lookback + 5:
        return ""
    p = price.iloc[-lookback:].values
    r = rsi.iloc[-lookback:].values
    ph, pl = _local_extremes(p)
    rh, rl = _local_extremes(r)
    if len(ph) >= 2 and len(rh) >= 2:
        if p[ph[-1]] > p[ph[-2]] and r[rh[-1]] < r[rh[-2]]:
            return "bearish"
    if len(pl) >= 2 and len(rl) >= 2:
        if p[pl[-1]] < p[pl[-2]] and r[rl[-1]] > r[rl[-2]]:
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
# Price slope
# ═════════════════════════════════════════════════════════════════════════════
def calc_slope(close: pd.Series, lookback: int = 5) -> str:
    """
    Net % change over last `lookback` candles.
    Lookback is TF-adaptive: 5 (5min) / 8 (1h) / 10 (4h).
    Returns '--' when market is closed — after-hours data is not meaningful.
    ↑↑ = net positive  (> +0.3%)
    ↓↓ = net negative  (< -0.3%)
    →  = lateral       (within ±0.3%)
    """
    if not _market_open():
        return "--"
    if len(close) < lookback + 1:
        return ""
    c_now  = float(close.iloc[-1])
    c_prev = float(close.iloc[-1 - lookback])
    if c_prev == 0:
        return ""
    chg = (c_now - c_prev) / c_prev
    if abs(chg) < 0.003:
        return "→"
    return "↑↑" if chg > 0 else "↓↓"


# ═════════════════════════════════════════════════════════════════════════════
# Multi-TF alignment signal
# ═════════════════════════════════════════════════════════════════════════════
def calc_alignment(ticker: str, tf_data: dict) -> Text:
    """
    Priority order:
      1. BLOQ    — divergence active in any TF (noise filter)
      2. LONG    — 4H>55 AND 1H>55 AND 5M≤35 AND vol≥threshold
      3. SHORT   — 4H<45 AND 1H<45 AND 5M≥65 AND vol≥threshold
      4. NEUTRAL — contradicting TFs
      5. ESPERAR — no clear setup yet
    """
    # BLOQ: any active divergence blocks the trade
    for info in tf_data.values():
        div  = info.get("divergence", "")
        cond = info.get("condition", "")
        if div:
            sig = resolve_signal(cond, div)
            t = Text()
            if "CONT" in sig:
                t.append("BLOQ CONT", style="bold yellow")
            elif div == "bearish":
                t.append("BLOQ ↓div", style="bold red")
            else:
                t.append("BLOQ ↑div", style="bold green")
            return t

    rsi_5m    = tf_data.get("5min", {}).get("rsi")
    rsi_1h    = tf_data.get("1h",   {}).get("rsi")
    rsi_4h    = tf_data.get("4h",   {}).get("rsi")
    vol_5m    = tf_data.get("5min", {}).get("vol_ratio", 0.0)
    threshold = VOL_THRESHOLDS.get(ticker, VOL_THRESHOLD_DEFAULT)

    if None in (rsi_5m, rsi_1h, rsi_4h):
        return Text("—", style="dim")

    vol_ok = vol_5m is not None and vol_5m >= threshold
    t = Text()
    if rsi_4h > 55 and rsi_1h > 55 and rsi_5m <= 35 and vol_ok:
        if _market_open():
            t.append("LONG", style="bold green")
        else:
            t.append("ESPERAR", style="dim")
    elif rsi_4h < 45 and rsi_1h < 45 and rsi_5m >= 65 and vol_ok:
        if _market_open():
            t.append("SHORT", style="bold red")
        else:
            t.append("ESPERAR", style="dim")
    elif max(rsi_4h, rsi_1h, rsi_5m) > 55 and min(rsi_4h, rsi_1h, rsi_5m) < 45:
        t.append("NEUTRAL", style="dim white")
    else:
        t.append("ESPERAR", style="dim")
    return t


# ═════════════════════════════════════════════════════════════════════════════
# Schedule helpers
# ═════════════════════════════════════════════════════════════════════════════
def _market_open() -> bool:
    """True if current ET time is within regular session (Mon–Fri 09:30–16:00)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    hm = (now.hour, now.minute)
    return (9, 30) <= hm < (16, 0)


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
def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}
    ).dropna()


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


def fetch_alphavantage(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    try:
        import requests
        av_map = {
            "5m": ("TIME_SERIES_INTRADAY", "5min"),
            "1h": ("TIME_SERIES_INTRADAY", "60min"),
            "1d": ("TIME_SERIES_DAILY_ADJUSTED", None),
        }
        if interval not in av_map:
            return None
        func, av_interval = av_map[interval]
        params: dict = {"function": func, "symbol": ticker, "apikey": api_key,
                        "outputsize": "full", "datatype": "json"}
        if av_interval:
            params["interval"] = av_interval
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
        data = r.json()
        ts_key = next((k for k in data if "Time Series" in k), None)
        if not ts_key:
            return None
        df = pd.DataFrame(data[ts_key]).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = [c.split(". ")[1].capitalize() for c in df.columns]
        df = df.rename(columns={"Adjusted close": "Close"})
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        days = {"5d": 5, "30d": 30, "60d": 60, "180d": 180}.get(period, 30)
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]
        return df if not df.empty else None
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
    n_bars  = {"5m": 390, "1h": 200, "1d": 130}.get(interval, 150)
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

    freq = {"5m": "5min", "1h": "h", "1d": "D"}.get(interval, "h")
    idx  = pd.date_range(end=datetime.now(), periods=n_bars, freq=freq)

    return pd.DataFrame({
        "Open": open_, "High": close + spread, "Low": close - spread,
        "Close": close, "Volume": volume,
    }, index=idx)


def fetch_ohlcv(ticker: str, interval: str, period: str,
                use_demo: bool = False, seed_offset: int = 0) -> pd.DataFrame | None:
    if use_demo:
        return fetch_demo(ticker, interval, period, seed_offset)
    df = fetch_yfinance(ticker, interval, period)
    if df is not None:
        return df
    return fetch_alphavantage(ticker, interval, period)


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
        if tf_name == "4h":
            df = _resample_4h(df)
        if len(df) < RSI_PERIOD + 10:
            continue
        vol_ratio  = calc_rel_volume(df)
        close      = df["Close"].squeeze()
        rsi_period = RSI_PERIODS.get(tf_name, RSI_PERIOD)
        rsi        = calc_rsi(close, rsi_period)
        last       = float(rsi.iloc[-1])
        if np.isnan(last):
            continue
        ob = OVERBOUGHT.get(tf_name, 80)
        os = OVERSOLD.get(tf_name, 20)
        if last >= ob:
            cond = "overbought"
        elif last <= os:
            cond = "oversold"
        else:
            cond = ""
        results[ticker] = {
            "rsi":       round(last, 1),
            "condition": cond,
            "divergence": detect_divergence(close, rsi) if cond else "",
            "slope":     calc_slope(close, SLOPE_LOOKBACK.get(tf_name, 5)),
            "vol_ratio": vol_ratio,
            "price":     round(float(close.iloc[-1]), 4),
        }
    return results


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
    slope     = info.get("slope", "")
    vol_ratio = info.get("vol_ratio", 0.0)
    cell      = Text()
    if cond:
        signal = resolve_signal(cond, div)
        color  = "red" if cond == "overbought" else "green"
        icon   = "🔺" if cond == "overbought" else "🔻"
        cell.append(f"RSI {rsi_val} {slope} {icon}", style=f"bold {color}")
        if div == "bullish":
            cell.append("  ↑div", style="bold green")
        elif div == "bearish":
            cell.append("  ↓div", style="bold red")
        cell.append_text(_vol_text(vol_ratio))
        cell.append(f"\n{signal}")
    else:
        cell.append(f"RSI {rsi_val} {slope}", style="dim")
        cell.append_text(_vol_text(vol_ratio))
    return cell


_SIGNAL_CLEAN = {
    "LONG":       "LONG",
    "SHORT":      "SHORT",
    "NEUTRAL":    "NEUTRAL",
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
    d5 = tf_data.get("5min", {})
    d1 = tf_data.get("1h",   {})
    d4 = tf_data.get("4h",   {})
    row = {
        "timestamp":   datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
        "ticker":      ticker,
        "señal":       _clean_signal(signal),
        "market_open": _market_open(),
        "rsi_5m":      d5.get("rsi", ""),  "slope_5m": d5.get("slope", ""),
        "vol_5m":      d5.get("vol_ratio", ""),
        "rsi_1h":      d1.get("rsi", ""),  "slope_1h": d1.get("slope", ""),
        "vol_1h":      d1.get("vol_ratio", ""),
        "rsi_4h":      d4.get("rsi", ""),  "slope_4h": d4.get("slope", ""),
        "vol_4h":      d4.get("vol_ratio", ""),
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
        show_lines=True,
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
        for ticker, tf_data in sorted(scan_data.items()):
            alignment = calc_alignment(ticker, tf_data)
            log_signal(ticker, alignment.plain, tf_data)
            row: list = [ticker, alignment]
            for tf in tf_cols:
                if tf not in tf_data:
                    row.append(Text("·", style="dim"))
                else:
                    row.append(_cell(tf_data[tf]))
            table.add_row(*row)

    return table


def build_legend() -> Text:
    t = Text()
    t.append("Leyenda  ", style="bold")
    t.append("🔺 Sobrecompra 5M≥80 1H≥70 4H≥65  🔻 Sobreventa 5M≤20 1H≤30 4H≤35  ")
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
# One-shot scan (no watch)
# ═════════════════════════════════════════════════════════════════════════════
def run_once(tickers: list[str], use_demo: bool):
    console.print(
        f"\n[bold cyan]Escaneando {len(tickers)} acciones del S&P 500…[/bold cyan]\n"
        f"RSI(4/7/14)  Sobrecompra [bold red]80/70/65[/bold red]  "
        f"Sobreventa [bold green]20/30/35[/bold green]  "
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
    return p.parse_args()


def main():
    args    = parse_args()
    tickers = args.tickers or SP500_TICKERS[:(args.top or DEFAULT_TOP)]
    demo = args.demo or not _yfinance_available()

    if demo and not args.demo:
        console.print(
            "[yellow]⚠  yfinance / Alpha Vantage no disponible — "
            "usando datos sintéticos (--demo).[/yellow]\n"
        )

    if args.watch:
        # Print schedule info before entering live screen
        console.print(
            f"\n[bold cyan]Watch mode — {len(tickers)} acciones[/bold cyan]\n"
            f"  [bold]5min[/bold] → refresco cada [cyan]15 minutos[/cyan]\n"
            f"  [bold]1h[/bold]   → refresco cada [cyan]5 horas[/cyan]\n"
            f"  [bold]4h[/bold]   → refresco cada [cyan]4 horas[/cyan]\n"
        )
        time.sleep(1.5)
        run_watch(tickers, demo)
    else:
        run_once(tickers, demo)


if __name__ == "__main__":
    main()
