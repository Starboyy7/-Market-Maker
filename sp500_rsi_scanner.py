"""
S&P 500 RSI(4) Multi-Timeframe Scanner
Detects overbought/oversold conditions and bullish/bearish divergences.

Data sources supported (in order of priority):
  1. yfinance  — requires internet access to Yahoo Finance
  2. Alpha Vantage — set env ALPHAVANTAGE_API_KEY
  3. --demo  flag — synthetic data for offline testing / CI environments
"""

import os
import sys
import argparse
import random
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

warnings.filterwarnings("ignore")

console = Console()

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
    "5min": {"interval": "5m",  "period": "5d",   "bars": 390},
    "1h":   {"interval": "1h",  "period": "30d",  "bars": 200},
    "4h":   {"interval": "1h",  "period": "60d",  "bars": 120},  # resampled
    "1D":   {"interval": "1d",  "period": "180d", "bars": 130},
}

RSI_PERIOD   = 4
OVERBOUGHT   = 80
OVERSOLD     = 20
DIV_LOOKBACK = 20


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


def detect_divergence(price: pd.Series, rsi: pd.Series, lookback: int = DIV_LOOKBACK) -> str:
    """
    Bullish  : price lower low + RSI higher low  → BUY setup
    Bearish  : price higher high + RSI lower high → SELL setup
    """
    if len(price) < lookback + 5:
        return ""

    p  = price.iloc[-lookback:].values
    r  = rsi.iloc[-lookback:].values

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
    if condition == "oversold"    and divergence == "bullish":
        return "🟢 BUY"
    if condition == "overbought"  and divergence == "bearish":
        return "🔴 SELL"
    if condition == "oversold"    and divergence == "bearish":
        return "⚠️  CONT ↓"
    if condition == "overbought"  and divergence == "bullish":
        return "⚠️  CONT ↑"
    return "—"


# ═════════════════════════════════════════════════════════════════════════════
# Data providers
# ═════════════════════════════════════════════════════════════════════════════
def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
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
    """
    Alpha Vantage free tier — requires ALPHAVANTAGE_API_KEY env var.
    Maps yfinance-style intervals to AV function names.
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    try:
        import requests

        av_map = {
            "5m":  ("TIME_SERIES_INTRADAY", "5min"),
            "1h":  ("TIME_SERIES_INTRADAY", "60min"),
            "1d":  ("TIME_SERIES_DAILY_ADJUSTED", None),
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

        # find the time series key
        ts_key = next((k for k in data if "Time Series" in k), None)
        if not ts_key:
            return None

        df = pd.DataFrame(data[ts_key]).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = [c.split(". ")[1].capitalize() for c in df.columns]
        df = df.rename(columns={"Adjusted close": "Close", "Close": "Close"})
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        # trim to requested period
        days = {"5d": 5, "30d": 30, "60d": 60, "180d": 180}.get(period, 30)
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]
        return df if not df.empty else None
    except Exception:
        return None


# ─── Demo / synthetic data ────────────────────────────────────────────────────
def _synthetic_price(n: int, start: float, vol: float = 0.01,
                     trend: float = 0.0) -> np.ndarray:
    """Geometric Brownian Motion."""
    rng   = np.random.default_rng(abs(hash(str(n) + str(start))) % (2**32))
    steps = rng.normal(trend, vol, n)
    return start * np.exp(np.cumsum(steps))


def _force_extreme(close: np.ndarray, condition: str, period: int = RSI_PERIOD) -> np.ndarray:
    """
    Nudge the last few bars so the RSI lands in the extreme zone.
    condition: 'overbought' | 'oversold'
    """
    close = close.copy()
    if condition == "overbought":
        # big consecutive up candles at the end
        for i in range(-period * 3, 0):
            close[i] *= 1.012
    else:
        for i in range(-period * 3, 0):
            close[i] *= 0.988
    return close


def _maybe_add_divergence(close: np.ndarray, condition: str, div_type: str) -> np.ndarray:
    """Inject a synthetic divergence pattern in the lookback window."""
    close = close.copy()
    lb = DIV_LOOKBACK + 5
    if len(close) < lb:
        return close

    # Place two swing points at roughly -lb and -lb//2
    i1 = -lb
    i2 = -(lb // 2)

    if div_type == "bullish" and condition == "oversold":
        # price: lower low at i2; RSI: higher low → need price to drop more at i2
        close[i1] *= 0.985
        close[i2] *= 0.980   # lower low in price
        # RSI will naturally be higher at i2 because the drop was more gradual
    elif div_type == "bearish" and condition == "overbought":
        close[i1] *= 1.015
        close[i2] *= 1.020   # higher high in price; RSI lower high via deceleration
        # flatten gains just before i2 so RSI decelerates
        for j in range(i2 + 1, 0):
            close[j] *= 0.998

    return close


def fetch_demo(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    """
    Generate realistic synthetic OHLCV data.
    ~30 % of calls produce an extreme RSI condition, and half of those
    include a divergence, to give the table interesting content.
    """
    n_bars = {"5m": 390, "1h": 200, "1d": 130}.get(interval, 150)
    rng    = random.Random(hash(ticker + interval))

    start_price = rng.uniform(20, 800)
    base_close  = _synthetic_price(n_bars, start_price)

    # Decide what scenario this ticker+timeframe shows
    scenario = rng.random()
    condition = None
    div_type  = None

    if scenario < 0.15:
        condition = "overbought"
        div_type  = rng.choice(["bearish", "bullish", None, None])
        base_close = _force_extreme(base_close, "overbought")
        if div_type:
            base_close = _maybe_add_divergence(base_close, condition, div_type)
    elif scenario < 0.30:
        condition = "oversold"
        div_type  = rng.choice(["bullish", "bearish", None, None])
        base_close = _force_extreme(base_close, "oversold")
        if div_type:
            base_close = _maybe_add_divergence(base_close, condition, div_type)

    # Build OHLCV frame
    spread = np.abs(np.diff(base_close, prepend=base_close[0])) * 0.5 + start_price * 0.002
    high   = base_close + spread
    low    = base_close - spread
    open_  = np.roll(base_close, 1)
    open_[0] = base_close[0]
    volume = np.abs(np.random.default_rng(42).normal(1_000_000, 300_000, n_bars)).astype(int)

    freq_map = {"5m": "5min", "1h": "h", "1d": "D"}
    freq     = freq_map.get(interval, "h")
    idx      = pd.date_range(end=datetime.now(), periods=n_bars, freq=freq)

    df = pd.DataFrame({
        "Open":   open_, "High": high, "Low": low,
        "Close":  base_close, "Volume": volume,
    }, index=idx)

    return df


# ─── Unified fetch ────────────────────────────────────────────────────────────
def fetch_ohlcv(ticker: str, interval: str, period: str,
                use_demo: bool = False) -> pd.DataFrame | None:
    if use_demo:
        return fetch_demo(ticker, interval, period)

    df = fetch_yfinance(ticker, interval, period)
    if df is not None:
        return df

    df = fetch_alphavantage(ticker, interval, period)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Ticker scanner
# ═════════════════════════════════════════════════════════════════════════════
def scan_ticker(ticker: str, use_demo: bool = False) -> dict | None:
    results = {}

    for tf_name, cfg in TIMEFRAMES.items():
        df = fetch_ohlcv(ticker, cfg["interval"], cfg["period"], use_demo)
        if df is None:
            continue

        if tf_name == "4h":
            df = _resample_4h(df)

        if len(df) < RSI_PERIOD + 10:
            continue

        close = df["Close"].squeeze()
        rsi   = calc_rsi(close)
        last  = float(rsi.iloc[-1])

        if np.isnan(last):
            continue

        if last >= OVERBOUGHT:
            condition = "overbought"
        elif last <= OVERSOLD:
            condition = "oversold"
        else:
            continue

        div = detect_divergence(close, rsi)
        results[tf_name] = {
            "rsi":       round(last, 1),
            "condition": condition,
            "divergence": div,
        }

    return results if results else None


# ═════════════════════════════════════════════════════════════════════════════
# Rendering
# ═════════════════════════════════════════════════════════════════════════════
def _cell(info: dict) -> Text:
    rsi_val  = info["rsi"]
    cond     = info["condition"]
    div      = info["divergence"]
    signal   = resolve_signal(cond, div)

    color    = "red" if cond == "overbought" else "green"
    icon     = "🔺" if cond == "overbought" else "🔻"

    cell = Text()
    cell.append(f"RSI {rsi_val} {icon}", style=f"bold {color}")
    if div == "bullish":
        cell.append("  ↑div", style="bold green")
    elif div == "bearish":
        cell.append("  ↓div", style="bold red")
    cell.append(f"\n{signal}")
    return cell


def render_table(scan_data: dict, demo: bool = False):
    tf_cols = list(TIMEFRAMES.keys())
    mode_tag = "  [dim yellow][DEMO][/dim yellow]" if demo else ""

    table = Table(
        title=(
            f"[bold cyan]S&P 500 · RSI({RSI_PERIOD}) Scanner[/bold cyan]"
            f"{mode_tag}  "
            f"[dim]{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}[/dim]"
        ),
        box=box.SIMPLE_HEAD,
        show_lines=True,
        expand=True,
    )

    table.add_column("Ticker", style="bold white", width=8)
    for tf in tf_cols:
        table.add_column(tf, justify="center", width=22)

    for ticker, tf_data in sorted(scan_data.items()):
        row: list = [ticker]
        for tf in tf_cols:
            if tf not in tf_data:
                row.append(Text("·", style="dim"))
            else:
                row.append(_cell(tf_data[tf]))
        table.add_row(*row)

    console.print(table)


def print_legend():
    console.print(
        "\n[bold]Leyenda[/bold]\n"
        "  🔺 Sobrecompra RSI ≥ [bold red]80[/bold red]   "
        "🔻 Sobreventa RSI ≤ [bold green]20[/bold green]\n"
        "  [bold green]↑div[/bold green] divergencia alcista (precio LL · RSI HL)   "
        "[bold red]↓div[/bold red] divergencia bajista (precio HH · RSI LH)\n"
        "  🟢 BUY  = sobreventa + div alcista   "
        "🔴 SELL = sobrecompra + div bajista\n"
        "  ⚠️  CONT = divergencia confirma tendencia (señal de continuación)\n"
    )


def print_summary(scan_data: dict):
    buy_s  = sell_s = 0
    for td in scan_data.values():
        for info in td.values():
            sig = resolve_signal(info["condition"], info["divergence"])
            if sig == "🟢 BUY":
                buy_s += 1
            elif sig == "🔴 SELL":
                sell_s += 1

    console.print(
        f"[bold]Resumen[/bold]  "
        f"Acciones con señal activa: [cyan]{len(scan_data)}[/cyan]  |  "
        f"[bold green]BUY: {buy_s}[/bold green]  |  "
        f"[bold red]SELL: {sell_s}[/bold red]\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="S&P 500 RSI(4) multi-timeframe scanner with divergence detection."
    )
    p.add_argument("--demo",    action="store_true",
                   help="Use synthetic data (no internet required).")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Scan specific tickers only, e.g. --tickers AAPL MSFT NVDA")
    p.add_argument("--top",     type=int, default=None,
                   help="Scan only first N tickers from the list (e.g. --top 20).")
    return p.parse_args()


def main():
    args    = parse_args()
    tickers = args.tickers or (
        SP500_TICKERS[:args.top] if args.top else SP500_TICKERS
    )
    demo    = args.demo or not _yfinance_available()

    if demo and not args.demo:
        console.print(
            "[yellow]⚠  yfinance / Alpha Vantage no disponible — "
            "usando datos sintéticos (--demo).[/yellow]\n"
        )

    console.print(
        f"\n[bold cyan]Escaneando {len(tickers)} acciones del S&P 500…[/bold cyan]\n"
        f"RSI period=[bold]{RSI_PERIOD}[/bold]  "
        f"Sobrecompra≥[bold red]{OVERBOUGHT}[/bold red]  "
        f"Sobreventa≤[bold green]{OVERSOLD}[/bold green]  "
        f"Timeframes: [italic]{', '.join(TIMEFRAMES)}[/italic]\n"
    )

    scan_data: dict = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        console.print(f"[dim]({i:>3}/{total}) {ticker:<8}[/dim]", end="\r")
        result = scan_ticker(ticker, use_demo=demo)
        if result:
            scan_data[ticker] = result

    console.print(" " * 60, end="\r")

    if not scan_data:
        console.print(
            "[yellow]No se encontraron acciones en condición extrema "
            "en este momento.[/yellow]"
        )
        return

    render_table(scan_data, demo=demo)
    print_legend()
    print_summary(scan_data)


def _yfinance_available() -> bool:
    try:
        import yfinance as yf
        df = yf.download("SPY", interval="1d", period="5d",
                         auto_adjust=True, progress=False)
        return df is not None and not df.empty
    except Exception:
        return False


if __name__ == "__main__":
    main()
