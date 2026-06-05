# Market Maker — S&P 500 RSI(4) Scanner

Scanner de operativa para las 100 acciones del S&P 500 con RSI de período 4, detección de sobrecompra/sobreventa y divergencias alcistas/bajistas en múltiples temporalidades.

## Características

- **RSI(4)** calculado con EWM (Wilder smoothing)
- **4 temporalidades**: 5min · 1h · 4h · 1D
- **Extremos**: Sobrecompra ≥ 80 · Sobreventa ≤ 20
- **Divergencias clásicas**: alcista (precio LL + RSI HL) y bajista (precio HH + RSI LH)
- **Señales**: BUY / SELL / CONT (continuación)
- **Fuentes de datos**: yfinance → Alpha Vantage → modo demo sintético

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Escaneo completo (requiere internet — Yahoo Finance)
python sp500_rsi_scanner.py

# Solo ciertos tickers
python sp500_rsi_scanner.py --tickers AAPL MSFT NVDA TSLA

# Primeras N acciones de la lista
python sp500_rsi_scanner.py --top 20

# Modo demo (sin internet, datos sintéticos)
python sp500_rsi_scanner.py --demo

# Alpha Vantage (gratis en alphavantage.co)
ALPHAVANTAGE_API_KEY=tu_clave python sp500_rsi_scanner.py
```

## Tabla de salida

```
Ticker │ 5min              │ 1h                │ 4h                │ 1D
───────┼───────────────────┼───────────────────┼───────────────────┼──────────
AAPL   │ RSI 14.2 🔻 ↑div  │ RSI 17.5 🔻        │ ·                 │ ·
       │ 🟢 BUY            │ —                 │                   │
```

## Señales

| Icono | Significado |
|-------|-------------|
| 🔺    | Sobrecompra (RSI ≥ 80) |
| 🔻    | Sobreventa  (RSI ≤ 20) |
| ↑div  | Divergencia alcista — precio baja más, RSI baja menos |
| ↓div  | Divergencia bajista — precio sube más, RSI sube menos |
| 🟢 BUY  | Sobreventa + divergencia alcista → posible rebote |
| 🔴 SELL | Sobrecompra + divergencia bajista → posible caída |
| ⚠️ CONT | Divergencia confirma la tendencia (continuación) |

## Parámetros ajustables

En `sp500_rsi_scanner.py`:

```python
RSI_PERIOD   = 4    # período del RSI
OVERBOUGHT   = 80   # umbral sobrecompra
OVERSOLD     = 20   # umbral sobreventa
DIV_LOOKBACK = 20   # velas atrás para buscar divergencias
```
