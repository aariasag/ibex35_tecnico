import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="IBEX 35 - Expert Swing Trader v2",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 10px;
    }
    .bullish { color: #00ff88; font-weight: bold; }
    .bearish { color: #ff0055; font-weight: bold; }
    .neutral { color: #ffcc00; font-weight: bold; }
    .stDataFrame { font-size: 13px; }
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🦅 IBEX 35 — Sistema Swing Trading Experto v2.0")

# --- DATOS ESTÁTICOS ---
NOMBRES_IBEX = {
    "ACS.MC": "ACS", "ACX.MC": "Acerinox", "AENA.MC": "Aena", "AMS.MC": "Amadeus",
    "ANA.MC": "Acciona", "ANE.MC": "Acciona Energía", "BBVA.MC": "BBVA", "BKT.MC": "Bankinter",
    "CABK.MC": "CaixaBank", "CLNX.MC": "Cellnex", "COL.MC": "Colonial", "ELE.MC": "Endesa",
    "ENG.MC": "Enagás", "FDR.MC": "Fluidra", "FER.MC": "Ferrovial", "GRF.MC": "Grifols",
    "IAG.MC": "IAG (Iberia)", "IBE.MC": "Iberdrola", "IDR.MC": "Indra", "ITX.MC": "Inditex",
    "LOG.MC": "Logista", "MAP.MC": "Mapfre", "MRL.MC": "Merlin Prop.", "MTS.MC": "ArcelorMittal",
    "NTGY.MC": "Naturgy", "PUIG.MC": "Puig Brands", "RED.MC": "Redeia", "REP.MC": "Repsol",
    "ROVI.MC": "Rovi", "SAB.MC": "Sabadell", "SAN.MC": "Santander", "SCYR.MC": "Sacyr",
    "SLR.MC": "Solaria", "TEF.MC": "Telefónica", "UNI.MC": "Unicaja"
}
IBEX35_TICKERS = list(NOMBRES_IBEX.keys())
IBEX_INDEX = "^IBEX"

# =============================================================================
# 1. BIBLIOTECA DE INDICADORES (CORREGIDOS Y AMPLIADOS)
# =============================================================================

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calcular_atr_series(df, period=14):
    high, low, close_prev = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    # Wilder smoothing (EWM con alpha = 1/period)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()

def calcular_adx(df, period=14):
    """
    ADX corregido usando el método de suavizado de Wilder en toda la cadena de cálculo,
    consistente con TradingView y plataformas profesionales.
    """
    alpha = 1 / period

    plus_dm_raw = df['High'].diff()
    minus_dm_raw = -df['Low'].diff()

    plus_dm = np.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), plus_dm_raw, 0.0)
    minus_dm = np.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), minus_dm_raw, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, min_periods=period).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, min_periods=period).mean()
    atr = calcular_atr_series(df, period)

    plus_di = 100 * (plus_dm_s / atr)
    minus_di = 100 * (minus_dm_s / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.ewm(alpha=alpha, min_periods=period).mean()
    return adx, plus_di, minus_di

def calcular_obv(df):
    """On-Balance Volume — mide la presión compradora/vendedora acumulada."""
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()

def calcular_volume_ratio(df, period=20):
    """Ratio del volumen actual respecto a su media móvil. >1.5 = volumen expandido."""
    vol_ma = df['Volume'].rolling(period).mean()
    return df['Volume'] / vol_ma

def calcular_breakout_signal(df, lookback=20):
    """
    Detecta rotura alcista: precio cierra por encima del máximo de las últimas `lookback` velas
    con volumen expandido (ratio > 1.5).
    Retorna Serie booleana.
    """
    prev_high = df['High'].shift(1).rolling(lookback).max()
    vol_ratio = calcular_volume_ratio(df)
    breakout = (df['Close'] > prev_high) & (vol_ratio > 1.5)
    return breakout, prev_high, vol_ratio

# =============================================================================
# 2. FILTRO DE RÉGIMEN DE MERCADO (IBEX INDEX)
# =============================================================================

@st.cache_data(ttl=3600)
def get_market_regime():
    """
    Descarga el IBEX 35 y determina si el mercado está en régimen alcista o bajista.
    Alcista = precio > SMA200. En régimen bajista, NO se generan señales largas.
    """
    try:
        df = yf.download(IBEX_INDEX, period="2y", progress=False)
        if df.empty:
            return True, None  # Fallback: asumir alcista si no hay datos
        df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
        sma200 = df['Close'].rolling(200).mean()
        last_close = df['Close'].iloc[-1]
        last_sma200 = sma200.iloc[-1]
        regime_bullish = last_close > last_sma200
        return bool(regime_bullish), df
    except:
        return True, None

# =============================================================================
# 3. LÓGICA DE SCORING (MEJORADA)
# =============================================================================

def analizar_ticker(ticker, df_diario, config, market_bullish=True):
    try:
        if len(df_diario) < 260:
            return None

        # --- MARCO SEMANAL ---
        df_semanal = df_diario.resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        if len(df_semanal) < 52:
            return None

        df_semanal['EMA30'] = df_semanal['Close'].ewm(span=30).mean()
        weekly_close = df_semanal['Close'].iloc[-1]
        weekly_ema30 = df_semanal['EMA30'].iloc[-1]
        weekly_trend_bullish = weekly_close > weekly_ema30

        # --- MARCO DIARIO ---
        close = df_diario['Close'].iloc[-1]
        sma50 = df_diario['Close'].rolling(50).mean().iloc[-1]
        sma200 = df_diario['Close'].rolling(200).mean().iloc[-1]
        rsi = calcular_rsi(df_diario['Close'], 14).iloc[-1]
        atr = calcular_atr_series(df_diario, 14).iloc[-1]
        adx, plus_di, minus_di = calcular_adx(df_diario, 14)
        adx_val = adx.iloc[-1]
        plus_di_val = plus_di.iloc[-1]
        minus_di_val = minus_di.iloc[-1]

        # --- VOLUMEN ---
        obv = calcular_obv(df_diario)
        obv_sma = obv.rolling(20).mean()
        obv_bullish = obv.iloc[-1] > obv_sma.iloc[-1]
        vol_ratio = calcular_volume_ratio(df_diario).iloc[-1]

        # --- DISPARADOR DE ENTRADA ---
        breakout_series, prev_high_series, vol_ratio_series = calcular_breakout_signal(df_diario)
        breakout_today = breakout_series.iloc[-1]
        prev_high_val = prev_high_series.iloc[-1]

        # --- SCORING ---
        score = 0
        score_log = []

        # FILTRO MAESTRO: Régimen de mercado
        if not market_bullish:
            score -= 30
            score_log.append({
                "Regla": "🌐 Régimen de Mercado (IBEX SMA200)",
                "Valor": "⚠️ BAJISTA",
                "Puntos": "-30",
                "Detalle": "IBEX 35 por debajo de SMA200 — Sistema en modo DEFENSIVO"
            })
        else:
            score_log.append({
                "Regla": "🌐 Régimen de Mercado (IBEX SMA200)",
                "Valor": "✅ Alcista",
                "Puntos": "0",
                "Detalle": "IBEX 35 por encima de SMA200 — Generación de señales activa"
            })

        # 1. Tendencia semanal
        if weekly_trend_bullish:
            pts = 25
            score += pts
            score_log.append({"Regla": "Tendencia Semanal (EMA30)", "Valor": "🟢 Alcista", "Puntos": f"+{pts}", "Detalle": f"Precio ({weekly_close:.2f}) > EMA30 ({weekly_ema30:.2f})"})
        else:
            pts = -20
            score += pts
            score_log.append({"Regla": "Tendencia Semanal (EMA30)", "Valor": "🔴 Bajista", "Puntos": f"{pts}", "Detalle": f"Precio ({weekly_close:.2f}) < EMA30 ({weekly_ema30:.2f})"})

        # 2. SMA 200
        if close > sma200:
            pts = 20
            score += pts
            score_log.append({"Regla": "Soporte Mayor (SMA200)", "Valor": "✅ Sobre soporte", "Puntos": f"+{pts}", "Detalle": f"Precio ({close:.2f}) > SMA200 ({sma200:.2f})"})
        else:
            pts = -15
            score += pts
            score_log.append({"Regla": "Soporte Mayor (SMA200)", "Valor": "❌ Bajo soporte", "Puntos": f"{pts}", "Detalle": f"Precio ({close:.2f}) < SMA200 ({sma200:.2f})"})

        # 3. SMA50
        if close > sma50:
            pts = 10
            score += pts
            score_log.append({"Regla": "Tendencia Mediano Plazo (SMA50)", "Valor": "✅ Alcista", "Puntos": f"+{pts}", "Detalle": f"Precio ({close:.2f}) > SMA50 ({sma50:.2f})"})
        else:
            score_log.append({"Regla": "Tendencia Mediano Plazo (SMA50)", "Valor": "❌ Bajista", "Puntos": "0", "Detalle": f"Precio ({close:.2f}) < SMA50 ({sma50:.2f})"})

        # 4. ADX + DI direccional
        if adx_val > 25 and plus_di_val > minus_di_val:
            pts = 15
            score += pts
            score_log.append({"Regla": "Fuerza Direccional (ADX + DI+)", "Valor": "💪 Fuerte Alcista", "Puntos": f"+{pts}", "Detalle": f"ADX ({adx_val:.1f}) > 25 y DI+ ({plus_di_val:.1f}) > DI- ({minus_di_val:.1f})"})
        elif adx_val > 25 and plus_di_val < minus_di_val:
            pts = -10
            score += pts
            score_log.append({"Regla": "Fuerza Direccional (ADX + DI-)", "Valor": "⚠️ Fuerte Bajista", "Puntos": f"{pts}", "Detalle": f"ADX ({adx_val:.1f}) > 25 pero DI- ({minus_di_val:.1f}) domina"})
        elif adx_val < 20:
            pts = -5
            score += pts
            score_log.append({"Regla": "Fuerza de Tendencia (ADX)", "Valor": "😴 Débil/Lateral", "Puntos": f"{pts}", "Detalle": f"ADX ({adx_val:.1f}) < 20 — Mercado sin dirección"})
        else:
            score_log.append({"Regla": "Fuerza de Tendencia (ADX)", "Valor": "🟡 Neutral", "Puntos": "0", "Detalle": f"ADX ({adx_val:.1f}) en zona neutra (20-25)"})

        # 5. RSI (CORREGIDO: zona pullback 40-60 es la oportunidad real)
        if 40 <= rsi <= 60:
            pts = 15
            score += pts
            score_log.append({"Regla": "Momentum RSI (Pullback)", "Valor": "🎯 Zona de Entrada", "Puntos": f"+{pts}", "Detalle": f"RSI ({rsi:.1f}) en zona óptima pullback (40-60)"})
        elif 60 < rsi <= 70:
            pts = 5
            score += pts
            score_log.append({"Regla": "Momentum RSI", "Valor": "🟡 Movimiento avanzado", "Puntos": f"+{pts}", "Detalle": f"RSI ({rsi:.1f}) — el movimiento ya está parcialmente hecho"})
        elif rsi > 75:
            pts = -10
            score += pts
            score_log.append({"Regla": "Momentum RSI", "Valor": "🔴 Sobrecompra", "Puntos": f"{pts}", "Detalle": f"RSI ({rsi:.1f}) > 75 — Riesgo de corrección"})
        elif rsi < 40:
            pts = -10
            score += pts
            score_log.append({"Regla": "Momentum RSI", "Valor": "🔴 Debilidad", "Puntos": f"{pts}", "Detalle": f"RSI ({rsi:.1f}) < 40 — Momentum negativo"})
        else:
            score_log.append({"Regla": "Momentum RSI", "Valor": "🟡 Neutral", "Puntos": "0", "Detalle": f"RSI ({rsi:.1f})"})

        # 6. OBV — Volumen confirma dirección
        if obv_bullish:
            pts = 10
            score += pts
            score_log.append({"Regla": "Presión de Volumen (OBV)", "Valor": "🟢 Compradora", "Puntos": f"+{pts}", "Detalle": "OBV por encima de su SMA20 — dinero entrando"})
        else:
            pts = -5
            score += pts
            score_log.append({"Regla": "Presión de Volumen (OBV)", "Valor": "🔴 Vendedora", "Puntos": f"{pts}", "Detalle": "OBV por debajo de su SMA20 — distribución"})

        # 7. Zona de Valor (pullback a SMA50)
        dist_sma50 = (close - sma50) / sma50
        if 0 < dist_sma50 < 0.05:
            pts = 10
            score += pts
            score_log.append({"Regla": "Zona de Valor (Pullback SMA50)", "Valor": "🎯 Oportunidad", "Puntos": f"+{pts}", "Detalle": f"Precio dentro del 5% de SMA50 — zona de rebote potencial"})

        # 8. Disparador de entrada: rotura con volumen
        if breakout_today:
            pts = 15
            score += pts
            score_log.append({"Regla": "🚨 Disparador de Entrada", "Valor": "✅ ROTURA ACTIVA", "Puntos": f"+{pts}", "Detalle": f"Cierre ({close:.2f}€) > Máx. 20 días ({prev_high_val:.2f}€) con volumen x{vol_ratio:.1f}"})
        else:
            score_log.append({"Regla": "🚨 Disparador de Entrada", "Valor": "⏳ Sin rotura", "Puntos": "0", "Detalle": f"Precio aún no supera máx. 20 días ({prev_high_val:.2f}€)"})

        # --- GESTIÓN DE RIESGO ---
        stop_loss = close - (atr * config['atr_mult'])
        risk_per_share = close - stop_loss
        risk_pct = (risk_per_share / close) * 100

        # Target dinámico: nivel de resistencia o 2R (lo que sea menor para ser conservador)
        target_fixed = close + (risk_per_share * config['rr_ratio'])
        high_52w = df_diario['High'].rolling(252).max().iloc[-1]
        target = min(target_fixed, high_52w * 0.98) if target_fixed > high_52w * 0.95 else target_fixed

        return {
            "Ticker": ticker.replace(".MC", ""),
            "Empresa": NOMBRES_IBEX.get(ticker, ticker),
            "Precio": close,
            "Score": round(score),
            "Tendencia Semanal": "🟢 Alcista" if weekly_trend_bullish else "🔴 Bajista",
            "Régimen Mercado": "🟢 Alcista" if market_bullish else "🔴 Bajista",
            "SMA200": sma200,
            "SMA50": sma50,
            "ADX": adx_val,
            "DI+": plus_di_val,
            "DI-": minus_di_val,
            "RSI(14)": rsi,
            "OBV Alcista": obv_bullish,
            "Vol Ratio": vol_ratio,
            "Rotura": "🚨 SÍ" if breakout_today else "—",
            "Stop Loss": stop_loss,
            "Target": target,
            "Riesgo %": risk_pct,
            "ATR": atr,
            "Score_Log": score_log,
            "_df": df_diario  # para gráfico
        }
    except Exception as e:
        return None

# =============================================================================
# 4. BACKTESTING VECTORIZADO
# =============================================================================

def run_backtest(df_raw, config, market_regime_df=None):
    """
    Backtesting vectorizado sobre datos históricos.
    Lógica:
    - Señal de ENTRADA: precio cierra por encima del máximo de 20 días con volumen expandido
      + EMA30 semanal alcista + RSI entre 30 y 70.
    - Filtro régimen: IBEX > SMA200.
    - SALIDA: Stop Loss (2xATR trailing) o Target (2R fijo).
    - Se simula operación a operación, una posición a la vez.
    """
    results = []
    all_trades_df = []

    tickers = [t for t in df_raw.columns.get_level_values(0).unique() if t in IBEX35_TICKERS]

    for ticker in tickers:
        try:
            df = df_raw[ticker].copy().dropna()
            if len(df) < 300:
                continue

            # Indicadores
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['RSI'] = calcular_rsi(df['Close'], 14)
            df['ATR'] = calcular_atr_series(df, 14)
            df['OBV'] = calcular_obv(df)
            df['OBV_SMA'] = df['OBV'].rolling(20).mean()
            df['VolRatio'] = calcular_volume_ratio(df)

            # Semanal EMA30
            df_w = df.resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            df_w['EMA30'] = df_w['Close'].ewm(span=30).mean()
            df_w['WeeklyTrend'] = df_w['Close'] > df_w['EMA30']
            weekly_trend = df_w['WeeklyTrend'].resample('D').ffill()
            df['WeeklyTrend'] = weekly_trend.reindex(df.index, method='ffill')

            # Régimen IBEX
            if market_regime_df is not None and not market_regime_df.empty:
                ibex_sma200 = market_regime_df['Close'].rolling(200).mean()
                ibex_regime = (market_regime_df['Close'] > ibex_sma200)
                df['MarketBullish'] = ibex_regime.reindex(df.index, method='ffill').fillna(True)
            else:
                df['MarketBullish'] = True

            # Señal de entrada
            df['PrevHigh20'] = df['High'].shift(1).rolling(20).max()
            df['Entry_Signal'] = (
                (df['Close'] > df['PrevHigh20']) &
                (df['VolRatio'] > 1.5) &
                (df['WeeklyTrend'] == True) &
                (df['Close'] > df['SMA200']) &
                (df['RSI'] > 30) & (df['RSI'] < 70) &
                (df['OBV'] > df['OBV_SMA']) &
                (df['MarketBullish'] == True)
            )

            # Simulación trade a trade
            in_trade = False
            entry_price = 0
            stop = 0
            target_price = 0
            entry_date = None
            trade_atr = 0

            ticker_trades = []
            closes = df['Close'].values
            atrs = df['ATR'].values
            signals = df['Entry_Signal'].values
            dates = df.index

            for i in range(1, len(df)):
                if not in_trade:
                    if signals[i]:
                        in_trade = True
                        entry_price = closes[i]
                        trade_atr = atrs[i]
                        stop = entry_price - (trade_atr * config['atr_mult'])
                        target_price = entry_price + ((entry_price - stop) * config['rr_ratio'])
                        entry_date = dates[i]
                else:
                    # Gestión trailing stop (ajusta stop si el precio sube)
                    new_stop = closes[i] - (atrs[i] * config['atr_mult'])
                    stop = max(stop, new_stop)  # trailing: nunca baja

                    hit_stop = closes[i] <= stop
                    hit_target = closes[i] >= target_price

                    if hit_stop or hit_target:
                        exit_price = stop if hit_stop else target_price
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                        duration = (dates[i] - entry_date).days
                        ticker_trades.append({
                            "Ticker": ticker.replace(".MC", ""),
                            "Empresa": NOMBRES_IBEX.get(ticker, ticker),
                            "Entrada": entry_date.date(),
                            "Salida": dates[i].date(),
                            "Precio Entrada": round(entry_price, 2),
                            "Precio Salida": round(exit_price, 2),
                            "Resultado": "✅ Profit" if not hit_stop else "❌ Stop",
                            "PnL %": round(pnl_pct, 2),
                            "Duración (días)": duration
                        })
                        in_trade = False

            if ticker_trades:
                ticker_df = pd.DataFrame(ticker_trades)
                all_trades_df.append(ticker_df)
                wins = ticker_df[ticker_df['PnL %'] > 0]
                losses = ticker_df[ticker_df['PnL %'] <= 0]
                results.append({
                    "Ticker": ticker.replace(".MC", ""),
                    "Empresa": NOMBRES_IBEX.get(ticker, ticker),
                    "Total Trades": len(ticker_df),
                    "% Acierto": round(len(wins) / len(ticker_df) * 100, 1),
                    "PnL Medio %": round(ticker_df['PnL %'].mean(), 2),
                    "PnL Total %": round(ticker_df['PnL %'].sum(), 2),
                    "Mejor Trade %": round(ticker_df['PnL %'].max(), 2),
                    "Peor Trade %": round(ticker_df['PnL %'].min(), 2),
                    "Duración Media": round(ticker_df['Duración (días)'].mean(), 1),
                    "Profit Factor": round(wins['PnL %'].sum() / abs(losses['PnL %'].sum()), 2) if len(losses) > 0 and losses['PnL %'].sum() != 0 else float('inf')
                })
        except Exception as e:
            continue

    summary_df = pd.DataFrame(results).sort_values("PnL Total %", ascending=False) if results else pd.DataFrame()
    all_trades = pd.concat(all_trades_df, ignore_index=True) if all_trades_df else pd.DataFrame()
    return summary_df, all_trades

# =============================================================================
# 5. SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    with st.expander("🌐 Filtro de Mercado", expanded=True):
        use_market_filter = st.toggle("Filtro Régimen IBEX SMA200", value=True)
        st.caption("Desactiva señales largas cuando el IBEX cotiza por debajo de su SMA200.")

    with st.expander("🛡️ Gestión de Riesgo", expanded=True):
        capital = st.number_input("Capital (€)", value=10000, step=1000)
        risk_model_pct = st.slider("Riesgo/Op. (%)", 0.5, 3.0, 1.0, 0.1)
        atr_mult = st.slider("Stop x ATR", 1.5, 3.5, 2.0, 0.1)
        rr_ratio = st.slider("Ratio Riesgo/Beneficio", 1.5, 4.0, 2.0, 0.1)

    risk_amount = capital * risk_model_pct / 100
    st.info(f"**Riesgo Max por Op.:** {risk_amount:.2f}€")

    with st.expander("📡 Señal de Entrada", expanded=False):
        st.markdown("""
        **Condiciones activas:**
        - Cierre > Máximo 20 días
        - Volumen > 1.5x su media
        - EMA30 semanal alcista
        - Precio > SMA200
        - RSI entre 30 y 70
        - OBV > SMA20 OBV
        """)

    config = {'atr_mult': atr_mult, 'rr_ratio': rr_ratio}

# =============================================================================
# 6. TABS
# =============================================================================

tab_scanner, tab_backtest, tab_manual = st.tabs([
    "🚀 Scanner de Oportunidades",
    "📊 Backtesting",
    "📘 Manual y Metodología"
])

# ---------------------------------------------------------------------------
# TAB 1 — SCANNER
# ---------------------------------------------------------------------------
with tab_scanner:
    st.markdown("### 📊 Panel de Control Swing Trading")

    # Régimen de mercado siempre visible
    market_bullish, ibex_df = get_market_regime()
    regime_color = "🟢" if market_bullish else "🔴"
    regime_text = "ALCISTA — Sistema generando señales" if market_bullish else "BAJISTA — Sistema en modo DEFENSIVO"

    if not use_market_filter:
        market_bullish = True

    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        st.metric("Régimen IBEX 35", f"{regime_color} {regime_text}")
    with col_r2:
        if not market_bullish and use_market_filter:
            st.warning("⚠️ **ADVERTENCIA:** El IBEX 35 está en tendencia bajista (por debajo de SMA200). El sistema aplica penalización severa a todos los activos. No es momento de buscar largos.")

    st.divider()

    if st.button("🚀 Escanear Mercado", type="primary"):
        with st.spinner("Descargando datos y analizando IBEX 35..."):
            progreso = st.progress(0)
            resultados = []

            data_raw = yf.download(IBEX35_TICKERS, period="2y", group_by='ticker', progress=False)
            st.session_state['scan_data_raw'] = data_raw
            st.session_state['ibex_df'] = ibex_df

            for i, ticker in enumerate(IBEX35_TICKERS):
                try:
                    if isinstance(data_raw.columns, pd.MultiIndex):
                        df = data_raw[ticker].copy().dropna()
                    else:
                        df = data_raw.copy().dropna()

                    res = analizar_ticker(ticker, df, config, market_bullish)
                    if res:
                        resultados.append(res)
                except:
                    pass
                progreso.progress((i + 1) / len(IBEX35_TICKERS))

            progreso.empty()

            if resultados:
                df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ('Score_Log', '_df')} for r in resultados])
                df_res = df_res.sort_values("Score", ascending=False)
                st.session_state['scan_results'] = df_res
                st.session_state['scan_results_full'] = resultados
            else:
                st.warning("No hay datos disponibles.")

    if 'scan_results' in st.session_state:
        df_res = st.session_state['scan_results']
        data_raw = st.session_state['scan_data_raw']
        resultados_full = st.session_state['scan_results_full']

        # Tabla principal
        st.subheader("🏆 Ranking de Candidatos")

        # Filtro mínimo score
        min_score = st.slider("Mostrar scores ≥", -50, 100, 40, step=5)
        df_filtered = df_res[df_res['Score'] >= min_score]

        cols_display = ['Ticker', 'Empresa', 'Score', 'Rotura', 'Tendencia Semanal',
                        'Precio', 'RSI(14)', 'ADX', 'Vol Ratio', 'SMA50', 'SMA200']

        df_disp = df_filtered[cols_display].copy()
        df_disp['Precio'] = df_disp['Precio'].apply(lambda x: f"{x:.2f}€")
        df_disp['RSI(14)'] = df_disp['RSI(14)'].apply(lambda x: f"{x:.1f}")
        df_disp['ADX'] = df_disp['ADX'].apply(lambda x: f"{x:.1f}")
        df_disp['Vol Ratio'] = df_disp['Vol Ratio'].apply(lambda x: f"{x:.2f}x")
        df_disp['SMA50'] = df_disp['SMA50'].apply(lambda x: f"{x:.2f}€")
        df_disp['SMA200'] = df_disp['SMA200'].apply(lambda x: f"{x:.2f}€")

        st.dataframe(
            df_disp.style
            .applymap(lambda x: 'color: #00ff88' if '🟢' in str(x) else ('color: #ff0055' if '🔴' in str(x) else ''), subset=['Tendencia Semanal'])
            .applymap(lambda x: 'color: #ff9900; font-weight: bold' if '🚨' in str(x) else '', subset=['Rotura'])
            .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-30, vmax=110),
            use_container_width=True
        )

        st.divider()

        # Selección detallada
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔎 Análisis Detallado")
            ticker_calc = st.selectbox("Seleccionar Activo", df_res['Ticker'].tolist())
            row = df_res[df_res['Ticker'] == ticker_calc].iloc[0]
            row_full = next((r for r in resultados_full if r['Ticker'] == ticker_calc), None)

            st.markdown(f"#### Score Total: **{row['Score']} / 100+**")

            if row_full:
                df_log = pd.DataFrame(row_full['Score_Log'])
                st.table(df_log)

            st.info("Las reglas se suman algebraicamente. El Disparador de Entrada (🚨) indica si HOY hay una señal concreta.")

        with col2:
            st.subheader("🧮 Calculadora de Posición")
            stop_loss_val = row['Stop Loss']
            dist_stop = row['Precio'] - stop_loss_val
            shares = int(risk_amount / dist_stop) if dist_stop > 0 else 0
            inversion = shares * row['Precio']

            c1, c2 = st.columns(2)
            c1.metric("Acciones a Comprar", f"{shares} uds.")
            c2.metric("Inversión Total", f"{inversion:,.2f}€")

            c3, c4 = st.columns(2)
            c3.metric("Riesgo Monetario", f"-{risk_amount:.2f}€")
            c4.metric("Beneficio Potencial", f"+{risk_amount * rr_ratio:.2f}€")

            st.markdown("---")
            st.markdown(f"""
            | Nivel | Precio |
            |-------|--------|
            | 🟢 Entrada | **{row['Precio']:.2f}€** |
            | 🔴 Stop Loss | **{stop_loss_val:.2f}€** |
            | 🎯 Objetivo | **{row['Target']:.2f}€** |
            | 📉 Riesgo % | **{row['Riesgo %']:.2f}%** |
            """)

            if row['Rotura'] == "🚨 SÍ":
                st.success("🚨 SEÑAL ACTIVA: Hay un disparador de entrada confirmado HOY.")
            else:
                st.info("⏳ Sin disparador: Activo en observación. Esperar rotura con volumen.")

        # Gráfico técnico
        st.subheader(f"📈 Gráfico Técnico: {ticker_calc}")
        ticker_full = ticker_calc + ".MC"

        try:
            if isinstance(data_raw.columns, pd.MultiIndex):
                df_g = data_raw[ticker_full].copy().dropna()
            else:
                df_g = data_raw.copy().dropna()

            df_g = df_g.tail(400)
            df_g['SMA50'] = df_g['Close'].rolling(50).mean()
            df_g['SMA200'] = df_g['Close'].rolling(200).mean()
            atr_g = calcular_atr_series(df_g, 14)
            df_g['StopLine'] = df_g['Close'] - (atr_g * atr_mult)
            df_g['OBV'] = calcular_obv(df_g)
            df_g['OBV_SMA'] = df_g['OBV'].rolling(20).mean()
            df_g['PrevHigh20'] = df_g['High'].shift(1).rolling(20).max()
            rsi_g = calcular_rsi(df_g['Close'], 14)
            adx_g, plus_di_g, minus_di_g = calcular_adx(df_g, 14)

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.55, 0.22, 0.23],
                subplot_titles=["Precio + Medias + Stop", "RSI (14)", "ADX / DI"],
                vertical_spacing=0.05
            )

            # Row 1: Precio
            fig.add_trace(go.Candlestick(
                x=df_g.index, open=df_g['Open'], high=df_g['High'],
                low=df_g['Low'], close=df_g['Close'], name='Precio',
                increasing_line_color='#00ff88', decreasing_line_color='#ff0055'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_g.index, y=df_g['SMA50'], line=dict(color='orange', width=1.5), name='SMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_g.index, y=df_g['SMA200'], line=dict(color='royalblue', width=2), name='SMA 200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_g.index, y=df_g['StopLine'], line=dict(color='red', dash='dot', width=1), name='Trailing Stop'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_g.index, y=df_g['PrevHigh20'], line=dict(color='yellow', dash='dash', width=1), name='Máx 20 días'), row=1, col=1)

            # Row 2: RSI
            fig.add_trace(go.Scatter(x=df_g.index, y=rsi_g, line=dict(color='cyan', width=1.5), name='RSI 14'), row=2, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=2, col=1)
            fig.add_hline(y=70, line=dict(color='red', dash='dot', width=1), row=2, col=1)
            fig.add_hline(y=50, line=dict(color='gray', dash='dot', width=1), row=2, col=1)
            fig.add_hline(y=30, line=dict(color='green', dash='dot', width=1), row=2, col=1)

            # Row 3: ADX y DI
            fig.add_trace(go.Scatter(x=df_g.index, y=adx_g, line=dict(color='purple', width=2), name='ADX'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_g.index, y=plus_di_g, line=dict(color='#00ff88', width=1), name='DI+'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_g.index, y=minus_di_g, line=dict(color='#ff0055', width=1), name='DI-'), row=3, col=1)
            fig.add_hline(y=25, line=dict(color='gray', dash='dot', width=1), row=3, col=1)

            fig.update_layout(
                height=700,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.01),
                margin=dict(l=40, r=40, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generando gráfico: {e}")

# ---------------------------------------------------------------------------
# TAB 2 — BACKTESTING
# ---------------------------------------------------------------------------
with tab_backtest:
    st.markdown("### 📊 Motor de Backtesting Vectorizado")
    st.markdown("""
    Simula la estrategia completa sobre **datos históricos reales** de cada componente del IBEX 35.
    Usa las mismas condiciones que el Scanner: rotura de máximos con volumen, EMA30 semanal, SMA200 y filtro de régimen.
    El Stop es **trailing** (se ajusta al alza con el precio). El Target es dinámico.
    """)

    col_bt1, col_bt2 = st.columns([2, 1])
    with col_bt1:
        bt_period = st.select_slider("Periodo histórico", options=["1y", "2y", "3y", "5y"], value="3y")
    with col_bt2:
        bt_filter = st.toggle("Aplicar filtro régimen IBEX", value=True, key="bt_filter")

    if st.button("▶️ Ejecutar Backtesting", type="primary"):
        with st.spinner(f"Descargando {bt_period} de datos y simulando todas las operaciones..."):
            bt_raw = yf.download(IBEX35_TICKERS, period=bt_period, group_by='ticker', progress=False)
            ibex_bt = yf.download(IBEX_INDEX, period=bt_period, progress=False)
            if isinstance(ibex_bt.columns, pd.MultiIndex):
                ibex_bt.columns = ibex_bt.columns.droplevel(1)

            ibex_for_bt = ibex_bt if bt_filter else None
            summary, all_trades = run_backtest(bt_raw, config, ibex_for_bt)

            st.session_state['bt_summary'] = summary
            st.session_state['bt_trades'] = all_trades

    if 'bt_summary' in st.session_state:
        summary = st.session_state['bt_summary']
        all_trades = st.session_state['bt_trades']

        if summary.empty:
            st.warning("No se generaron trades con los parámetros actuales. Prueba aumentar el periodo o relajar los filtros.")
        else:
            # KPIs globales
            total_trades = all_trades.shape[0]
            wins = all_trades[all_trades['PnL %'] > 0]
            losses = all_trades[all_trades['PnL %'] <= 0]
            win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
            avg_win = wins['PnL %'].mean() if len(wins) > 0 else 0
            avg_loss = losses['PnL %'].mean() if len(losses) > 0 else 0
            profit_factor = wins['PnL %'].sum() / abs(losses['PnL %'].sum()) if losses['PnL %'].sum() != 0 else float('inf')
            expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

            st.subheader("📈 Estadísticas Globales del Sistema")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Operaciones", total_trades)
            c2.metric("Tasa de Acierto", f"{win_rate:.1f}%")
            c3.metric("Profit Factor", f"{profit_factor:.2f}", help=">1.5 es aceptable, >2 es bueno")
            c4.metric("Expectativa/Op.", f"{expectancy:.2f}%")
            c5.metric("Ganancia Media", f"{avg_win:.2f}%", delta=f"Pérdida media: {avg_loss:.2f}%")

            if profit_factor >= 2:
                st.success("✅ Sistema con Profit Factor robusto (≥ 2.0). Tiene expectativa matemática positiva.")
            elif profit_factor >= 1.5:
                st.info("🟡 Sistema acceptable (Profit Factor 1.5-2.0). Optimizable.")
            else:
                st.error("🔴 Sistema débil (Profit Factor < 1.5). Revisa parámetros o filtros.")

            st.divider()

            # PnL acumulado en el tiempo (curva de equity simulada)
            st.subheader("📉 Curva de Equity Acumulada (Sistema Completo)")

            if not all_trades.empty:
                equity_df = all_trades.copy()
                equity_df['Salida'] = pd.to_datetime(equity_df['Salida'])
                equity_df = equity_df.sort_values('Salida')
                equity_df['PnL Acum %'] = equity_df['PnL %'].cumsum()

                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=equity_df['Salida'],
                    y=equity_df['PnL Acum %'],
                    fill='tozeroy',
                    line=dict(color='#00ff88', width=2),
                    name="PnL Acumulado %"
                ))
                fig_eq.add_hline(y=0, line=dict(color='white', dash='dot'))
                fig_eq.update_layout(
                    template="plotly_dark", height=300,
                    xaxis_title="Fecha de Salida",
                    yaxis_title="PnL Acumulado (%)",
                    margin=dict(l=40, r=20, t=20, b=40)
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            # Tabla por ticker
            st.subheader("🏆 Rendimiento por Activo")

            def color_pnl(val):
                if isinstance(val, (int, float)):
                    return 'color: #00ff88' if val > 0 else 'color: #ff0055'
                return ''

            st.dataframe(
                summary.style
                .applymap(color_pnl, subset=['PnL Total %', 'PnL Medio %', 'Mejor Trade %', 'Peor Trade %'])
                .background_gradient(subset=['% Acierto'], cmap='RdYlGn', vmin=30, vmax=70)
                .background_gradient(subset=['Profit Factor'], cmap='RdYlGn', vmin=0.5, vmax=3.0),
                use_container_width=True
            )

            # Distribución de resultados
            st.subheader("📊 Distribución de Resultados")

            col_h1, col_h2 = st.columns(2)
            with col_h1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=all_trades['PnL %'],
                    nbinsx=40,
                    marker_color=np.where(
                        all_trades['PnL %'] > 0, '#00ff88', '#ff0055'
                    ),
                    name="Trades"
                ))
                fig_hist.add_vline(x=0, line=dict(color='white', dash='dot'))
                fig_hist.update_layout(
                    template="plotly_dark", height=300,
                    title="Distribución PnL por Trade (%)",
                    xaxis_title="PnL %", yaxis_title="Frecuencia"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_h2:
                fig_dur = go.Figure()
                fig_dur.add_trace(go.Histogram(
                    x=all_trades['Duración (días)'],
                    nbinsx=30,
                    marker_color='royalblue',
                    name="Duración"
                ))
                fig_dur.update_layout(
                    template="plotly_dark", height=300,
                    title="Distribución Duración de Trades (días)",
                    xaxis_title="Días", yaxis_title="Frecuencia"
                )
                st.plotly_chart(fig_dur, use_container_width=True)

            # Log de todas las operaciones
            st.subheader("📋 Log de Todas las Operaciones")
            st.dataframe(
                all_trades.sort_values('Salida', ascending=False).style
                .applymap(color_pnl, subset=['PnL %'])
                .applymap(lambda x: 'color: #00ff88' if '✅' in str(x) else 'color: #ff0055', subset=['Resultado']),
                use_container_width=True,
                height=400
            )

# ---------------------------------------------------------------------------
# TAB 3 — MANUAL
# ---------------------------------------------------------------------------
with tab_manual:
    st.markdown("""
    # 📘 Manual del Operador Swing — v2.0

    Esta aplicación implementa una estrategia **Trend Following con Disparador de Entrada** diseñada para capturar movimientos de varias semanas dentro de tendencias establecidas.

    ---

    ## 🌐 Filtro Maestro: Régimen de Mercado

    **Antes de cualquier análisis individual, el sistema evalúa la salud del mercado.**

    - **Condición**: El IBEX 35 (índice completo) debe cotizar por encima de su SMA200.
    - **Lógica**: "No puedes pescar peces en un estanque sin agua." En mercados bajistas, la mayoría de acciones caen independientemente de sus fundamentales. Operar largos en mercados bajistas es nadar contra la marea.
    - **Efecto**: Si el régimen es bajista, se aplica una penalización de -30 puntos a todos los activos y se muestra advertencia en pantalla.

    ---

    ## 1. El Algoritmo de Scoring

    | Regla | Condición | Puntos |
    |-------|-----------|--------|
    | Régimen Mercado | IBEX < SMA200 | -30 |
    | Tendencia Semanal | Precio > EMA30 semanal | +25 / -20 |
    | Soporte Mayor | Precio > SMA200 diaria | +20 / -15 |
    | Tendencia Media | Precio > SMA50 diaria | +10 |
    | ADX + DI | ADX>25 y DI+>DI- | +15 / -10 |
    | RSI Pullback | RSI en 40-60 (zona óptima) | +15 |
    | OBV | OBV > SMA20(OBV) | +10 / -5 |
    | Zona de Valor | Precio < 5% de SMA50 | +10 |
    | **Disparador de Entrada** | **Rotura máximo 20 días + volumen >1.5x** | **+15** |

    ---

    ## 2. El Disparador de Entrada 🚨

    **Este es el cambio más importante respecto a v1.0.**

    En v1.0, el sistema era un *screener*: te decía qué activos tenían buenas condiciones, pero no *cuándo* entrar. Comprar en "condiciones técnicas favorables" sin un disparador es como saber que va a llover sin saber cuándo salir con paraguas.

    **Señal de Entrada v2.0**:
    1. El precio de cierre supera el **máximo de los últimos 20 días** (rotura de resistencia reciente)
    2. El volumen del día de rotura es **≥ 1.5x** su media de 20 días (confirmación institucional)

    Una rotura sin volumen es una trampa. Una rotura con volumen expandido indica que hay dinero institucional detrás del movimiento.

    ---

    ## 3. Correcciones Técnicas v2.0

    **ADX (Corregido)**: Ahora usa el suavizado de Wilder (EWM con alpha=1/period) en toda la cadena, incluyendo DM+, DM- y TR. Los valores son ahora consistentes con TradingView y MetaTrader.

    **Adición de DI+ y DI-**: No basta con que el ADX sea alto. Un ADX alto con DI- > DI+ indica una tendencia *bajista* fuerte, no alcista. Ahora se distingue entre ambos casos.

    **RSI corregido**: La zona "óptima" cambia de 50-70 a **40-60**. Esta es la zona de *pullback* dentro de una tendencia alcista, donde el precio ha corregido lo suficiente para ofrecer una entrada con buen R:R sin estar ya en sobrecompra.

    **OBV añadido**: El On-Balance Volume suma el volumen de los días alcistas y resta el de los bajistas. Cuando el OBV está por encima de su media, hay más presión compradora que vendedora, confirmando la dirección del precio.

    ---

    ## 4. Stop Loss y Target Dinámicos

    **Stop Loss Trailing**: El stop se coloca a `precio - (ATR x multiplicador)`. A diferencia de v1.0 donde el stop era fijo desde la entrada, ahora el stop **sube con el precio** (trailing), protegiendo beneficios si el movimiento se desarrolla favorablemente.

    **Target Dinámico**: El objetivo no es siempre un múltiplo fijo del riesgo. El sistema comprueba el máximo de 52 semanas y ajusta el target si el precio está cerca de esa resistencia histórica, evitando objetivos irreales.

    ---

    ## 5. Backtesting — Cómo Interpretar los Resultados

    | Métrica | Qué mide | Referencia |
    |---------|----------|------------|
    | **Tasa de Acierto** | % operaciones ganadoras | 40-60% es normal en trend following |
    | **Profit Factor** | Suma ganancias / Suma pérdidas | >1.5 aceptable, >2.0 robusto |
    | **Expectativa/Op.** | Ganancia esperada promedio | Debe ser positiva |
    | **Duración Media** | Días promedio en posición | 15-40 días es swing típico |

    > 💡 **Un sistema puede acertar solo el 40% de las veces y ser muy rentable** si el Profit Factor es alto (las ganancias son mucho mayores que las pérdidas). El Profit Factor importa más que la tasa de acierto.

    ---

    ## 6. Gestión de Riesgo — Reglas de Oro

    1. **Nunca arriesgues más del 1-2%** del capital en una operación.
    2. **El tamaño de posición lo dicta el stop, no la convicción.** Calcula siempre cuántas acciones comprar basándote en el riesgo monetario y la distancia al stop.
    3. **El stop basado en ATR respeta la volatilidad del activo.** Un stop de 2x ATR le da al precio espacio para moverse sin que el ruido normal te saque de la operación.
    4. **No muevas el stop a la baja nunca.** Solo puede subir (trailing hacia arriba).
    5. **Si el mercado entra en régimen bajista (IBEX < SMA200), reduce exposición o cierra posiciones abiertas.**
    """)
    st.info("⚡ Esta aplicación es una herramienta de análisis técnico. No constituye asesoramiento financiero. Opera siempre con gestión de riesgo estricta.")

