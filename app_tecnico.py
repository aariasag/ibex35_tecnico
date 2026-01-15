import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Monitor IBEX 35 - Sistema Pro", layout="wide")
st.title("📊 Monitor IBEX 35 - Momentum & Gestión de Riesgo")

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

# -----------------------------------------------------------------------------
# 1. PANEL LATERAL COMPLETO
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Gatillos de Operativa")
    buy_score = st.slider("Score Compra (Gatillo)", 40, 90, 70)
    sell_score = st.slider("Score Venta (Salida)", 20, 60, 30)
    
    st.header("🛠️ Configuración del Score")
    with st.expander("📈 Tendencia", expanded=True):
        p_sma_bull_cross = st.slider("Precio > SMA50 > SMA200", 0, 50, 50)
        p_sma_bull_simple = st.slider("Precio > SMA50 y SMA200", 0, 50, 50)
        p_sma_min = st.slider("Precio > SMA200 solamente", 0, 30, 30)
        
    with st.expander("🚀 Momentum (RSI 21)"):
        rsi_range = st.slider("Rango Ideal (Puntos Max)", 0, 100, (60, 75))
        p_rsi_hot = st.slider("Puntos en Rango Ideal", 0, 40, 25)
        p_rsi_ob = st.slider("Penalización Sobrecompra (>75)", 0, 30, 8)
        
    with st.expander("📊 MACD y Volumen"):
        p_macd_strong = st.slider("MACD > Signal & > 0", 0, 20, 10)
        p_vol_high = st.slider("Volumen > 1.5x media", 0, 15, 6)
        
    with st.expander("🛡️ Gestión de Riesgo (ATR)"):
        atr_mult_stop = st.slider("Multiplicador ATR (Stop Loss)", 1.0, 4.0, 2.0, step=0.1)
        rr_ratio = st.slider("Ratio Riesgo/Beneficio (R:R)", 1.0, 5.0, 2.0, step=0.5)
        p_dist_max = st.slider("Distancia a SMA50 max (%)", 5, 30, 15)
        p_pen_dist = st.slider("Penalización Sobreextensión", 0, 30, 10)
        p_pen_volat = st.slider("Penalización Volatilidad Alta", 0, 40, 0)

    params = {
        'buy_threshold': buy_score, 'sell_threshold': sell_score,
        'w_sma_bull_cross': p_sma_bull_cross, 'w_sma_bull_simple': p_sma_bull_simple, 'w_sma_min': p_sma_min,
        'thr_overextended': p_dist_max, 'pen_overextended': p_pen_dist,
        'rsi_high_min': rsi_range[0], 'rsi_high_max': rsi_range[1], 'w_rsi_hot': p_rsi_hot,
        'rsi_overbought': 75, 'pen_rsi_ob': p_rsi_ob,
        'w_macd_strong': p_macd_strong, 'thr_vol_high': 1.5, 'w_vol_high': p_vol_high,
        'thr_volat_high': 35, 'pen_volat_high': p_pen_volat,
        'atr_mult': atr_mult_stop, 'rr': rr_ratio
    }

# -----------------------------------------------------------------------------
# 2. FUNCIONES TÉCNICAS
# -----------------------------------------------------------------------------
def calcular_rsi(series, period=21):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calcular_atr(df, period=14):
    high, low, close_prev = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, abs(high - close_prev), abs(low - close_prev)], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# -----------------------------------------------------------------------------
# 3. PROCESAMIENTO
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def obtener_datos_completos(tickers, p):
    rows = []
    data = yf.download(tickers, period="1y", group_by='ticker', progress=False)
    
    for ticker in tickers:
        try:
            df = data[ticker].copy().dropna() if len(tickers) > 1 else data.copy().dropna()
            if len(df) < 200: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            close = df['Close'].iloc[-1]
            sma50 = df['Close'].rolling(50).mean().iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            rsi = calcular_rsi(df['Close']).iloc[-1]
            
            ema12 = df['Close'].ewm(span=12).mean(); ema26 = df['Close'].ewm(span=26).mean()
            macd_line = (ema12 - ema26).iloc[-1]; macd_signal = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
            
            rvol = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
            volat = df['Close'].pct_change().rolling(20).std().iloc[-1] * (252**0.5) * 100
            
            atr_val = calcular_atr(df).iloc[-1]
            sl = close - (atr_val * p['atr_mult'])
            riesgo_stop_pct = ((sl - close) / close) * 100
            
            # Score
            score = 0
            if close > sma50 > sma200: score += p['w_sma_bull_cross']
            if p['rsi_high_min'] <= rsi <= p['rsi_high_max']: score += p['w_rsi_hot']
            if macd_line > macd_signal: score += p['w_macd_strong']
            if rvol > 1.5: score += p['w_vol_high']

            señal = "🟢 COMPRA" if score >= p['buy_threshold'] else "🔴 VENTA" if score <= p['sell_threshold'] else "🟡 MANTENER"

            rows.append({
                "Ticker": ticker.replace(".MC", ""),
                "Compañía": NOMBRES_IBEX.get(ticker),  # Nueva columna
                "Score": int(max(0, score)),
                "Señal": señal,
                "Precio": round(close, 2),
                "Stop Loss (€)": round(sl, 2),
                "Riesgo Stop %": f"{riesgo_stop_pct:.2f}%",
                "Volatilidad": f"{volat:.1f}%",
                "RSI(21)": round(rsi, 1),
                "Volumen Rel.": round(rvol, 2),
                "MACD": "Alcista" if macd_line > macd_signal else "Bajista"
            })
        except: continue
    return pd.DataFrame(rows)

# --- RENDER ---
df_final = obtener_datos_completos(IBEX35_TICKERS, params)

if not df_final.empty:
    df_final = df_final.sort_values("Score", ascending=False)
    st.subheader("📋 Dashboard de Mercado")
    st.dataframe(df_final, use_container_width=True, hide_index=True)

    st.divider()
    
    ticker_sel = st.selectbox("Analizar Gráfico (3 Años):", ["---"] + df_final["Ticker"].tolist())
    
    if ticker_sel != "---":
        df_hist = yf.download(ticker_sel + ".MC", period="3y", progress=False)
        if isinstance(df_hist.columns, pd.MultiIndex): df_hist.columns = df_hist.columns.get_level_values(0)
            
        if not df_hist.empty:
            df_hist['SMA50'] = df_hist['Close'].rolling(50).mean()
            df_hist['SMA200'] = df_hist['Close'].rolling(200).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], name='Precio', line=dict(color='#1f77b4', width=2)))
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA50'], name='SMA 50', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA200'], name='SMA 200', line=dict(color='red', width=1)))
            
            fig.update_layout(
                title=f"Histórico 3 Años: {NOMBRES_IBEX.get(ticker_sel + '.MC')}",
                xaxis_title="Fecha", yaxis_title="Precio (€)",
                height=500, template="plotly_white", hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
