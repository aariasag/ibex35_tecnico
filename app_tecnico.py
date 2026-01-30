import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Monitor IBEX 35 - Sistema Pro", layout="wide")
st.title("📊 Monitor IBEX 35 - Momentum & Gestión de Riesgo (Expert Mode)")

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
    with st.expander("📈 Tendencia (ADX & SMA)", expanded=True):
        p_sma_bull_cross = st.slider("Precio > SMA50 > SMA200", 0, 50, 50)
        p_adx_min = st.slider("Fuerza Tendencia (ADX > 20)", 0, 30, 20)
        
    with st.expander("🚀 Momentum (RSI Wilders)"):
        rsi_range = st.slider("Rango Ideal (Puntos Max)", 0, 100, (60, 75))
        p_rsi_hot = st.slider("Puntos en Rango Ideal", 0, 40, 25)
        p_rsi_ob = st.slider("Penalización Sobrecompra (>75)", 0, 30, 15)
        
    with st.expander("📊 MACD y Volumen"):
        p_macd_strong = st.slider("MACD > Signal & > 0", 0, 20, 10)
        p_vol_high = st.slider("Volumen > 1.5x media", 0, 15, 6)
        
    with st.expander("🛡️ Gestión de Riesgo"):
        atr_mult_stop = st.slider("Multiplicador ATR (Stop Loss)", 1.0, 5.0, 3.0, step=0.1)
        rr_ratio = st.slider("Ratio Riesgo/Beneficio (R:R)", 1.0, 5.0, 2.0, step=0.5)
        
        st.subheader("⚠️ Penalizaciones")
        p_dist_max = st.slider("Umbral Sobre-extensión (%)", 5, 30, 15)
        p_pen_dist = st.slider("Penalización Distancia SMA50", 0, 50, 20)
        p_pen_volat = st.slider("Penalización Volatilidad Extrema", 0, 40, 10)

    params = {
        'buy_threshold': buy_score, 'sell_threshold': sell_score,
        'w_sma_bull_cross': p_sma_bull_cross, 'w_adx_strong': p_adx_min,
        'thr_overextended': p_dist_max, 'pen_overextended': p_pen_dist,
        'rsi_high_min': rsi_range[0], 'rsi_high_max': rsi_range[1], 'w_rsi_hot': p_rsi_hot,
        'rsi_overbought': 75, 'pen_rsi_ob': p_rsi_ob,
        'w_macd_strong': p_macd_strong, 'thr_vol_high': 1.5, 'w_vol_high': p_vol_high,
        'thr_volat_high': 40, 'pen_volat_high': p_pen_volat,
        'atr_mult': atr_mult_stop
    }

# -----------------------------------------------------------------------------
# 2. FUNCIONES TÉCNICAS
# -----------------------------------------------------------------------------
def wilder_smooth(data, period):
    """Aplica el suavizado de Wilder (equivalente a EMA con alpha=1/period)"""
    return data.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calcular_rsi(series, period=14):
    """Calcula el RSI usando el método de suavizado de Wilder"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcular_atr_series(df, period=14):
    """Devuelve la serie ATR completa suavizada"""
    high, low, close_prev = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, abs(high - close_prev), abs(low - close_prev)], axis=1).max(axis=1)
    return wilder_smooth(tr, period)

def calcular_adx(df, period=14):
    """Calcula el ADX usando el método estándar de Wilder"""
    # 1. True Range (ya suavizado para ADX es el ATR)
    atr = calcular_atr_series(df, period)
    
    # 2. Directional Movement (+DM, -DM)
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff() # Drop is positive movement
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)
    
    # 3. Suavizar DM
    plus_dm_smooth = wilder_smooth(plus_dm_series, period)
    minus_dm_smooth = wilder_smooth(minus_dm_series, period)
    
    # 4. Calcular DI (+DI, -DI)
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # 5. Calcular DX y ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = wilder_smooth(dx, period)
    
    return adx

# -----------------------------------------------------------------------------
# 3. PROCESAMIENTO
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def obtener_datos_completos(tickers, p):
    rows = []
    # Descargamos 2 años para asegurar suficiente historial para el suavizado de Wilder
    data = yf.download(tickers, period="2y", group_by='ticker', progress=False) 
    
    for ticker in tickers:
        try:
            df = data[ticker].copy().dropna() if len(tickers) > 1 else data.copy().dropna()
            
            # Validación básica de datos
            if len(df) < 200: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            # --- CÁLCULOS TÉCNICOS ---
            close = df['Close'].iloc[-1]
            sma50 = df['Close'].rolling(50).mean().iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            
            # Indicadores (aseguramos que no den NaN tomando el último valor válido si es necesario)
            rsi_series = calcular_rsi(df['Close'])
            rsi = rsi_series.iloc[-1]
            
            adx_series = calcular_adx(df)
            adx = adx_series.iloc[-1]
            
            # Si ADX/RSI son NaN por falta de datos al inicio, saltamos o ponemos 0
            if pd.isna(rsi) or pd.isna(adx):
                continue

            ema12 = df['Close'].ewm(span=12).mean(); ema26 = df['Close'].ewm(span=26).mean()
            macd_line = (ema12 - ema26).iloc[-1]; macd_signal = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
            
            rvol = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
            volat_anual = df['Close'].pct_change().rolling(20).std().iloc[-1] * (252**0.5) * 100
            
            atr_val = calcular_atr_series(df).iloc[-1]
            sl = close - (atr_val * p['atr_mult'])
            
            # Distancia a la media (Mean Reversion Risk)
            dist_sma50_pct = ((close - sma50) / sma50) * 100
            
            # --- SISTEMA DE PUNTUACIÓN (SCORE) ---
            score = 0
            buy_reasons = []
            penalties = []
            
            # 1. Tendencia
            if close > sma50 > sma200: 
                score += p['w_sma_bull_cross']
                buy_reasons.append("Tendencia Alcista (SMA)")
            
            if adx > 20: 
                score += p['w_adx_strong']
                buy_reasons.append(f"Tendencia Fuerte (ADX {adx:.0f})")

            # 2. Momentum (RSI)
            if p['rsi_high_min'] <= rsi <= p['rsi_high_max']: 
                score += p['w_rsi_hot']
                buy_reasons.append("RSI Óptimo")
            elif rsi > p['rsi_overbought']:
                score -= p['pen_rsi_ob']
                penalties.append("RSI Sobrecompra")

            # 3. MACD y Volumen
            if macd_line > macd_signal: 
                score += p['w_macd_strong']
                buy_reasons.append("MACD > Signal")
            if rvol > p['thr_vol_high']: 
                score += p['w_vol_high']
                buy_reasons.append("Volumen Alto")

            # 4. Penalizaciones de Riesgo
            if dist_sma50_pct > p['thr_overextended']:
                score -= p['pen_overextended']
                penalties.append(f"Extendido (+{dist_sma50_pct:.0f}%)")
                
            if volat_anual > p['thr_volat_high']:
                score -= p['pen_volat_high']
                penalties.append(f"Volatilidad Alta ({volat_anual:.0f}%)")

            # Señal Final
            señal = "🟢 COMPRA" if score >= p['buy_threshold'] else "🔴 VENTA" if score <= p['sell_threshold'] else "🟡 MANTENER"

            rows.append({
                "Ticker": ticker.replace(".MC", ""),
                "Compañía": NOMBRES_IBEX.get(ticker),
                "Score": int(min(100, max(0, score))),
                "Señal": señal,
                "Precio": round(close, 2),
                "RSI(14)": round(rsi, 2),
                "ADX(14)": round(adx, 2),
                "Dist. SMA50": f"{dist_sma50_pct:.1f}%",
                "Volatilidad": f"{volat_anual:.1f}%",
                "Factores Positivos": " + ".join(buy_reasons) if buy_reasons else "-",
                "Stop Loss (€)": round(sl, 2),
                "Stop Loss (%)": f"{((close - sl) / close * 100):.2f}%",
                "Alertas": " + ".join(penalties) if penalties else "OK"
            })
        except Exception as e:
            # print(f"Error en {ticker}: {e}") # Debug only
            continue
            
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 4. RENDER CON PESTAÑAS
# -----------------------------------------------------------------------------
tab_dashboard, tab_glosario = st.tabs(["🖥️ Dashboard & Análisis", "📖 Glosario & Metodología"])

with tab_dashboard:
    df_final = obtener_datos_completos(IBEX35_TICKERS, params)

    if not df_final.empty:
        df_final = df_final.sort_values("Score", ascending=False)
        
        # Métricas Globales
        c1, c2, c3 = st.columns(3)
        c1.metric("Oportunidades de Compra", len(df_final[df_final["Señal"]=="🟢 COMPRA"]))
        c2.metric("Tickers en Alerta (Riesgo)", len(df_final[df_final["Alertas"] != "OK"]))
        c3.metric("Volatilidad Media IBEX", f"{pd.to_numeric(df_final['Volatilidad'].str.replace('%','')).mean():.1f}%")

        st.subheader("📋 Dashboard de Mercado")
        
        # Formateo condicional
        st.dataframe(
            df_final.style.map(lambda x: 'color: red' if 'VENTA' in str(x) else 'color: green' if 'COMPRA' in str(x) else '', subset=['Señal']),
            use_container_width=True, 
            hide_index=True
        )

        st.divider()
        
        col_sel, col_chart = st.columns([1, 4])
        
        with col_sel:
            st.info("Selecciona un valor para ver su configuración técnica detallada.")
            ticker_sel = st.radio("Valores:", df_final["Ticker"].tolist())

        with col_chart:
            if ticker_sel:
                ticker_full = ticker_sel + ".MC"
                df_hist = yf.download(ticker_full, period="2y", progress=False)
                if isinstance(df_hist.columns, pd.MultiIndex): df_hist.columns = df_hist.columns.get_level_values(0)
                    
                if not df_hist.empty:
                    df_hist['SMA50'] = df_hist['Close'].rolling(50).mean()
                    df_hist['SMA200'] = df_hist['Close'].rolling(200).mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], name='Precio', line=dict(color='black', width=1.5)))
                    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA50'], name='SMA 50', line=dict(color='#2ca02c', width=2)))
                    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA200'], name='SMA 200', line=dict(color='#d62728', width=2)))
                    
                    fig.update_layout(
                        title=f"{NOMBRES_IBEX.get(ticker_full)} - Análisis de Tendencia",
                        xaxis_title="", yaxis_title="Precio (€)",
                        height=500, template="plotly_white", hovermode="x unified",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Info adicional del valor seleccionado
                    row = df_final[df_final["Ticker"] == ticker_sel].iloc[0]
                    c_info1, c_info2, c_info3 = st.columns(3)
                    c_info1.success(f"**Factores Positivos:**\n\n{row['Factores Positivos']}")
                    if row['Alertas'] != "OK":
                        c_info2.error(f"**Alertas de Riesgo:**\n\n{row['Alertas']}")
                    else:
                        c_info2.info("**Riesgo:** Bajo/Controlado")
                        
                    c_info3.markdown(f"**Stop Loss Sugerido:**\n\n{row['Precio'] - (float(row['Volatilidad'].strip('%'))/100/16 * row['Precio']):.2f} € (aprox basado en vol)")

with tab_glosario:
    st.markdown("""
    # 📚 Glosario de Indicadores y Estrategia

    Esta aplicación utiliza un algoritmo de análisis técnico profesional multifactorial. A continuación se explica cada componente:

    ## 1. Indicadores de Tendencia
    *   **SMA50 y SMA200 (Medias Móviles Simples)**:
        *   La SMA50 representa la tendencia a medio plazo y la SMA200 la tendencia estructural a largo plazo.
        *   **Cruce Dorado**: Cuando el precio y la SMA50 están por encima de la SMA200, consideramos que el activo está en tendencia alcista sana.
    *   **ADX (Average Directional Index)**:
        *   Mide la **fuerza** de la tendencia, sin importar la dirección.
        *   **Regla**: Un ADX > 20 indica que hay una tendencia presente. Un ADX < 20 sugiere un mercado lateral ("ruido"), donde los sistemas de seguimiento de tendencia suelen fallar.

    ## 2. Indicadores de Momentum
    *   **RSI (Relative Strength Index)**:
        *   Mide la velocidad y el cambio de los movimientos de precios.
        *   **Zona Óptima (60-75)**: En tendencias alcistas fuertes, el RSI suele mantenerse en rangos altos sin caer de 40. Buscamos valores que muestren fuerza pero no exageración.
        *   **Sobrecompra (>75)**: Alerta de que el precio ha subido demasiado rápido y podría corregir inminentemente.

    ## 3. Gestión de Riesgo (Risk Management)
    *   **Distancia a SMA50 (Sobre-extensión)**:
        *   Los precios tienden a volver a su media ("Mean Reversion"). Si un precio sube un 15-20% por encima de su SMA50 en poco tiempo, el riesgo de "pullback" (retroceso) es altísimo. Penalizamos esto en el Score.
    *   **Volatilidad Anualizada**:
        *   Calculada sobre los últimos 20 días. Una volatilidad > 40% indica un activo muy nervioso/peligroso, lo que reduce la fiabilidad de las señales técnicas estándar.

    ## 🔍 Cómo interpretar el SCORE
    El **Score (0-100)** es la suma ponderada de todas las evidencias técnicas:
    *   **> 70 (🟢 COMPRA)**: Alineación perfecta de tendencia, momentum y volumen, con riesgo controlado.
    *   **< 30 (🔴 VENTA)**: Debilidad técnica severa o ruptura de soportes clave.
    *   **30 - 70 (🟡 MANTENER)**: Situación indefinida, lateral o con señales conflictivas (ej. buena tendencia pero sobrecomprado).
    """)
