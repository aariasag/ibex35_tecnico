import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
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

st.title("🦅 IBEX 35 — Sistema Swing Trading Experto v3.0")

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

def calcular_breakout_signal(df, lookback=20, vol_threshold=1.5):
    """
    Detecta rotura alcista: precio cierra por encima del máximo de las últimas `lookback` velas
    con volumen expandido (ratio > vol_threshold).
    Retorna Serie booleana.
    """
    prev_high = df['High'].shift(1).rolling(lookback).max()
    vol_ratio = calcular_volume_ratio(df)
    breakout = (df['Close'] > prev_high) & (vol_ratio > vol_threshold)
    return breakout, prev_high, vol_ratio

def calcular_rs_vs_ibex(df_ticker, df_ibex, period=20):
    """
    Fortaleza Relativa (RS) del activo vs el IBEX 35.
    RS = rendimiento del activo en `period` días / rendimiento del IBEX en `period` días.
    RS > 1.0  → el activo supera al índice → señal positiva
    RS < 1.0  → el activo va peor que el índice → señal negativa

    Esta métrica captura exactamente lo que el RSI no puede: si el precio
    del activo está subiendo MÁS que el mercado en conjunto, hay dinero
    institucional posicionándose en él específicamente.
    """
    ret_ticker = df_ticker['Close'].pct_change(period)
    ret_ibex   = df_ibex['Close'].pct_change(period).reindex(df_ticker.index, method='ffill')
    # Evitar división por cero cuando el IBEX no se mueve
    rs = (1 + ret_ticker) / (1 + ret_ibex.replace(0, np.nan))
    return rs

def calcular_rsi_prebreakout(rsi_series, ventana=5):
    """
    RSI medio de los N días ANTERIORES al día actual (no incluye el día actual).
    Mide si el activo estaba en zona de acumulación sana ANTES de la rotura,
    sin penalizar el RSI alto que inevitablemente acompaña a la rotura misma.
    """
    return rsi_series.shift(1).rolling(ventana).mean()

# =============================================================================
# 2. PATRONES DE ANTICIPACIÓN DE ROTURA
# =============================================================================
# Detectan la compresión ANTES de que ocurra la rotura, dando 1-5 días de
# ventaja para prepararse. Tres patrones complementarios:
#
# [VCP]  Volatility Contraction Pattern (Minervini)
#        Correcciones decrecientes en % y rango — el precio "aprieta" antes
#        de la expansión. Patrón estadísticamente más estudiado.
#
# [SQZ]  Squeeze de Bollinger + Keltner
#        Las BB entran dentro del canal de Keltner → volatilidad en mínimos.
#        Periodos de baja volatilidad preceden a movimientos grandes.
#
# [NR7]  Narrow Range 7 (y NR4)
#        Día con el rango más pequeño de los últimos 7 (o 4) días.
#        Máxima compresión intradiaria — señal de quietud antes del impulso.
# =============================================================================

def calcular_vcp(df, n_swings=3, max_ratio=0.6):
    """
    Volatility Contraction Pattern de Minervini.

    Detecta si los últimos `n_swings` retrocesos forman una serie decreciente
    donde cada corrección es significativamente menor que la anterior.

    Retorna dict con:
      - 'detectado': bool
      - 'correcciones': list de % de cada retroceso
      - 'ratio':  última corrección / primera (cuanto más bajo, más comprimido)
      - 'score':  0-100 indicando la calidad del patrón
      - 'pivot':  nivel de precio donde se espera la rotura
      - 'dias_compresion': cuántos días lleva el activo comprimiendo
    """
    if len(df) < 60:
        return {'detectado': False, 'score': 0}

    close  = df['Close']
    high   = df['High']
    low    = df['Low']

    # Encontrar máximos y mínimos locales con ventana de 5 días
    ventana = 5
    max_local = (high == high.rolling(ventana * 2 + 1, center=True).max())
    min_local = (low  == low.rolling(ventana * 2 + 1, center=True).min())

    idx_maximos = df.index[max_local].tolist()
    idx_minimos = df.index[min_local].tolist()

    if len(idx_maximos) < 2 or len(idx_minimos) < 2:
        return {'detectado': False, 'score': 0}

    # Tomar los últimos swings
    ultimos_max = idx_maximos[-n_swings:]
    ultimos_min = idx_minimos[-n_swings:]

    correcciones = []
    for i in range(min(len(ultimos_max) - 1, len(ultimos_min))):
        m_alto  = float(high.loc[ultimos_max[i]])
        m_bajo  = float(low.loc[ultimos_min[i]])
        corr_pct = (m_alto - m_bajo) / m_alto * 100
        correcciones.append(round(corr_pct, 1))

    if len(correcciones) < 2:
        return {'detectado': False, 'score': 0}

    # Verificar que cada corrección es menor que la anterior
    es_decreciente = all(correcciones[i] > correcciones[i+1]
                         for i in range(len(correcciones) - 1))

    ratio = correcciones[-1] / correcciones[0] if correcciones[0] > 0 else 1.0

    # Pivot = máximo reciente = nivel de rotura esperado
    pivot = float(high.iloc[-20:].max())

    # Días de compresión = días desde el último máximo significativo
    idx_ultimo_max = idx_maximos[-1]
    dias_compresion = len(df.loc[idx_ultimo_max:]) - 1

    # Score: mejor cuanto más decreciente y más comprimido
    score = 0
    if es_decreciente:
        score += 40
        score += max(0, int((1 - ratio) * 40))   # ratio bajo → más puntos
        if correcciones[-1] < 5:   score += 20    # última corrección muy pequeña
        elif correcciones[-1] < 8: score += 10
        if 10 <= dias_compresion <= 40: score += 10  # ventana temporal ideal

    return {
        'detectado':       es_decreciente and ratio <= max_ratio,
        'score':           min(score, 100),
        'correcciones':    correcciones,
        'ratio':           round(ratio, 2),
        'pivot':           round(pivot, 2),
        'dias_compresion': dias_compresion,
    }


def calcular_squeeze_bb_kc(df, bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5):
    """
    Squeeze de Bollinger Bands + Keltner Channel (John Carter).

    Squeeze activo cuando las BB están DENTRO del KC → volatilidad en mínimos.
    Indica que el precio está "cargando" energía para un movimiento.

    Retorna dict con:
      - 'squeeze_activo': bool — squeeze actualmente activo
      - 'dias_squeeze':   días consecutivos en squeeze
      - 'histograma':     valor del momentum (positivo = alcista)
      - 'direccion':      'alcista' / 'bajista' / 'neutral'
      - 'score':          0-100
    """
    if len(df) < bb_period + 5:
        return {'squeeze_activo': False, 'score': 0, 'dias_squeeze': 0}

    close = df['Close']
    high  = df['High']
    low   = df['Low']

    # Bollinger Bands
    bb_mid  = close.rolling(bb_period).mean()
    bb_std_s = close.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_s
    bb_lower = bb_mid - bb_std * bb_std_s

    # Keltner Channel (basado en ATR)
    atr_s   = calcular_atr_series(df, kc_period)
    kc_upper = bb_mid + kc_mult * atr_s
    kc_lower = bb_mid - kc_mult * atr_s

    # Squeeze: BB dentro de KC
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Días consecutivos en squeeze
    squeeze_actual = bool(squeeze.iloc[-1])
    dias_squeeze = 0
    if squeeze_actual:
        for i in range(len(squeeze) - 1, -1, -1):
            if squeeze.iloc[i]:
                dias_squeeze += 1
            else:
                break

    # Momentum (simplificado): distancia del close a la media del rango reciente
    lookback = 20
    max_high = high.rolling(lookback).max()
    min_low  = low.rolling(lookback).min()
    mid_hl   = (max_high + min_low) / 2
    mid_close = close.rolling(lookback).mean()
    delta    = close - (mid_hl + mid_close) / 2
    mom      = delta.ewm(span=12).mean()
    mom_val  = float(mom.iloc[-1])
    mom_prev = float(mom.iloc[-2]) if len(mom) > 1 else 0

    if mom_val > 0 and mom_val >= mom_prev:
        direccion = 'alcista'
    elif mom_val > 0 and mom_val < mom_prev:
        direccion = 'alcista_debilitando'
    elif mom_val < 0:
        direccion = 'bajista'
    else:
        direccion = 'neutral'

    # Score
    score = 0
    if squeeze_actual:
        score += 40
        score += min(30, dias_squeeze * 3)   # más días → más energía acumulada
        if direccion == 'alcista':        score += 30
        elif direccion == 'alcista_debilitando': score += 15

    return {
        'squeeze_activo': squeeze_actual,
        'dias_squeeze':   dias_squeeze,
        'histograma':     round(mom_val, 4),
        'direccion':      direccion,
        'score':          min(score, 100),
    }


def calcular_nr7(df):
    """
    Narrow Range 7 y Narrow Range 4 (Toby Crabel).

    NR7: el rango de hoy (High-Low) es el más pequeño de los últimos 7 días.
    NR4: idem para 4 días.

    Indica máxima compresión intradiaria — suele preceder a un movimiento
    direccional fuerte en los 1-3 días siguientes.

    Retorna dict con:
      - 'nr7': bool
      - 'nr4': bool
      - 'rango_hoy':    rango en % del precio
      - 'rango_medio':  rango medio 20 días en %
      - 'compresion_pct': cuánto más pequeño es el rango vs la media (%)
      - 'score': 0-100
    """
    if len(df) < 20:
        return {'nr7': False, 'nr4': False, 'score': 0}

    rango     = df['High'] - df['Low']
    rango_pct = rango / df['Close'] * 100

    rango_hoy    = float(rango_pct.iloc[-1])
    min_7        = float(rango_pct.iloc[-7:].min())
    min_4        = float(rango_pct.iloc[-4:].min())
    rango_med    = float(rango_pct.iloc[-20:].mean())

    nr7 = rango_hoy <= min_7
    nr4 = rango_hoy <= min_4
    compresion = (1 - rango_hoy / rango_med) * 100 if rango_med > 0 else 0

    score = 0
    if nr4:  score += 30
    if nr7:  score += 40
    score += min(30, int(compresion * 0.5))   # más comprimido vs media → más puntos

    return {
        'nr7':            nr7,
        'nr4':            nr4,
        'rango_hoy':      round(rango_hoy, 2),
        'rango_medio':    round(rango_med, 2),
        'compresion_pct': round(compresion, 1),
        'score':          min(score, 100),
    }


def analizar_compresion(ticker, df, df_ibex=None):
    """
    Calcula los 3 patrones de compresión para un ticker y devuelve
    un score combinado (0-100) junto con el detalle de cada patrón.

    Solo considera activos en tendencia alcista (precio > SMA200)
    para no generar falsos positivos en activos en caída.
    """
    try:
        if len(df) < 260:
            return None

        close  = float(df['Close'].iloc[-1])
        sma200 = float(df['Close'].rolling(200).mean().iloc[-1])
        sma50  = float(df['Close'].rolling(50).mean().iloc[-1])

        # Solo activos alcistas — compresión en tendencia bajista no es señal
        if close <= sma200:
            return None

        vcp = calcular_vcp(df)
        sqz = calcular_squeeze_bb_kc(df)
        nr  = calcular_nr7(df)

        # Score combinado ponderado
        # VCP pesa más porque requiere más condiciones cumplidas
        score_total = round(
            vcp['score'] * 0.45 +
            sqz['score'] * 0.35 +
            nr['score']  * 0.20
        )

        # Nivel de pivot (precio objetivo de la rotura)
        pivot = vcp.get('pivot') or float(df['High'].iloc[-20:].max())

        # RS vs IBEX
        rs_val = None
        if df_ibex is not None and not df_ibex.empty:
            try:
                ret_t = float(df['Close'].pct_change(20).iloc[-1])
                ret_i = float(df_ibex['Close'].pct_change(20)
                              .reindex(df.index, method='ffill').iloc[-1])
                rs_val = round((1 + ret_t) / (1 + ret_i), 3) if ret_i != 0 else 1.0
            except Exception:
                pass

        # Descripción del estado
        patrones_activos = []
        if vcp['detectado']:             patrones_activos.append("VCP")
        if sqz['squeeze_activo']:        patrones_activos.append("Squeeze")
        if nr['nr7']:                    patrones_activos.append("NR7")
        elif nr['nr4']:                  patrones_activos.append("NR4")

        if not patrones_activos:
            return None   # Sin patrones activos, no mostrar en la tabla

        return {
            'Ticker':         ticker.replace('.MC', ''),
            'Empresa':        NOMBRES_IBEX.get(ticker, ticker),
            'Score Compres.': score_total,
            'Patrones':       ' + '.join(patrones_activos),
            'Precio':         round(close, 2),
            'Pivot (€)':      round(pivot, 2),
            'Dist. Pivot %':  round((pivot - close) / close * 100, 1),
            'RS vs IBEX':     rs_val,
            'SMA50 ok':       '✅' if close > sma50 else '❌',
            # Detalles por patrón
            '_vcp':  vcp,
            '_sqz':  sqz,
            '_nr':   nr,
        }
    except Exception:
        return None


@st.cache_data(ttl=1800)   # Caché 30 min — los patrones de compresión cambian poco
def escanear_compresion(data_raw, ibex_df):
    """Escanea todos los tickers buscando patrones de compresión."""

    def _extraer(data, ticker):
        if not isinstance(data.columns, pd.MultiIndex):
            return data.copy().dropna()
        if ticker in data.columns.get_level_values(0).unique():
            return data[ticker].copy().dropna()
        return data.xs(ticker, axis=1, level=1).copy().dropna()

    resultados = []
    for ticker in IBEX35_TICKERS:
        try:
            df = _extraer(data_raw, ticker)
            if df.empty or len(df) < 60:
                continue
            res = analizar_compresion(ticker, df, ibex_df)
            if res:
                resultados.append(res)
        except Exception:
            pass
    return sorted(resultados, key=lambda x: x['Score Compres.'], reverse=True)


# =============================================================================
# 2. ALERTAS INTRADIARIAS — detecta roturas en formación durante la sesión
# =============================================================================
# Problema que resuelve: el scanner diario detecta roturas al CIERRE del día.
# Si la rotura ocurrió a las 10:30 y el usuario mira el scanner a las 17:30,
# entra con el movimiento ya hecho y el R:R destruido (como pasó con Endesa).
#
# Solución: descargar datos de 5d en intervalo 15min para cada ticker y
# detectar si el precio está rompiendo el máximo de 20 días AHORA, durante
# la sesión actual, con cuánto lleva subido desde el trigger y si el volumen
# de la sesión parcial ya supera el umbral.

def _extraer_ticker_df(data, ticker):
    """
    Extrae el DataFrame de un ticker desde un yf.download() multi-ticker.
    Compatible con ambos formatos de MultiIndex:
      - yfinance <0.2.38:  nivel 0 = Ticker, nivel 1 = Price  → data['ITX.MC']
      - yfinance ≥0.2.38:  nivel 0 = Price,  nivel 1 = Ticker → data.xs('ITX.MC', level=1)
    """
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy().dropna()
    if ticker in data.columns.get_level_values(0).unique():
        return data[ticker].copy().dropna()          # formato antiguo
    return data.xs(ticker, axis=1, level=1).copy().dropna()  # formato nuevo


@st.cache_data(ttl=300)  # Refresca cada 5 minutos — datos intradiarios
def escanear_alertas_intradiarias(tickers, vol_threshold=1.5):
    """
    Para cada ticker, descarga datos intradiarios (15min, 5 días) y comprueba
    si el precio actual está cruzando el máximo de 20 días diarios.

    Retorna lista de dicts con las alertas activas ordenadas por urgencia.
    """
    df_diario_ref = yf.download(tickers, period="30d", group_by='ticker', progress=False)

    alertas = []
    for ticker in tickers:
        try:
            df_d = _extraer_ticker_df(df_diario_ref, ticker)
            if len(df_d) < 21:
                continue

            nivel_rotura = df_d['High'].iloc[:-1].tail(20).max()

            df_intra = yf.download(ticker, period="5d", interval="15m", progress=False)
            if df_intra.empty:
                continue
            # Para descarga de un solo ticker con intervalo, yfinance a veces
            # devuelve MultiIndex y a veces no — _extraer_ticker_df lo maneja
            if isinstance(df_intra.columns, pd.MultiIndex):
                try:
                    df_intra = _extraer_ticker_df(df_intra, ticker)
                except Exception:
                    df_intra = df_intra.droplevel(1, axis=1) if df_intra.columns.nlevels > 1 else df_intra
            df_intra = df_intra.dropna()

            precio_actual = df_intra['Close'].iloc[-1]
            dist_pct      = (precio_actual - nivel_rotura) / nivel_rotura * 100

            vol_hoy = df_d['Volume'].iloc[-1] if len(df_d) > 0 else 0
            vol_med = df_d['Volume'].iloc[:-1].tail(20).mean()
            vol_ratio_hoy = vol_hoy / vol_med if vol_med > 0 else 0

            if -0.5 <= dist_pct <= 5.0:
                if dist_pct < 0:
                    estado = "🔔 Acercándose"; urgencia = 3
                elif dist_pct <= 1.5:
                    estado = "🚨 Rotura Fresca"; urgencia = 1
                elif dist_pct <= 3.0:
                    estado = "⚠️ Entrada Tardía"; urgencia = 2
                else:
                    estado = "🚫 Muy Tarde"; urgencia = 4

                alertas.append({
                    "Ticker":        ticker.replace(".MC", ""),
                    "Empresa":       NOMBRES_IBEX.get(ticker, ticker),
                    "Estado":        estado,
                    "Precio Actual": round(float(precio_actual), 2),
                    "Nivel Rotura":  round(float(nivel_rotura), 2),
                    "Dist. %":       round(float(dist_pct), 2),
                    "Vol Hoy/Med":   round(float(vol_ratio_hoy), 2),
                    "_urgencia":     urgencia
                })
        except Exception:
            continue

    alertas.sort(key=lambda x: x['_urgencia'])
    return alertas


# =============================================================================
# 3. FILTRO DE RÉGIMEN DE MERCADO (IBEX INDEX)
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
            return True, None
        # Aplanar MultiIndex independientemente del formato
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Eliminar timezone del índice para compatibilidad con resample
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        sma200 = df['Close'].rolling(200).mean()
        regime_bullish = bool(df['Close'].iloc[-1] > sma200.iloc[-1])
        return regime_bullish, df
    except:
        return True, None

# =============================================================================
# 3. GMAIL — función auxiliar de envío
# =============================================================================

def _gmail_send(gmail_user: str, gmail_app_password: str,
                destinatario: str, asunto: str, cuerpo_html: str) -> bool:
    """
    Envía un email HTML via Gmail SMTP (TLS, puerto 587).
    Requiere una App Password de Google (no la contraseña habitual):
      Cuenta Google → Seguridad → Verificación en 2 pasos → Contraseñas de aplicación
    Retorna True si el envío tiene éxito.
    """
    import smtplib
    import ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = asunto
        msg['From']    = gmail_user
        msg['To']      = destinatario
        msg.attach(MIMEText(cuerpo_html, 'html'))
        context = ssl.create_default_context()
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(gmail_user, gmail_app_password)
            server.sendmail(gmail_user, destinatario, msg.as_string())
        return True
    except Exception:
        return False


def _gmail_formatear_senal(empresa, ticker, precio, trigger, dist_pct,
                            stop, target, riesgo_pct, vol_ratio, rsi_pre,
                            rs_val, adx, rr, score) -> tuple[str, str]:
    """Devuelve (asunto, cuerpo_html) con la señal detectada en el scanner."""
    calidad = "✅ Óptima" if dist_pct <= 1.5 else "⚠️ Tardía"
    rr_str  = f"3R 🔥 (ADX fuerte)" if rr >= 3.0 else f"{rr:.1f}R"
    rs_str  = f"<br>📈 RS vs IBEX: <b>{rs_val:.3f}</b>" if rs_val else ""
    asunto  = f"🚨 Señal IBEX 35 — {empresa} ({ticker}) · Score {score}/100"
    cuerpo  = f"""
    <html><body style="font-family:Arial,sans-serif;background:#1a1a1a;color:#f0f0f0;padding:20px">
    <h2 style="color:#00ff88">🚨 SEÑAL DE COMPRA — {empresa} ({ticker})</h2>
    <p style="font-size:18px">🏆 Score: <b style='color:#ffcc00'>{score}/100</b></p>
    <hr style="border-color:#444">
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="padding:6px">💰 Precio actual</td><td><b>{precio:.2f} €</b></td></tr>
      <tr><td style="padding:6px">🎯 Trigger (rotura)</td>
          <td><b>{trigger:.2f} €</b> <span style="color:#aaa">(+{dist_pct:.1f}% — {calidad})</span></td></tr>
      <tr><td style="padding:6px">🛑 Stop Loss</td>
          <td><b style='color:#ff4444'>{stop:.2f} €</b> <span style="color:#aaa">({riesgo_pct:.1f}% riesgo)</span></td></tr>
      <tr><td style="padding:6px">🏁 Target</td>
          <td><b style='color:#00ff88'>{target:.2f} €</b> <span style="color:#aaa">(R:R {rr_str})</span></td></tr>
    </table>
    <hr style="border-color:#444">
    <p>📊 Vol x{vol_ratio:.1f} &nbsp;|&nbsp; RSI pre-rotura {rsi_pre:.0f} &nbsp;|&nbsp; ADX {adx:.0f}{rs_str}</p>
    <p style="color:#888;font-size:12px">⚠️ No es asesoramiento financiero.</p>
    </body></html>
    """
    return asunto, cuerpo


# =============================================================================
# 4. SCORING (LÓGICA PRINCIPAL)
# =============================================================================

def analizar_ticker(ticker, df_diario, config, market_bullish=True, df_ibex=None):
    # No capturamos excepciones aquí — el scanner las recoge y las muestra
    if len(df_diario) < 260:
        return None

    # Eliminar timezone del índice — yfinance nuevo devuelve UTC,
    # lo que rompe resample('W') al comparar con índices sin tz
    if hasattr(df_diario.index, 'tz') and df_diario.index.tz is not None:
        df_diario = df_diario.copy()
        df_diario.index = df_diario.index.tz_localize(None)

    # Normalizar nombres de columna (por si vienen en minúsculas)
    df_diario.columns = [c.capitalize() if c.lower() in
                         ['open','high','low','close','volume'] else c
                         for c in df_diario.columns]

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
    close  = df_diario['Close'].iloc[-1]
    sma50  = df_diario['Close'].rolling(50).mean().iloc[-1]
    sma200 = df_diario['Close'].rolling(200).mean().iloc[-1]

    # RSI completo — para información y scoring de contexto
    rsi_series  = calcular_rsi(df_diario['Close'], 14)
    rsi_actual  = rsi_series.iloc[-1]

    # RSI PRE-ROTURA: media de los 5 días anteriores al día actual
    # Mide si el activo estaba "descansando" antes del impulso actual
    rsi_pre     = calcular_rsi_prebreakout(rsi_series, ventana=5).iloc[-1]

    atr = calcular_atr_series(df_diario, 14).iloc[-1]
    adx, plus_di, minus_di = calcular_adx(df_diario, 14)
    adx_val      = adx.iloc[-1]
    plus_di_val  = plus_di.iloc[-1]
    minus_di_val = minus_di.iloc[-1]

    # --- VOLUMEN ---
    obv         = calcular_obv(df_diario)
    obv_sma     = obv.rolling(20).mean()
    obv_bullish = obv.iloc[-1] > obv_sma.iloc[-1]
    vol_ratio   = calcular_volume_ratio(df_diario).iloc[-1]

    # --- FORTALEZA RELATIVA VS IBEX ---
    rs_val      = None
    rs_bullish  = False
    rs_20d      = None
    if df_ibex is not None and not df_ibex.empty:
        rs_series  = calcular_rs_vs_ibex(df_diario, df_ibex, period=20)
        rs_val     = rs_series.iloc[-1]
        rs_bullish = bool(rs_val > 1.0) if rs_val is not None and not np.isnan(rs_val) else False
        rs_20d     = rs_val

    # --- DISPARADOR DE ENTRADA ---
    vol_thr = config.get('vol_threshold', 1.5)
    breakout_series, prev_high_series, vol_ratio_series = calcular_breakout_signal(df_diario, vol_threshold=vol_thr)
    breakout_today = breakout_series.iloc[-1]
    prev_high_val  = prev_high_series.iloc[-1]

    # --- SCORING ---
    # ┌─────────────────────────────────────────────────────────────┐
    # │  DISTRIBUCIÓN DE PUNTOS — MÁXIMO POSITIVO EXACTO: 100 pts  │
    # ├────────────────────────────┬────────────┬───────────────────┤
    # │ Regla                      │ Positivo   │ Negativo          │
    # ├────────────────────────────┼────────────┼───────────────────┤
    # │ 1. Tendencia semanal EMA30 │ +20        │ -15               │
    # │ 2. SMA200 (soporte mayor)  │ +20        │ -15               │
    # │ 3. SMA50  (tend. media)    │ +10        │   0               │
    # │ 4. ADX + DI direccional    │ +10        │ -10               │
    # │ 5. RSI Pre-Rotura          │ +10        │  -8               │
    # │ 6. OBV presión vol.        │ +10        │  -5               │
    # │ 7. RS vs IBEX              │ +10        │  -8               │
    # │ 8. Zona valor (SMA50)      │  +5        │   0               │
    # │ 9. Disparador de entrada   │ +10        │   0               │
    # ├────────────────────────────┼────────────┼───────────────────┤
    # │ TOTAL MÁXIMO               │ 100        │                   │
    # │ Filtro régimen (no cuenta) │            │ -20 (penalización)│
    # └────────────────────────────┴────────────┴───────────────────┘
    # El filtro de régimen de mercado es una penalización externa al
    # sistema de 100 puntos — puede llevar el score por debajo de 0
    # en mercados bajistas, lo cual es correcto e intencionado.

    score     = 0
    score_log = []

    # FILTRO MAESTRO: Régimen de mercado (penalización externa, no suma al máximo)
    if not market_bullish:
        score -= 20
        score_log.append({
            "Regla": "🌐 Régimen de Mercado (IBEX SMA200)",
            "Valor": "⚠️ BAJISTA",
            "Puntos": "-20",
            "Detalle": "IBEX 35 por debajo de SMA200 — Sistema en modo DEFENSIVO"
        })
    else:
        score_log.append({
            "Regla": "🌐 Régimen de Mercado (IBEX SMA200)",
            "Valor": "✅ Alcista",
            "Puntos": "0",
            "Detalle": "IBEX 35 por encima de SMA200 — Generación de señales activa"
        })

    # 1. Tendencia semanal — +20 / -15
    if weekly_trend_bullish:
        pts = 20; score += pts
        score_log.append({"Regla": "Tendencia Semanal (EMA30)", "Valor": "🟢 Alcista", "Puntos": f"+{pts}", "Detalle": f"Precio ({weekly_close:.2f}) > EMA30 ({weekly_ema30:.2f})"})
    else:
        pts = -15; score += pts
        score_log.append({"Regla": "Tendencia Semanal (EMA30)", "Valor": "🔴 Bajista", "Puntos": f"{pts}", "Detalle": f"Precio ({weekly_close:.2f}) < EMA30 ({weekly_ema30:.2f})"})

    # 2. SMA200 — +20 / -15
    if close > sma200:
        pts = 20; score += pts
        score_log.append({"Regla": "Soporte Mayor (SMA200)", "Valor": "✅ Sobre soporte", "Puntos": f"+{pts}", "Detalle": f"Precio ({close:.2f}) > SMA200 ({sma200:.2f})"})
    else:
        pts = -15; score += pts
        score_log.append({"Regla": "Soporte Mayor (SMA200)", "Valor": "❌ Bajo soporte", "Puntos": f"{pts}", "Detalle": f"Precio ({close:.2f}) < SMA200 ({sma200:.2f})"})

    # 3. SMA50 — +10 / 0
    if close > sma50:
        pts = 10; score += pts
        score_log.append({"Regla": "Tendencia Mediano Plazo (SMA50)", "Valor": "✅ Alcista", "Puntos": f"+{pts}", "Detalle": f"Precio ({close:.2f}) > SMA50 ({sma50:.2f})"})
    else:
        score_log.append({"Regla": "Tendencia Mediano Plazo (SMA50)", "Valor": "❌ Bajista", "Puntos": "0", "Detalle": f"Precio ({close:.2f}) < SMA50 ({sma50:.2f})"})

    # 4. ADX + DI — +10 / -10 / -5
    if adx_val > 25 and plus_di_val > minus_di_val:
        pts = 10; score += pts
        score_log.append({"Regla": "Fuerza Direccional (ADX + DI+)", "Valor": "💪 Fuerte Alcista", "Puntos": f"+{pts}", "Detalle": f"ADX ({adx_val:.1f}) > 25 y DI+ ({plus_di_val:.1f}) > DI- ({minus_di_val:.1f})"})
    elif adx_val > 25 and plus_di_val < minus_di_val:
        pts = -10; score += pts
        score_log.append({"Regla": "Fuerza Direccional (ADX + DI-)", "Valor": "⚠️ Fuerte Bajista", "Puntos": f"{pts}", "Detalle": f"ADX ({adx_val:.1f}) > 25 pero DI- ({minus_di_val:.1f}) domina"})
    elif adx_val < 20:
        pts = -5; score += pts
        score_log.append({"Regla": "Fuerza de Tendencia (ADX)", "Valor": "😴 Débil/Lateral", "Puntos": f"{pts}", "Detalle": f"ADX ({adx_val:.1f}) < 20 — Mercado sin dirección"})
    else:
        score_log.append({"Regla": "Fuerza de Tendencia (ADX)", "Valor": "🟡 Neutral", "Puntos": "0", "Detalle": f"ADX ({adx_val:.1f}) en zona neutra (20-25)"})

    # 5. RSI Pre-Rotura — +10 / -8 / -5 / 0
    if not np.isnan(rsi_pre):
        if 45 <= rsi_pre <= 68:
            pts = 10; score += pts
            score_log.append({"Regla": "RSI Pre-Rotura (5d previos)", "Valor": "🎯 Acumulación Sana", "Puntos": f"+{pts}", "Detalle": f"RSI medio 5d previos ({rsi_pre:.1f}) en zona óptima 45-68"})
        elif rsi_pre > 78:
            pts = -8; score += pts
            score_log.append({"Regla": "RSI Pre-Rotura (5d previos)", "Valor": "🔴 Ya Agotado", "Puntos": f"{pts}", "Detalle": f"RSI previo ({rsi_pre:.1f}) > 78 — llegó a la rotura sobrecomprado"})
        elif rsi_pre < 40:
            pts = -5; score += pts
            score_log.append({"Regla": "RSI Pre-Rotura (5d previos)", "Valor": "⚠️ Momentum Débil", "Puntos": f"{pts}", "Detalle": f"RSI previo ({rsi_pre:.1f}) < 40 — poco momentum antes de la rotura"})
        else:
            score_log.append({"Regla": "RSI Pre-Rotura (5d previos)", "Valor": "🟡 Neutral", "Puntos": "0", "Detalle": f"RSI previo ({rsi_pre:.1f}) — zona aceptable"})
        score_log.append({"Regla": "  ↳ RSI Actual (informativo)", "Valor": "—", "Puntos": "0", "Detalle": f"RSI hoy: {rsi_actual:.1f} — esperado alto en día de rotura, no penaliza"})
    else:
        score_log.append({"Regla": "RSI Pre-Rotura", "Valor": "⚪ Sin datos", "Puntos": "0", "Detalle": "Datos insuficientes"})

    # 6. OBV — +10 / -5
    if obv_bullish:
        pts = 10; score += pts
        score_log.append({"Regla": "Presión de Volumen (OBV)", "Valor": "🟢 Compradora", "Puntos": f"+{pts}", "Detalle": "OBV por encima de su SMA20 — dinero entrando"})
    else:
        pts = -5; score += pts
        score_log.append({"Regla": "Presión de Volumen (OBV)", "Valor": "🔴 Vendedora", "Puntos": f"{pts}", "Detalle": "OBV por debajo de su SMA20 — distribución"})

    # 7. Fortaleza Relativa vs IBEX — +10 / +5 / 0 / -8
    if rs_val is not None and not np.isnan(rs_val):
        if rs_val > 1.05:
            pts = 10; score += pts
            score_log.append({"Regla": "💪 Fortaleza Relativa vs IBEX", "Valor": "🟢 Líder de Mercado", "Puntos": f"+{pts}", "Detalle": f"RS 20d = {rs_val:.3f} — sube {(rs_val-1)*100:.1f}% más que el IBEX"})
        elif rs_val > 1.0:
            pts = 5; score += pts
            score_log.append({"Regla": "💪 Fortaleza Relativa vs IBEX", "Valor": "🟡 Supera al Índice", "Puntos": f"+{pts}", "Detalle": f"RS 20d = {rs_val:.3f} — ligeramente por encima del IBEX"})
        elif rs_val > 0.95:
            score_log.append({"Regla": "💪 Fortaleza Relativa vs IBEX", "Valor": "⚪ En línea con Índice", "Puntos": "0", "Detalle": f"RS 20d = {rs_val:.3f} — movimiento similar al IBEX"})
        else:
            pts = -8; score += pts
            score_log.append({"Regla": "💪 Fortaleza Relativa vs IBEX", "Valor": "🔴 Rezagado", "Puntos": f"{pts}", "Detalle": f"RS 20d = {rs_val:.3f} — cae más (o sube menos) que el IBEX"})
    else:
        score_log.append({"Regla": "💪 Fortaleza Relativa vs IBEX", "Valor": "⚪ Sin datos IBEX", "Puntos": "0", "Detalle": "No se pudo calcular sin datos del índice"})

    # 8. Zona de Valor — +5 / 0
    dist_sma50 = (close - sma50) / sma50
    if 0 < dist_sma50 < 0.05:
        pts = 5; score += pts
        score_log.append({"Regla": "Zona de Valor (Pullback SMA50)", "Valor": "🎯 Oportunidad", "Puntos": f"+{pts}", "Detalle": f"Precio dentro del 5% de SMA50 — zona de rebote potencial"})

    # 9. Disparador de entrada — +10 / 0
    if breakout_today:
        pts = 10; score += pts
        score_log.append({"Regla": "🚨 Disparador de Entrada", "Valor": "✅ ROTURA ACTIVA", "Puntos": f"+{pts}", "Detalle": f"Cierre ({close:.2f}€) > Máx. 20 días ({prev_high_val:.2f}€) con volumen x{vol_ratio:.1f}"})
    else:
        score_log.append({"Regla": "🚨 Disparador de Entrada", "Valor": "⏳ Sin rotura", "Puntos": "0", "Detalle": f"Precio aún no supera máx. 20 días ({prev_high_val:.2f}€)"})

    # 10. CALIDAD DE ENTRADA — distancia al nivel de rotura exacto
    # Problema: el scanner se ejecuta al cierre. Si la rotura ocurrió a media
    # sesión, el precio puede haber subido un 3-4% por encima del trigger.
    # Entrar ahí destruye el R:R porque el recorrido ya está parcialmente hecho.
    #
    # Regla:
    #   ≤ 1.5% sobre el máximo de rotura → entrada válida, sin penalización
    #   1.5% - 3.0%                        → entrada tardía, penaliza -10 pts y alerta
    #   > 3.0%                             → entrada muy tardía, penaliza -20 pts y alerta roja
    #
    # La distancia se mide respecto al prev_high_val (el nivel de rotura),
    # no respecto al precio de ayer, para ser preciso sobre cuánto se ha
    # "comido" el movimiento desde el trigger.
    if prev_high_val and prev_high_val > 0:
        dist_entrada_pct = (close - prev_high_val) / prev_high_val * 100
    else:
        dist_entrada_pct = 0.0

    if dist_entrada_pct <= 1.5:
        entrada_calidad = "✅ Óptima"
        score_log.append({
            "Regla": "📐 Calidad de Entrada",
            "Valor": f"✅ Óptima (+{dist_entrada_pct:.1f}% sobre trigger)",
            "Puntos": "0",
            "Detalle": f"Precio ({close:.2f}€) está a solo {dist_entrada_pct:.1f}% del nivel de rotura ({prev_high_val:.2f}€). Entrada con R:R intacto."
        })
    elif dist_entrada_pct <= 3.0:
        pts = -10; score += pts
        entrada_calidad = "⚠️ Tardía"
        score_log.append({
            "Regla": "📐 Calidad de Entrada",
            "Valor": f"⚠️ Tardía (+{dist_entrada_pct:.1f}% sobre trigger)",
            "Puntos": f"{pts}",
            "Detalle": f"Precio ({close:.2f}€) está {dist_entrada_pct:.1f}% por encima del trigger ({prev_high_val:.2f}€). El movimiento está parcialmente hecho. Considera esperar un pullback a {prev_high_val:.2f}€."
        })
    else:
        pts = -20; score += pts
        entrada_calidad = "🚫 Muy Tardía"
        score_log.append({
            "Regla": "📐 Calidad de Entrada",
            "Valor": f"🚫 Muy Tardía (+{dist_entrada_pct:.1f}% sobre trigger)",
            "Puntos": f"{pts}",
            "Detalle": f"Precio ({close:.2f}€) está {dist_entrada_pct:.1f}% por encima del trigger ({prev_high_val:.2f}€). ENTRADA INVÁLIDA — el recorrido está hecho. Espera retroceso o descarta."
        })

    # Verificación: el score nunca debe superar 100 en positivo
    # (puede bajar de 0 por penalizaciones de régimen y señales bajistas)
    score = min(score, 100)

    # --- GESTIÓN DE RIESGO ---
    stop_loss      = close - (atr * config['atr_mult'])
    risk_per_share = close - stop_loss
    risk_pct       = (risk_per_share / close) * 100
    target_fixed   = close + (risk_per_share * config['rr_ratio'])
    high_52w       = df_diario['High'].rolling(252).max().iloc[-1]

    # Cap de target: solo aplicar si el máximo de 52 semanas está
    # SIGNIFICATIVAMENTE por encima del precio actual (>5% de margen).
    # Si el activo ya cotiza cerca o en máximos anuales (rotura real),
    # el cap provocaría un target por debajo del precio de entrada — absurdo.
    margen_52w = (high_52w - close) / close if close > 0 else 0
    if margen_52w > 0.05 and target_fixed > high_52w * 0.95:
        target = min(target_fixed, high_52w * 0.98)
    else:
        target = target_fixed

    # Salvaguarda final: el target NUNCA puede ser menor que entrada + 0.5R
    # Cubre cualquier edge case (datos erróneos, activos en máximos históricos, etc.)
    target = max(target, close + risk_per_share * 0.5)

    return {
        "Ticker":           ticker.replace(".MC", ""),
        "Empresa":          NOMBRES_IBEX.get(ticker, ticker),
        "Precio":           close,
        "Score":            round(score),
        "Tendencia Semanal": "🟢 Alcista" if weekly_trend_bullish else "🔴 Bajista",
        "Régimen Mercado":  "🟢 Alcista" if market_bullish else "🔴 Bajista",
        "SMA200":           sma200,
        "SMA50":            sma50,
        "ADX":              adx_val,
        "DI+":              plus_di_val,
        "DI-":              minus_di_val,
        "RSI Actual":       rsi_actual,
        "RSI Pre-Rotura":   round(rsi_pre, 1) if not np.isnan(rsi_pre) else None,
        "RS vs IBEX":       round(rs_val, 3) if rs_val and not np.isnan(rs_val) else None,
        "OBV Alcista":      obv_bullish,
        "Vol Ratio":        vol_ratio,
        "Rotura":           "🚨 SÍ" if breakout_today else "—",
        "Entrada":          entrada_calidad,
        "Dist. Trigger %":  round(dist_entrada_pct, 1),
        "Trigger (€)":      round(prev_high_val, 2) if prev_high_val else None,
        "Stop Loss":        stop_loss,
        "Target":           target,
        "Riesgo %":         risk_pct,
        "ATR":              atr,
        "Score_Log":        score_log,
        "_df":              df_diario
    }

# =============================================================================
# 4. BACKTESTING VECTORIZADO v4 — Profit Factor Maximizado
# =============================================================================
#
# MEJORAS ACTIVAS (acumulativas sobre versiones anteriores):
#
# [A] SALIDA PARCIAL 1R + STOP BREAKEVEN
#     50% cierra en 1R, stop restante → precio entrada. Elimina pérdidas
#     en trades que "casi llegan" y vuelven.
#
# [B] FILTRO ATR MÍNIMO (configurable)
#     Descarta activos inertes que raramente alcanzan el target.
#
# [C] UMBRAL DE VOLUMEN CONFIGURABLE
#     Parámetro ajustable desde sidebar.
#
# [D] TARGET DINÁMICO SEGÚN ADX
#     ADX > 35 en entrada → target 3R en vez de 2R.
#
# [E] RSI PRE-ROTURA (v3)
#     Evalúa RSI de los 5 días anteriores, no el del día de rotura.
#     Elimina la contradicción RSI-alto vs precio-rompiendo-máximos.
#
# [F] FORTALEZA RELATIVA VS IBEX (v3)
#     Sólo entra si el activo supera al índice en los últimos 20 días.
#     Filtra roturas por inercia de mercado (baja calidad).
#
# [G] CONFIRMACIÓN SEMANAL DE ROTURA (NUEVO)
#     La rotura diaria debe coincidir con que el precio también supere
#     el máximo de las últimas 8 semanas en el cierre semanal más reciente.
#     Elimina roturas falsas que son ruido dentro de un rango lateral mayor.
#
# [H] LÍMITE DE CONCENTRACIÓN SECTORIAL (NUEVO)
#     Máximo 1 posición abierta por sector simultáneamente.
#     Los bancos del IBEX están altamente correlacionados: abrir BBVA,
#     Santander y CaixaBank a la vez es en realidad una sola apuesta triple.
#     Este filtro reduce el riesgo concentrado sin reducir el número de señales
#     totales a largo plazo (simplemente las escala en el tiempo).
#
# [I] STOP MÍNIMO GARANTIZADO (NUEVO)
#     El stop trailing nunca sube tan rápido que deje menos de 0.5R de
#     espacio al precio actual. Evita que el trailing agresivo cierre
#     trades ganadores en retrocesos normales antes de llegar al target.

# Mapa de sectores para filtro de concentración
SECTOR_MAP = {}
for sector, tickers in {
    "Utilities":   ["IBE.MC","ELE.MC","REP.MC","NTGY.MC","ENG.MC","RED.MC","ANE.MC","SLR.MC"],
    "Bancos":      ["BBVA.MC","SAN.MC","CABK.MC","SAB.MC","BKT.MC","UNI.MC","MAP.MC"],
    "Industria":   ["ACS.MC","FER.MC","ANA.MC","SCYR.MC","IDR.MC","MTS.MC","FDR.MC","ACX.MC"],
    "Consumo":     ["ITX.MC","ROVI.MC","PUIG.MC","LOG.MC","GRF.MC","IAG.MC","AMS.MC"],
    "RealEstate":  ["TEF.MC","MRL.MC","COL.MC","CLNX.MC"],
    "Aeropuertos": ["AENA.MC"],
}.items():
    for t in tickers:
        SECTOR_MAP[t] = sector


def run_backtest(df_raw, config, market_regime_df=None):
    """
    Backtesting multi-activo con gestión de cartera real:
    - Simula todas las señales de todos los tickers en paralelo cronológico
    - Aplica filtro de concentración sectorial (máx 1 posición por sector)
    - Todas las demás mejoras A-I activas
    """
    vol_threshold  = config.get('vol_threshold', 1.5)
    atr_min_pct    = config.get('atr_min_pct', 1.0)
    partial_exit   = config.get('partial_exit', True)
    adx_dynamic_rr = config.get('adx_dynamic_rr', True)
    sector_limit   = config.get('sector_limit', True)   # [H]
    weekly_confirm = config.get('weekly_confirm', True)  # [G]

    # Detectar tickers disponibles en cualquier formato de MultiIndex
    if isinstance(df_raw.columns, pd.MultiIndex):
        lvl0 = df_raw.columns.get_level_values(0).unique().tolist()
        lvl1 = df_raw.columns.get_level_values(1).unique().tolist()
        # Formato nuevo (≥0.2.38): tickers en nivel 1
        available = [t for t in lvl1 if t in IBEX35_TICKERS] or \
                    [t for t in lvl0 if t in IBEX35_TICKERS]
    else:
        available = IBEX35_TICKERS
    tickers = available

    # ── PASO 1: Precalcular indicadores y señales para todos los tickers ──────
    ticker_data = {}   # ticker → DataFrame con señales

    for ticker in tickers:
        try:
            df = _extraer_ticker_df(df_raw, ticker).copy().dropna()
            if len(df) < 300:
                continue

            df['SMA200']   = df['Close'].rolling(200).mean()
            df['SMA50']    = df['Close'].rolling(50).mean()
            df['RSI']      = calcular_rsi(df['Close'], 14)
            df['ATR']      = calcular_atr_series(df, 14)
            df['ATR_PCT']  = (df['ATR'] / df['Close']) * 100
            df['OBV']      = calcular_obv(df)
            df['OBV_SMA']  = df['OBV'].rolling(20).mean()
            df['VolRatio'] = calcular_volume_ratio(df)
            df['RSI_PRE']  = calcular_rsi_prebreakout(df['RSI'], ventana=5)

            adx_s, _, _    = calcular_adx(df, 14)
            df['ADX']      = adx_s

            # EMA30 semanal
            df_w = df.resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            df_w['EMA30']       = df_w['Close'].ewm(span=30).mean()
            df_w['WeeklyTrend'] = df_w['Close'] > df_w['EMA30']

            # [G] Confirmación semanal: cierre semanal > máximo de las 8 semanas anteriores
            df_w['WeekHigh8']       = df_w['High'].shift(1).rolling(8).max()
            df_w['WeeklyBreakout']  = df_w['Close'] > df_w['WeekHigh8']

            weekly_trend   = df_w['WeeklyTrend'].resample('D').ffill()
            weekly_bo      = df_w['WeeklyBreakout'].resample('D').ffill()
            df['WeeklyTrend']    = weekly_trend.reindex(df.index, method='ffill')
            df['WeeklyBreakout'] = weekly_bo.reindex(df.index, method='ffill')

            # Régimen IBEX
            if market_regime_df is not None and not market_regime_df.empty:
                ibex_sma200    = market_regime_df['Close'].rolling(200).mean()
                ibex_regime    = market_regime_df['Close'] > ibex_sma200
                df['MarketBullish'] = ibex_regime.reindex(df.index, method='ffill').fillna(True)

                # [F] RS vs IBEX
                rs = calcular_rs_vs_ibex(df, market_regime_df, period=20)
                df['RS_IBEX'] = rs
            else:
                df['MarketBullish'] = True
                df['RS_IBEX']       = 1.0

            # Señal de rotura diaria
            df['PrevHigh20'] = df['High'].shift(1).rolling(20).max()

            # Señal de entrada completa
            cond_base = (
                (df['Close'] > df['PrevHigh20']) &
                (df['VolRatio'] > vol_threshold) &
                (df['WeeklyTrend'] == True) &
                (df['Close'] > df['SMA200']) &
                (df['RSI_PRE'] > 40) & (df['RSI_PRE'] < 75) &  # [E]
                (df['RS_IBEX'] >= 1.0) &                        # [F]
                (df['OBV'] > df['OBV_SMA']) &
                (df['MarketBullish'] == True) &
                (df['ATR_PCT'] >= atr_min_pct)                  # [B]
            )

            if weekly_confirm:
                cond_base = cond_base & (df['WeeklyBreakout'] == True)  # [G]

            df['Entry_Signal'] = cond_base
            ticker_data[ticker] = df

        except Exception:
            continue

    if not ticker_data:
        return pd.DataFrame(), pd.DataFrame()

    # ── PASO 2: Simulación cronológica multi-activo con control de cartera ───
    # Construir índice de fechas unificado
    all_dates = sorted(set().union(*[set(df.index) for df in ticker_data.values()]))

    all_trades_list = []

    # Estado de posiciones abiertas: ticker → dict con info del trade
    open_positions = {}
    # Control de sectores ocupados: sector → ticker ocupando
    sector_occupied = {}

    for date in all_dates:
        # ── Gestionar posiciones abiertas ──────────────────────────────────
        tickers_to_close = []

        for tkr, pos in open_positions.items():
            df_tkr = ticker_data[tkr]
            if date not in df_tkr.index:
                continue

            close_price = df_tkr.loc[date, 'Close']
            atr_now     = df_tkr.loc[date, 'ATR']
            adx_now     = df_tkr.loc[date, 'ADX'] if 'ADX' in df_tkr.columns else 25

            # Trailing stop
            new_stop = close_price - (atr_now * config['atr_mult'])

            # [I] Stop mínimo garantizado: nunca a menos de 0.5R del precio actual
            # Evita que el trailing sea tan agresivo que cierre en retrocesos normales
            min_stop_distance = pos['initial_risk'] * 0.5
            new_stop = min(new_stop, close_price - min_stop_distance)

            if pos['breakeven_active']:
                pos['stop'] = max(pos['stop'], new_stop, pos['entry_price'])
            else:
                pos['stop'] = max(pos['stop'], new_stop)

            # [A] Salida parcial en 1R
            if partial_exit and not pos['breakeven_active'] and close_price >= pos['target_partial']:
                pos['breakeven_active'] = True
                pos['stop'] = pos['entry_price']

                pnl_mitad = (pos['target_partial'] - pos['entry_price']) / pos['entry_price'] * 100
                duration  = (date - pos['entry_date']).days
                all_trades_list.append({
                    "Ticker":         tkr.replace(".MC", ""),
                    "Empresa":        NOMBRES_IBEX.get(tkr, tkr),
                    "Sector":         SECTOR_MAP.get(tkr, "General"),
                    "Entrada":        pos['entry_date'].date(),
                    "Salida":         date.date(),
                    "Precio Entrada": round(pos['entry_price'], 2),
                    "Precio Salida":  round(pos['target_partial'], 2),
                    "Resultado":      "✅ Parcial (1R)",
                    "PnL %":          round(pnl_mitad, 2),
                    "Duración (días)": duration,
                    "Tipo":           "1ª mitad"
                })
                continue  # la 2ª mitad sigue abierta

            # Salida final
            hit_stop   = close_price <= pos['stop']
            hit_target = close_price >= pos['target_full']

            if hit_stop or hit_target:
                exit_price = pos['stop'] if hit_stop else pos['target_full']
                pnl_pct    = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                duration   = (date - pos['entry_date']).days

                if pos['breakeven_active']:
                    if hit_target:
                        resultado = "✅ Target (2ª mitad)"
                    elif abs(exit_price - pos['entry_price']) < 0.001:
                        resultado = "⚪ Breakeven (2ª mitad)"
                    else:
                        resultado = "❌ Stop (2ª mitad)"
                    tipo = "2ª mitad"
                else:
                    resultado = "✅ Profit" if hit_target else "❌ Stop"
                    tipo = "Completa"

                all_trades_list.append({
                    "Ticker":         tkr.replace(".MC", ""),
                    "Empresa":        NOMBRES_IBEX.get(tkr, tkr),
                    "Sector":         SECTOR_MAP.get(tkr, "General"),
                    "Entrada":        pos['entry_date'].date(),
                    "Salida":         date.date(),
                    "Precio Entrada": round(pos['entry_price'], 2),
                    "Precio Salida":  round(exit_price, 2),
                    "Resultado":      resultado,
                    "PnL %":          round(pnl_pct, 2),
                    "Duración (días)": duration,
                    "Tipo":           tipo
                })
                tickers_to_close.append(tkr)

        for tkr in tickers_to_close:
            sector = SECTOR_MAP.get(tkr, "General")
            open_positions.pop(tkr, None)
            if sector_occupied.get(sector) == tkr:
                sector_occupied.pop(sector, None)

        # ── Evaluar nuevas señales ────────────────────────────────────────
        # Candidatos del día: tickers con señal activa HOY ordenados por RS (mejor primero)
        candidatos = []
        for tkr, df_tkr in ticker_data.items():
            if tkr in open_positions:
                continue  # ya en posición
            if date not in df_tkr.index:
                continue
            if not df_tkr.loc[date, 'Entry_Signal']:
                continue
            rs_val = df_tkr.loc[date, 'RS_IBEX'] if 'RS_IBEX' in df_tkr.columns else 1.0
            candidatos.append((tkr, float(rs_val) if not np.isnan(float(rs_val)) else 1.0))

        # Ordenar por RS descendente: primero los más fuertes vs IBEX
        candidatos.sort(key=lambda x: x[1], reverse=True)

        for tkr, rs_val in candidatos:
            sector = SECTOR_MAP.get(tkr, "General")

            # [H] Filtro concentración sectorial: máx 1 por sector
            if sector_limit and sector in sector_occupied:
                continue  # sector ya ocupado, ignorar señal

            df_tkr      = ticker_data[tkr]
            entry_price = df_tkr.loc[date, 'Close']
            atr_entry   = df_tkr.loc[date, 'ATR']
            adx_entry   = df_tkr.loc[date, 'ADX'] if 'ADX' in df_tkr.columns else 25

            initial_risk   = atr_entry * config['atr_mult']
            stop_inicial   = entry_price - initial_risk
            target_partial = entry_price + initial_risk  # 1R

            # [D] Target dinámico según ADX
            rr_efectivo  = 3.0 if (adx_dynamic_rr and adx_entry > 35) else config['rr_ratio']
            target_full  = entry_price + (initial_risk * rr_efectivo)

            open_positions[tkr] = {
                'entry_price':     entry_price,
                'initial_risk':    initial_risk,
                'stop':            stop_inicial,
                'target_partial':  target_partial,
                'target_full':     target_full,
                'breakeven_active': False,
                'entry_date':      date,
                'rr':              rr_efectivo
            }
            sector_occupied[sector] = tkr

    # ── PASO 3: Compilar resultados ────────────────────────────────────────
    if not all_trades_list:
        return pd.DataFrame(), pd.DataFrame()

    all_trades = pd.DataFrame(all_trades_list)

    # Resumen por ticker
    results = []
    for tkr_short in all_trades['Ticker'].unique():
        sub = all_trades[all_trades['Ticker'] == tkr_short]
        wins   = sub[sub['PnL %'] > 0]
        losses = sub[sub['PnL %'] <= 0]
        pf = round(wins['PnL %'].sum() / abs(losses['PnL %'].sum()), 2) \
             if len(losses) > 0 and losses['PnL %'].sum() != 0 else float('inf')
        results.append({
            "Ticker":         tkr_short,
            "Empresa":        sub['Empresa'].iloc[0],
            "Sector":         sub['Sector'].iloc[0],
            "Total Trades":   len(sub),
            "% Acierto":      round(len(wins) / len(sub) * 100, 1),
            "PnL Medio %":    round(sub['PnL %'].mean(), 2),
            "PnL Total %":    round(sub['PnL %'].sum(), 2),
            "Mejor Trade %":  round(sub['PnL %'].max(), 2),
            "Peor Trade %":   round(sub['PnL %'].min(), 2),
            "Duración Media": round(sub['Duración (días)'].mean(), 1),
            "Profit Factor":  pf
        })

    summary_df = pd.DataFrame(results).sort_values("PnL Total %", ascending=False)
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
        vol_threshold = st.slider(
            "Umbral Volumen (x media)", 1.0, 2.5, 1.5, 0.1,
            help="1.5x = volumen al menos 50% mayor que su media. Bajar a 1.2x genera más señales; subir a 2.0x filtra más."
        )
        atr_min_pct = st.slider(
            "ATR Mínimo (% del precio)", 0.5, 2.5, 1.0, 0.1,
            help="Filtra activos inertes que raramente alcanzan el target. 1.0% = el activo debe moverse al menos 1% de su precio al día de media."
        )
        st.caption("Fijos: Cierre > Máx 20d · EMA30 semanal · Precio > SMA200 · RSI 30-70 · OBV alcista")

    with st.expander("🎯 Gestión de Salida", expanded=True):
        partial_exit = st.toggle(
            "Salida Parcial en 1R + Breakeven",
            value=True,
            help="Cierra el 50% en 1R y mueve el stop al precio de entrada. Elimina pérdidas en trades que casi llegaron al target."
        )
        adx_dynamic_rr = st.toggle(
            "Target 3R cuando ADX > 35",
            value=True,
            help="En tendencias muy fuertes amplía el target de 2R a 3R para maximizar las mejores operaciones."
        )
        if partial_exit:
            st.success("✅ Stop Breakeven activo — la 2ª mitad no puede generar pérdida.")
        if adx_dynamic_rr:
            st.info("📈 Target dinámico: 2R normal / 3R si ADX > 35.")

    with st.expander("🏦 Control de Cartera", expanded=True):
        sector_limit = st.toggle(
            "Límite Concentración Sectorial",
            value=True,
            help="Máximo 1 posición abierta por sector simultáneamente. Evita tener BBVA + Santander + CaixaBank a la vez (es la misma apuesta × 3)."
        )
        weekly_confirm = st.toggle(
            "Confirmación Semanal de Rotura",
            value=True,
            help="Exige que el precio también supere el máximo de las últimas 8 semanas en el marco semanal. Elimina roturas falsas dentro de rangos laterales mayores."
        )
        if sector_limit:
            st.success("✅ Concentración controlada — candidatos ordenados por RS vs IBEX.")
        if weekly_confirm:
            st.info("📅 Doble confirmación: rotura diaria + rotura semanal 8 semanas.")

    with st.expander("📧 Alertas Gmail", expanded=False):
        st.markdown("""
        **Configuración** (solo la primera vez):
        1. Ve a [myaccount.google.com](https://myaccount.google.com) → **Seguridad**
        2. Activa **Verificación en 2 pasos** si no la tienes
        3. Ve a **Contraseñas de aplicación** → elige "Correo" + "Otro" → copia la contraseña de 16 caracteres
        4. Usa esa contraseña aquí (NO tu contraseña habitual de Gmail)
        """)
        gmail_user = st.text_input(
            "Tu correo Gmail",
            value=st.session_state.get('gmail_user', ''),
            placeholder="tunombre@gmail.com"
        )
        gmail_app_pw = st.text_input(
            "App Password (16 caracteres)",
            value=st.session_state.get('gmail_app_pw', ''),
            type="password",
            placeholder="xxxx xxxx xxxx xxxx"
        )
        gmail_dest = st.text_input(
            "Email destinatario (puede ser el mismo u otro)",
            value=st.session_state.get('gmail_dest', ''),
            placeholder="destino@gmail.com"
        )
        if st.button("💾 Guardar y probar conexión"):
            if gmail_user and gmail_app_pw and gmail_dest:
                ok = _gmail_send(
                    gmail_user, gmail_app_pw, gmail_dest,
                    "✅ IBEX 35 — Alertas Gmail conectadas",
                    "<h3>✅ Conexión correcta</h3><p>Recibirás alertas cuando haya señales de compra válidas.</p>"
                )
                if ok:
                    st.session_state['gmail_user']   = gmail_user
                    st.session_state['gmail_app_pw'] = gmail_app_pw
                    st.session_state['gmail_dest']   = gmail_dest
                    st.success("✅ Email de prueba enviado correctamente.")
                else:
                    st.error("❌ Fallo al enviar. Verifica el email y la App Password.")
            else:
                st.warning("Rellena los tres campos.")
        if st.session_state.get('gmail_user'):
            st.success(f"🟢 Gmail configurado · {st.session_state.get('gmail_user','')} → {st.session_state.get('gmail_dest','')}")

    config = {
        'rr_ratio':      rr_ratio,
        'vol_threshold': vol_threshold,
        'atr_min_pct':   atr_min_pct,
        'atr_mult':      atr_mult,
        'partial_exit':  partial_exit,
        'adx_dynamic_rr': adx_dynamic_rr,
        'sector_limit':  sector_limit,
        'weekly_confirm': weekly_confirm
    }

# =============================================================================
# 6. TABS
# =============================================================================

tab_scanner, tab_compresion, tab_backtest, tab_manual = st.tabs([
    "🚀 Scanner de Oportunidades",
    "🔭 Radar de Compresión",
    "📊 Backtesting",
    "📘 Manual y Metodología"
])

# ---------------------------------------------------------------------------
# TAB 2 — RADAR DE COMPRESIÓN
# ---------------------------------------------------------------------------
with tab_compresion:
    st.markdown("### 🔭 Radar de Compresión — Anticipar Roturas 1-5 Días")
    st.caption("""
    Detecta activos que están **cargando energía** antes de la rotura.
    Un score alto aquí significa que en los próximos 1-5 días podría generarse
    la señal en el Scanner. Te da tiempo para preparar la orden límite.
    """)

    if 'scan_data_raw' not in st.session_state:
        st.info("👆 Ejecuta primero **Escanear Mercado** en la pestaña Scanner para cargar datos.")
    else:
        data_raw_comp = st.session_state['scan_data_raw']
        ibex_df_comp  = st.session_state.get('ibex_df')

        with st.spinner("Analizando patrones de compresión..."):
            resultados_comp = escanear_compresion(data_raw_comp, ibex_df_comp)

        if not resultados_comp:
            st.info("No se detectan patrones de compresión activos en este momento.")
        else:
            # ── Tabla principal ──────────────────────────────────────────────
            st.subheader(f"📋 {len(resultados_comp)} activos en compresión")

            cols_comp = ['Ticker', 'Empresa', 'Score Compres.', 'Patrones',
                         'Precio', 'Pivot (€)', 'Dist. Pivot %', 'RS vs IBEX', 'SMA50 ok']
            df_comp = pd.DataFrame([
                {k: v for k, v in r.items() if not k.startswith('_')}
                for r in resultados_comp
            ])

            # Formateo
            if 'Precio'        in df_comp.columns: df_comp['Precio']        = df_comp['Precio'].apply(lambda x: f"{x:.2f}€")
            if 'Pivot (€)'     in df_comp.columns: df_comp['Pivot (€)']     = df_comp['Pivot (€)'].apply(lambda x: f"{x:.2f}€")
            if 'Dist. Pivot %' in df_comp.columns: df_comp['Dist. Pivot %'] = df_comp['Dist. Pivot %'].apply(lambda x: f"+{x:.1f}%")
            if 'RS vs IBEX'    in df_comp.columns: df_comp['RS vs IBEX']    = df_comp['RS vs IBEX'].apply(lambda x: f"{x:.3f}" if x else "—")

            styled_comp = df_comp[cols_comp].style \
                .background_gradient(subset=['Score Compres.'], cmap='YlOrRd', vmin=0, vmax=100) \
                .map(lambda x: 'color:#00ff88;font-weight:bold' if '✅' in str(x) else
                                    'color:#ff4444' if '❌' in str(x) else '',
                          subset=['SMA50 ok'])

            st.dataframe(styled_comp, width='stretch', hide_index=True)

            # Leyenda de patrones
            st.markdown("""
            <small>
            <b>VCP</b>: Correcciones decrecientes (Minervini) — el patrón más potente &nbsp;|&nbsp;
            <b>Squeeze</b>: Bollinger dentro de Keltner — volatilidad en mínimos &nbsp;|&nbsp;
            <b>NR7/NR4</b>: Día de mínimo rango — máxima compresión intradiaria
            </small>
            """, unsafe_allow_html=True)

            st.divider()

            # ── Detalle del activo seleccionado ──────────────────────────────
            st.subheader("🔎 Detalle del patrón")
            ticker_sel_comp = st.selectbox(
                "Seleccionar activo",
                [r['Ticker'] for r in resultados_comp],
                key="sel_comp"
            )
            res_sel = next(r for r in resultados_comp if r['Ticker'] == ticker_sel_comp)
            vcp = res_sel['_vcp']
            sqz = res_sel['_sqz']
            nr  = res_sel['_nr']

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                estado_vcp = "🟢 Detectado" if vcp['detectado'] else "⚪ No activo"
                st.metric("VCP — Volatility Contraction", estado_vcp)
                if vcp.get('correcciones'):
                    for i, c in enumerate(vcp['correcciones'], 1):
                        st.write(f"  Corrección {i}: **{c:.1f}%**")
                st.write(f"Ratio compresión: **{vcp.get('ratio', '—')}**")
                st.write(f"Días en compresión: **{vcp.get('dias_compresion', '—')}**")
                st.write(f"Pivot esperado: **{vcp.get('pivot', '—')}€**")
                st.progress(vcp['score'] / 100, text=f"Score VCP: {vcp['score']}/100")

            with col_b:
                estado_sqz = "🟢 Activo" if sqz['squeeze_activo'] else "⚪ No activo"
                st.metric("Squeeze BB + Keltner", estado_sqz)
                st.write(f"Días en squeeze: **{sqz['dias_squeeze']}**")
                st.write(f"Momentum: **{sqz['histograma']:.4f}**")
                dir_color = "🟢" if sqz['direccion'] == 'alcista' else "🔴" if sqz['direccion'] == 'bajista' else "🟡"
                st.write(f"Dirección: {dir_color} **{sqz['direccion']}**")
                st.progress(sqz['score'] / 100, text=f"Score Squeeze: {sqz['score']}/100")

            with col_c:
                estado_nr = ("🟢 NR7" if nr['nr7'] else "🟡 NR4" if nr['nr4'] else "⚪ No activo")
                st.metric("NR7 / NR4 — Narrow Range", estado_nr)
                st.write(f"Rango hoy: **{nr['rango_hoy']:.2f}%**")
                st.write(f"Rango medio 20d: **{nr['rango_medio']:.2f}%**")
                st.write(f"Compresión vs media: **{nr['compresion_pct']:.1f}%**")
                st.progress(nr['score'] / 100, text=f"Score NR: {nr['score']}/100")

            # ── Gráfico del activo ────────────────────────────────────────────
            st.divider()
            ticker_full_comp = ticker_sel_comp + ".MC"
            try:
                df_plot = data_raw_comp[ticker_full_comp].copy().dropna() \
                          if isinstance(data_raw_comp.columns, pd.MultiIndex) \
                          else data_raw_comp.copy().dropna()
                df_plot = df_plot.iloc[-90:]   # Últimos 90 días

                # Bollinger Bands para el gráfico
                bb_mid   = df_plot['Close'].rolling(20).mean()
                bb_std_p = df_plot['Close'].rolling(20).std()
                bb_up    = bb_mid + 2 * bb_std_p
                bb_lo    = bb_mid - 2 * bb_std_p

                fig_comp = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75, 0.25], vertical_spacing=0.03,
                    subplot_titles=[f"{res_sel['Empresa']} — Patrón de Compresión (90d)", "Volumen"]
                )

                # Velas
                fig_comp.add_trace(go.Candlestick(
                    x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                    low=df_plot['Low'], close=df_plot['Close'],
                    name="Precio",
                    increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
                ), row=1, col=1)

                # Bollinger Bands
                fig_comp.add_trace(go.Scatter(
                    x=df_plot.index, y=bb_up, line=dict(color='rgba(100,180,255,0.5)', width=1),
                    name="BB Superior", showlegend=False
                ), row=1, col=1)
                fig_comp.add_trace(go.Scatter(
                    x=df_plot.index, y=bb_lo, line=dict(color='rgba(100,180,255,0.5)', width=1),
                    fill='tonexty', fillcolor='rgba(100,180,255,0.05)',
                    name="BB Inferior", showlegend=False
                ), row=1, col=1)
                fig_comp.add_trace(go.Scatter(
                    x=df_plot.index, y=bb_mid, line=dict(color='rgba(100,180,255,0.4)', width=1, dash='dot'),
                    name="BB Media", showlegend=False
                ), row=1, col=1)

                # Línea de pivot
                pivot_val = res_sel['Pivot (€)'].replace('€', '') if isinstance(res_sel['Pivot (€)'], str) else res_sel.get('_vcp', {}).get('pivot')
                if pivot_val:
                    fig_comp.add_hline(
                        y=float(str(pivot_val).replace('€','')),
                        line=dict(color='#ffaa00', dash='dash', width=2),
                        annotation_text=f"Pivot {pivot_val}€",
                        row=1, col=1
                    )

                # Volumen coloreado por squeeze
                vol_colors = ['rgba(255,100,100,0.5)' if float(v) < float(df_plot['Volume'].rolling(20).mean().iloc[i])
                              else 'rgba(100,255,150,0.5)'
                              for i, v in enumerate(df_plot['Volume'])]
                fig_comp.add_trace(go.Bar(
                    x=df_plot.index, y=df_plot['Volume'],
                    marker_color=vol_colors, name="Volumen"
                ), row=2, col=1)

                fig_comp.update_layout(
                    template="plotly_dark", height=500,
                    xaxis_rangeslider_visible=False,
                    margin=dict(t=40, b=20),
                    legend=dict(orientation='h', y=1.05)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar el gráfico: {e}")

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

    # ── PANEL DE ALERTAS INTRADIARIAS ─────────────────────────────────────────
    # Resuelve el problema de "entrar tarde": muestra en tiempo casi-real
    # qué activos están rompiendo ahora mismo y cuánto llevan subido desde el trigger.
    with st.expander("⚡ Alertas Intradiarias — Roturas en Formación (actualiza cada 5 min)", expanded=True):
        st.caption("""
        Datos de 15 minutos, refresco cada 5 min. **Úsalo durante la sesión** (9:00-17:35) para ver
        qué activos están cruzando su máximo de 20 días AHORA, antes de que la vela diaria cierre.
        Así evitas entrar cuando el movimiento ya está hecho.
        """)

        col_al1, col_al2 = st.columns([3, 1])
        with col_al2:
            refrescar = st.button("🔄 Refrescar alertas")

        if refrescar:
            st.cache_data.clear()  # Fuerza re-descarga de datos intradiarios

        with st.spinner("Cargando alertas intradiarias..."):
            alertas = escanear_alertas_intradiarias(IBEX35_TICKERS, vol_threshold=config.get('vol_threshold', 1.5))

        if alertas:
            df_alertas = pd.DataFrame([{k: v for k, v in a.items() if k != '_urgencia'} for a in alertas])

            def color_estado(val):
                if '🚨' in str(val): return 'color: #ff9900; font-weight: bold'
                if '⚠️' in str(val): return 'color: #ffcc00'
                if '🔔' in str(val): return 'color: #88aaff'
                if '🚫' in str(val): return 'color: #ff4444'
                return ''

            def color_dist(val):
                try:
                    v = float(val)
                    if v <= 1.5:   return 'color: #00ff88; font-weight: bold'
                    elif v <= 3.0: return 'color: #ffcc00'
                    elif v < 0:    return 'color: #88aaff'
                    else:          return 'color: #ff4444'
                except: return ''

            st.dataframe(
                df_alertas.style
                .map(color_estado, subset=['Estado'])
                .map(color_dist, subset=['Dist. %']),
                width='stretch',
                hide_index=True
            )

            # Leyenda
            st.markdown("""
            <small>
            🚨 <b>Rotura Fresca</b> (0-1.5%): entrada con R:R intacto — actúa ahora &nbsp;|&nbsp;
            ⚠️ <b>Entrada Tardía</b> (1.5-3%): R:R deteriorado, espera pullback &nbsp;|&nbsp;
            🚫 <b>Muy Tarde</b> (+3%): descarta o espera retroceso al trigger &nbsp;|&nbsp;
            🔔 <b>Acercándose</b> (&lt;0%): prepárate, puede romper en breve
            </small>
            """, unsafe_allow_html=True)
        else:
            st.info("No hay activos en zona de rotura ahora mismo.")

    st.divider()

    if st.button("🚀 Escanear Mercado", type="primary"):
        with st.spinner("Descargando datos y analizando IBEX 35..."):
            progreso = st.progress(0)
            resultados = []

            # ── DIAGNÓSTICO: mostrar estructura real del DataFrame ─────────────
            data_raw = yf.download(IBEX35_TICKERS, period="2y", group_by='ticker', progress=False)

            with st.expander("🔬 Diagnóstico de datos (expande si no salen resultados)", expanded=False):
                st.write(f"**Shape:** {data_raw.shape}")
                st.write(f"**Tipo columnas:** {type(data_raw.columns).__name__}")
                if isinstance(data_raw.columns, pd.MultiIndex):
                    st.write(f"**Niveles:** {data_raw.columns.nlevels}")
                    st.write(f"**Nivel 0 (muestra):** {data_raw.columns.get_level_values(0).unique().tolist()[:8]}")
                    st.write(f"**Nivel 1 (muestra):** {data_raw.columns.get_level_values(1).unique().tolist()[:8]}")
                else:
                    st.write(f"**Columnas:** {data_raw.columns.tolist()}")
                # Probar extracción del primer ticker
                t0 = IBEX35_TICKERS[0]
                try:
                    df_test = _extraer_ticker_df(data_raw, t0)
                    st.success(f"✅ Extracción de {t0}: shape={df_test.shape}, cols={df_test.columns.tolist()}")
                    # Probar analizar_ticker directamente y mostrar el error real
                    if hasattr(df_test.index, 'tz') and df_test.index.tz is not None:
                        df_test.index = df_test.index.tz_localize(None)
                    try:
                        res_test = analizar_ticker(t0, df_test, config, market_bullish, df_ibex=ibex_df)
                        if res_test:
                            st.success(f"✅ analizar_ticker({t0}) devolvió resultado con score={res_test.get('Score')}")
                        else:
                            st.warning(f"⚠️ analizar_ticker({t0}) devolvió None — algún filtro rechaza este ticker")
                            # Mostrar valores clave para saber cuál filtro falla
                            import numpy as np
                            close = float(df_test['Close'].iloc[-1])
                            sma200 = float(df_test['Close'].rolling(200).mean().iloc[-1])
                            prev_high = float(df_test['High'].shift(1).rolling(20).max().iloc[-1])
                            vol_ratio = float(df_test['Volume'].iloc[-1] / df_test['Volume'].rolling(20).mean().iloc[-1])
                            st.code(f"close={close:.2f}  sma200={sma200:.2f}  sobre_sma200={close>sma200}\nprev_high_20d={prev_high:.2f}  rotura={close>prev_high}\nvol_ratio={vol_ratio:.2f}  (umbral={config.get('vol_threshold',1.5)})\nfilas={len(df_test)}  market_bullish={market_bullish}")
                    except Exception as e2:
                        st.error(f"❌ analizar_ticker lanzó excepción: {type(e2).__name__}: {e2}")
                        import traceback
                        st.code(traceback.format_exc())
                except Exception as e:
                    st.error(f"❌ Extracción de {t0} falló: {e}")

            st.session_state['scan_data_raw'] = data_raw
            st.session_state['ibex_df'] = ibex_df

            errores_detalle = []
            for i, ticker in enumerate(IBEX35_TICKERS):
                try:
                    df = _extraer_ticker_df(data_raw, ticker)
                    if df.empty or len(df) < 50:
                        errores_detalle.append(f"{ticker}: df vacío o muy corto ({len(df)} filas)")
                        progreso.progress((i + 1) / len(IBEX35_TICKERS))
                        continue
                    # Strip timezone — yfinance ≥0.2.38 devuelve índice UTC
                    if hasattr(df.index, 'tz') and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    res = analizar_ticker(ticker, df, config, market_bullish, df_ibex=ibex_df)
                    if res:
                        resultados.append(res)
                except Exception as e:
                    errores_detalle.append(f"{ticker}: {type(e).__name__}: {e}")
                progreso.progress((i + 1) / len(IBEX35_TICKERS))

            progreso.empty()

            if resultados:
                df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ('Score_Log', '_df')} for r in resultados])
                df_res = df_res.sort_values("Score", ascending=False)
                st.session_state['scan_results'] = df_res
                st.session_state['scan_results_full'] = resultados
                st.session_state['scan_errors'] = errores_detalle
            else:
                st.session_state['scan_errors'] = errores_detalle
                if errores_detalle:
                    with st.expander("❌ Errores del scanner — expande para ver el diagnóstico", expanded=True):
                        for e in errores_detalle[:10]:
                            st.code(e)
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

        # Columnas deseadas — incluye nuevas columnas de calidad de entrada
        cols_ideales = ['Ticker', 'Empresa', 'Score', 'Rotura', 'Entrada',
                        'Dist. Trigger %', 'Trigger (€)', 'Tendencia Semanal',
                        'Precio', 'RSI Actual', 'RSI Pre-Rotura', 'RS vs IBEX',
                        'ADX', 'Vol Ratio', 'SMA50', 'SMA200']
        cols_display = [c for c in cols_ideales if c in df_filtered.columns]

        df_disp = df_filtered[cols_display].copy()

        # Formateo defensivo
        fmt = {
            'Precio':          lambda x: f"{x:.2f}€" if pd.notna(x) else "—",
            'Trigger (€)':     lambda x: f"{x:.2f}€" if pd.notna(x) else "—",
            'Dist. Trigger %': lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
            'RSI Actual':      lambda x: f"{x:.1f}" if pd.notna(x) and x else "—",
            'RSI Pre-Rotura':  lambda x: f"{x:.1f}" if pd.notna(x) and x else "—",
            'RS vs IBEX':      lambda x: f"{x:.3f}" if pd.notna(x) and x else "—",
            'ADX':             lambda x: f"{x:.1f}" if pd.notna(x) else "—",
            'Vol Ratio':       lambda x: f"{x:.2f}x" if pd.notna(x) else "—",
            'SMA50':           lambda x: f"{x:.2f}€" if pd.notna(x) else "—",
            'SMA200':          lambda x: f"{x:.2f}€" if pd.notna(x) else "—",
        }
        for col, fn in fmt.items():
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].apply(fn)

        # Estilos
        styled = df_disp.style
        if 'Tendencia Semanal' in df_disp.columns:
            styled = styled.map(
                lambda x: 'color: #00ff88' if '🟢' in str(x) else ('color: #ff0055' if '🔴' in str(x) else ''),
                subset=['Tendencia Semanal']
            )
        if 'Rotura' in df_disp.columns:
            styled = styled.map(
                lambda x: 'color: #ff9900; font-weight: bold' if '🚨' in str(x) else '',
                subset=['Rotura']
            )
        if 'Entrada' in df_disp.columns:
            styled = styled.map(
                lambda x: ('color: #00ff88' if '✅' in str(x)
                           else 'color: #ffcc00' if '⚠️' in str(x)
                           else 'color: #ff4444' if '🚫' in str(x)
                           else ''),
                subset=['Entrada']
            )
        if 'Score' in df_disp.columns:
            styled = styled.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-30, vmax=110)

        st.dataframe(styled, width='stretch')

        # ── Botón de alertas Gmail ─────────────────────────────────────────────
        gmail_user   = st.session_state.get('gmail_user', '')
        gmail_app_pw = st.session_state.get('gmail_app_pw', '')
        gmail_dest   = st.session_state.get('gmail_dest', '')

        if gmail_user and gmail_app_pw and gmail_dest:
            col_gm1, col_gm2 = st.columns([2, 1])
            with col_gm1:
                score_minimo_gm = st.slider(
                    "Score mínimo para alertar por email", 50, 100, 65, 5,
                    key="gm_score_min"
                )
            with col_gm2:
                st.write("")
                st.write("")
                enviar_gm = st.button("📧 Enviar señales por Gmail", type="secondary")

            if enviar_gm:
                # Filtrar señales: score mínimo + rotura activa + entrada no inválida
                senales_gm = df_res[
                    (df_res['Score'] >= score_minimo_gm) &
                    (df_res.get('Rotura', pd.Series()) == "🚨 SÍ") &
                    (~df_res.get('Entrada', pd.Series('✅')).str.contains('🚫', na=False))
                ] if 'Rotura' in df_res.columns else df_res[df_res['Score'] >= score_minimo_gm]

                if senales_gm.empty:
                    st.info("No hay señales con rotura activa y entrada válida para alertar.")
                    _gmail_send(
                        gmail_user, gmail_app_pw, gmail_dest,
                        f"IBEX 35 — Sin señales válidas {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                        f"<p>Sin señales con score ≥ {score_minimo_gm} y rotura activa en este escaneo.</p>"
                    )
                else:
                    enviadas = 0
                    for _, row_gm in senales_gm.iterrows():
                        res_full = next(
                            (r for r in resultados_full if r['Ticker'] == row_gm['Ticker']),
                            None
                        )
                        if not res_full:
                            continue

                        asunto, cuerpo = _gmail_formatear_senal(
                            empresa    = res_full['Empresa'],
                            ticker     = res_full['Ticker'],
                            precio     = float(res_full['Precio']),
                            trigger    = float(res_full.get('Trigger (€)', res_full['Precio'])),
                            dist_pct   = float(res_full.get('Dist. Trigger %', 0)),
                            stop       = float(res_full['Stop Loss']),
                            target     = float(res_full['Target']),
                            riesgo_pct = float(res_full['Riesgo %']),
                            vol_ratio  = float(res_full['Vol Ratio']),
                            rsi_pre    = float(res_full.get('RSI Pre-Rotura') or res_full.get('RSI Actual', 50)),
                            rs_val     = res_full.get('RS vs IBEX'),
                            adx        = float(res_full['ADX']),
                            rr         = config['rr_ratio'],
                            score      = int(res_full['Score'])
                        )
                        if _gmail_send(gmail_user, gmail_app_pw, gmail_dest, asunto, cuerpo):
                            enviadas += 1

                    if enviadas > 0:
                        st.success(f"✅ {enviadas} alerta(s) enviada(s) a {gmail_dest}.")
                    else:
                        st.error("No se pudieron enviar los emails. Verifica las credenciales en el sidebar.")
        else:
            st.caption("💡 Configura Gmail en el sidebar para recibir alertas por email.")

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
    st.markdown("### 📊 Motor de Backtesting Vectorizado v3")
    st.markdown("""
    Simula la estrategia completa sobre **datos históricos reales** de cada componente del IBEX 35.
    Usa las mismas condiciones que el Scanner: rotura de máximos con volumen, EMA30 semanal, SMA200 y filtro de régimen.
    """)

    # Resumen de optimizaciones activas
    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
    col_opt1.metric("Salida Parcial 1R",     "✅" if config.get('partial_exit') else "❌")
    col_opt2.metric("Target ADX Dinámico",   "✅" if config.get('adx_dynamic_rr') else "❌")
    col_opt3.metric("Límite Sectorial",      "✅" if config.get('sector_limit') else "❌")
    col_opt4.metric("Conf. Semanal",         "✅" if config.get('weekly_confirm') else "❌")
    col_opt5, col_opt6, col_opt7, col_opt8 = st.columns(4)
    col_opt5.metric("RSI Pre-Rotura",        "✅ Activo")
    col_opt6.metric("RS vs IBEX",            "✅ Activo")
    col_opt7.metric("Umbral Volumen",         f"{config.get('vol_threshold',1.5)}x")
    col_opt8.metric("ATR Mín.",              f"{config.get('atr_min_pct',1.0)}%")
    st.caption("Stop Mínimo Garantizado (0.5R) siempre activo — evita trailing agresivo en retrocesos normales.")

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
                .map(color_pnl, subset=['PnL Total %', 'PnL Medio %', 'Mejor Trade %', 'Peor Trade %'])
                .background_gradient(subset=['% Acierto'], cmap='RdYlGn', vmin=30, vmax=70)
                .background_gradient(subset=['Profit Factor'], cmap='RdYlGn', vmin=0.5, vmax=3.0),
                width='stretch'
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

            # Desglose por tipo de salida si hay salida parcial activa
            if config.get('partial_exit') and 'Tipo' in all_trades.columns:
                col_leg1, col_leg2, col_leg3 = st.columns(3)
                parciales = all_trades[all_trades['Resultado'].str.contains('Parcial', na=False)]
                targets   = all_trades[all_trades['Resultado'].str.contains('Target', na=False)]
                stops     = all_trades[all_trades['Resultado'].str.contains('Stop', na=False)]
                col_leg1.metric("🎯 Salidas en 1R (1ª mitad)", len(parciales),
                                delta=f"Media: {parciales['PnL %'].mean():.2f}%" if len(parciales) > 0 else None)
                col_leg2.metric("✅ Targets alcanzados (2ª mitad)", len(targets),
                                delta=f"Media: {targets['PnL %'].mean():.2f}%" if len(targets) > 0 else None)
                col_leg3.metric("❌ Stops / Breakevens", len(stops),
                                delta=f"Media: {stops['PnL %'].mean():.2f}%" if len(stops) > 0 else None)
                st.caption("Cada operación con salida parcial genera 2 filas: 1ª mitad (cerrada en 1R) y 2ª mitad (breakeven o target).")

            cols_log = ['Ticker', 'Empresa', 'Sector', 'Entrada', 'Salida',
                        'Precio Entrada', 'Precio Salida', 'Resultado', 'PnL %',
                        'Duración (días)', 'Tipo']
            cols_log = [c for c in cols_log if c in all_trades.columns]

            def color_resultado(val):
                if 'Parcial' in str(val): return 'color: #88ccff'
                if '✅' in str(val): return 'color: #00ff88'
                if '❌' in str(val): return 'color: #ff0055'
                if '⚪' in str(val): return 'color: #aaaaaa'
                return ''

            st.dataframe(
                all_trades[cols_log].sort_values('Salida', ascending=False).style
                .map(color_pnl, subset=['PnL %'])
                .map(color_resultado, subset=['Resultado']),
                width='stretch',
                height=400
            )

# ---------------------------------------------------------------------------
# TAB 3 — MANUAL
# ---------------------------------------------------------------------------
with tab_manual:
    st.markdown("""
    # 📘 Manual del Operador Swing — v3.0

    Esta aplicación implementa una estrategia **Trend Following con Disparador de Entrada** diseñada para capturar movimientos de varias semanas dentro de tendencias establecidas.

    ---

    ## 🆕 Cambios principales en v3.0

    ### El problema que se resolvió: contradicción RSI vs Rotura

    En v2.0 existía una contradicción lógica: el disparador de entrada exigía que el precio superara el máximo de 20 días (lo que implica que el precio ha subido con fuerza), pero el filtro RSI penalizaba cuando el RSI superaba 70 (lo que ocurre inevitablemente cuando el precio sube con fuerza). El sistema se bloqueaba a sí mismo en los mejores momentos de entrada.

    **Solución v3.0:** Se separa el RSI en dos usos distintos con lógicas distintas.

    ---

    ## 🌐 Filtro Maestro: Régimen de Mercado

    - **Condición**: El IBEX 35 debe cotizar por encima de su SMA200.
    - **Efecto**: Si el régimen es bajista, penalización de -30 puntos y advertencia en pantalla.

    ---

    ## 1. El Algoritmo de Scoring v3

    | Regla | Condición | Puntos |
    |-------|-----------|--------|
    | Régimen Mercado | IBEX < SMA200 | -30 |
    | Tendencia Semanal | Precio > EMA30 semanal | +25 / -20 |
    | Soporte Mayor | Precio > SMA200 diaria | +20 / -15 |
    | Tendencia Media | Precio > SMA50 diaria | +10 |
    | ADX + DI | ADX>25 y DI+>DI- | +15 / -10 |
    | **RSI Pre-Rotura** | **RSI medio 5d previos en 45-68** | **+15 / -10** |
    | OBV | OBV > SMA20(OBV) | +10 / -5 |
    | **Fortaleza Relativa vs IBEX** | **RS 20d > 1.05** | **+15 / -10** |
    | Zona de Valor (pullback SMA50) | Precio < 5% de SMA50 | +5 |
    | Disparador de Entrada | Rotura máx 20d + volumen expandido | +15 |

    ---

    ## 2. RSI Pre-Rotura vs RSI Actual — La corrección clave

    **¿Por qué el RSI actual no sirve como filtro de entrada en roturas?**

    Imagina que una acción lleva 3 semanas subiendo poco a poco y hoy finalmente supera el máximo de los últimos 20 días con volumen alto. Eso es exactamente la señal que buscamos. Pero en ese momento el RSI inevitablemente estará entre 65 y 80, porque el precio ha subido mucho en 14 días. Usar RSI < 70 como condición eliminaría esta entrada perfecta.

    **Lo que realmente queremos saber** es si los días anteriores a la rotura el activo estaba en zona de consolidación sana, no agotado ni en caída libre. Para eso usamos el **RSI Pre-Rotura**: la media del RSI de los 5 días anteriores al día actual.

    | Situación | RSI Pre-Rotura | RSI día de rotura | Interpretación |
    |-----------|---------------|-------------------|----------------|
    | ✅ Ideal | 50-65 | 68-78 | Consolidó bien, ahora rompe con fuerza |
    | ✅ Aceptable | 45-68 | Cualquiera | Zona sana previa |
    | ❌ Agotado | >78 | >80 | Llegó a la rotura ya sobrecomprado |
    | ❌ Débil | <40 | <50 | Sin momentum real antes del impulso |

    El RSI actual sigue mostrándose en la tabla pero como información, no como filtro.

    ---

    ## 3. Fortaleza Relativa vs IBEX (RS) — El indicador más importante añadido

    **¿Qué mide?**

    Compara el rendimiento del activo con el del IBEX 35 en los últimos 20 días:

    `RS = (1 + rendimiento_activo_20d) / (1 + rendimiento_ibex_20d)`

    - **RS > 1.05**: El activo sube un 5% más que el IBEX → Hay dinero institucional entrando específicamente en este valor (+15 puntos)
    - **RS > 1.0**: Supera ligeramente al índice (+8 puntos)
    - **RS ~ 1.0**: Se mueve en línea con el mercado (0 puntos)
    - **RS < 0.95**: Va peor que el mercado → Evitar, hay distribución (-10 puntos)

    **¿Por qué es tan importante?**

    Un activo puede romper máximos simplemente porque todo el mercado sube. Eso no es una señal de calidad — es ruido correlacionado. Un activo que rompe máximos Y además supera al IBEX indica que hay dinero entrando específicamente en él, no solo por inercia del mercado. Es la diferencia entre surfear una ola grande y surfear una ola grande siendo el mejor surfista de la playa.

    Este concepto es el núcleo del sistema SEPA de Mark Minervini y del Relative Strength Rating de IBD/Investor's Business Daily.

    ---

    ## 4. El Disparador de Entrada: sin contradicción

    **Condiciones de entrada v3 (en el backtesting y el scanner):**

    1. Cierre > Máximo de 20 días (rotura de resistencia)
    2. Volumen ≥ umbral configurable × su media (confirmación institucional)
    3. EMA30 semanal alcista (tendencia de fondo)
    4. Precio > SMA200 (sobre soporte mayor)
    5. RSI Pre-Rotura entre 40-75 (zona sana los días previos)
    6. RS vs IBEX ≥ 1.0 (el activo supera al índice)
    7. OBV > SMA20 OBV (presión compradora acumulada)
    8. IBEX en régimen alcista (SMA200)
    9. ATR ≥ 1% del precio (volatilidad mínima para alcanzar target)

    Ninguna de estas condiciones se contradice con las demás.

    ---

    ## 5. Gestión de Salida (v3)

    **Stop Trailing + Salida Parcial en 1R:**

    - El stop sube con el precio (trailing basado en ATR × multiplicador)
    - Al alcanzar 1R de beneficio: se cierra el 50% y el stop se mueve a breakeven
    - La segunda mitad busca el target completo (2R normal, 3R si ADX > 35 en entrada)
    - Resultado: los trades que "casi llegaron" ya no son pérdidas completas

    **¿Qué es 1R y 2R?**

    R = distancia en euros entre precio de entrada y stop loss. Es el riesgo concreto de esa operación.

    - **1R** = has ganado lo mismo que arriesgaste (precio subió = stop loss desde entrada)
    - **2R** = has ganado el doble de lo que arriesgaste (target estándar)
    - **3R** = ganancia triple del riesgo (solo cuando ADX > 35 al entrar)
    """)
    st.info("⚡ Esta aplicación es una herramienta de análisis técnico educativa. No constituye asesoramiento financiero. Opera siempre con gestión de riesgo estricta.")
