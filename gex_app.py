"""
GEX Dashboard - Streamlit App
Instalar: pip install streamlit yfinance plotly pandas scipy
Ejecutar: streamlit run gex_app.py
Desplegar gratis: https://streamlit.io/cloud
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.stats import norm

# ── CONFIGURACION DE PAGINA ───────────────────────────────────────────────────
st.set_page_config(
    page_title="GEX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ESTILOS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    .block-container {
        padding-top: 0.3rem;
        padding-bottom: 1rem;
    }
    /* Ocultar header y footer de Streamlit */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer {
        display: none !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* Metricas */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.75rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.1rem !important;
    }
    /* Cards */
    .gex-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .gex-card h4 {
        color: #e3b341;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin: 0 0 10px 0;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    /* Modelos de estimacion */
    .model-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #21262d;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .model-row:last-child { border-bottom: none; }
    .model-name { color: #8b949e; }
    .model-val-up   { color: #2ea043; font-weight: 700; }
    .model-val-down { color: #da3633; font-weight: 700; }
    .model-val-neu  { color: #58a6ff; font-weight: 700; }
    /* Niveles */
    .level-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #21262d;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }
    .level-row:last-child { border-bottom: none; }
    .level-label { color: #8b949e; }
    /* Análisis */
    .analysis-item {
        padding: 6px 0;
        color: #f0f6fc;
        font-size: 0.84rem;
        border-bottom: 1px solid #21262d;
        line-height: 1.4;
    }
    .analysis-item:last-child { border-bottom: none; }
    /* Header */
    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 0.1em;
    }
    /* Consenso badge */
    .consenso-up   { color: #2ea043; font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .consenso-down { color: #da3633; font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .confidence-alta  { color: #2ea043; font-weight: 600; }
    .confidence-media { color: #e3b341; font-weight: 600; }
    .confidence-baja  { color: #da3633; font-weight: 600; }
    /* Advertencia */
    .warning-box {
        background: #1c2128;
        border: 1px solid #e3b341;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 8px;
    }
    div[data-testid="stButton"] button {
        background: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        width: 100%;
        padding: 10px;
    }
    div[data-testid="stButton"] button:hover {
        background: #2ea043;
    }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ── FUNCIONES DE CALCULO ──────────────────────────────────────────────────────
def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    except Exception:
        return 0.0


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data(symbol, num_exp, risk_free):
    """Descarga GEX, niveles y estimaciones. Cache de 5 minutos."""
    ticker = yf.Ticker(symbol)

    # Spot
    try:
        spot = float(ticker.fast_info["last_price"])
    except Exception:
        hist = ticker.history(period="2d")
        spot = float(hist["Close"].iloc[-1])

    # Ratio SPY->SPX
    try:
        spx_price = float(yf.Ticker("^GSPC").fast_info["last_price"])
        ratio = spx_price / spot
    except Exception:
        ratio = 10.0

    all_exps = ticker.options
    exps = all_exps[:num_exp]
    today = datetime.today()
    K_LO, K_HI = spot * 0.85, spot * 1.15
    rows = []

    for exp in exps:
        try:
            chain = ticker.option_chain(exp)
        except Exception:
            continue
        days = (datetime.strptime(exp, "%Y-%m-%d") - today).days
        T = max(days / 365.0, 1.0 / 365.0)
        w = max(0.40, 1.0 - days * 0.01)

        for _, row in chain.calls.iterrows():
            K = float(row.get("strike", 0))
            if not (K_LO <= K <= K_HI): continue
            oi_raw = row.get("openInterest")
            OI = 0 if (oi_raw is None or str(oi_raw) == "nan") else int(oi_raw)
            if OI == 0: continue
            iv = float(row.get("impliedVolatility") or 0)
            g  = row.get("gamma")
            g  = bs_gamma(spot, K, T, risk_free, iv) if (g is None or (isinstance(g, float) and np.isnan(g))) else float(g)
            rows.append({"strike": K, "tipo": "call", "gex": g * OI * 100 * spot * spot / 1e9 * w})

        for _, row in chain.puts.iterrows():
            K = float(row.get("strike", 0))
            if not (K_LO <= K <= K_HI): continue
            oi_raw = row.get("openInterest")
            OI = 0 if (oi_raw is None or str(oi_raw) == "nan") else int(oi_raw)
            if OI == 0: continue
            iv = float(row.get("impliedVolatility") or 0)
            g  = row.get("gamma")
            g  = bs_gamma(spot, K, T, risk_free, iv) if (g is None or (isinstance(g, float) and np.isnan(g))) else float(g)
            rows.append({"strike": K, "tipo": "put", "gex": -g * OI * 100 * spot * spot / 1e9 * w})

    raw  = pd.DataFrame(rows)
    pivot = raw.pivot_table(index="strike", columns="tipo", values="gex", aggfunc="sum").fillna(0)
    pivot.columns.name = None
    if "call" not in pivot.columns: pivot["call"] = 0.0
    if "put"  not in pivot.columns: pivot["put"]  = 0.0
    pivot["net"] = pivot["call"] + pivot["put"]
    pivot = pivot.reset_index()

    # Niveles
    dff = pivot.copy().sort_values("strike")
    lo_lvl, hi_lvl = spot * 0.90, spot * 1.10
    dff = dff[(dff["strike"] >= lo_lvl) & (dff["strike"] <= hi_lvl)]
    if len(dff) < 5: dff = pivot.copy().sort_values("strike")
    dff["cumnet"] = dff["net"].cumsum()
    signs = (dff["cumnet"].shift(1) * dff["cumnet"]) < 0
    if signs.any():
        flips = dff.loc[signs, "strike"]
        gamma_flip = float(flips.iloc[(flips - spot).abs().argsort().iloc[0]])
    else:
        gamma_flip = None
    call_wall = float(dff.loc[dff["call"].idxmax(), "strike"])
    put_wall  = float(dff.loc[dff["put"].idxmin(), "strike"])
    dff["abs_net"] = dff["net"].abs()
    max_pain  = float(dff.loc[dff["abs_net"].idxmin(), "strike"])
    net_gex   = float(dff["net"].sum())

    levels = {
        "spot": spot, "net_gex": net_gex,
        "gamma_flip": gamma_flip, "call_wall": call_wall,
        "put_wall": put_wall, "max_pain": max_pain,
    }

    # Estimaciones
    close = fetch_close_estimate(ticker, symbol, ratio, levels, all_exps, risk_free)

    return pivot, levels, ratio, exps, close


def fetch_close_estimate(ticker, symbol, ratio, levels, all_exps, risk_free):
    spot    = levels["spot"]
    net_gex = levels.get("net_gex", 0) or 0
    gf      = levels.get("gamma_flip")
    cw      = levels.get("call_wall")
    pw      = levels.get("put_wall")
    today   = datetime.today().date()
    is_expiry = str(today) in all_exps

    # ── OI cercano: solo +-5% del spot (antes era +-8%) ──────────────────────
    nearby = [e for e in all_exps
              if (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= 2]
    if not nearby: nearby = list(all_exps[:2])

    oi_map = {}
    lo_oi, hi_oi = spot * 0.95, spot * 1.05   # <-- reducido de 0.92/1.08
    for exp in nearby[:3]:
        try:
            chain = ticker.option_chain(exp)
        except Exception:
            continue
        for _, row in chain.calls.iterrows():
            k = float(row["strike"])
            if not (lo_oi <= k <= hi_oi): continue
            oi = 0 if str(row.get("openInterest", "nan")) == "nan" else int(row.get("openInterest") or 0)
            oi_map.setdefault(k, {"call": 0, "put": 0})
            oi_map[k]["call"] += oi
        for _, row in chain.puts.iterrows():
            k = float(row["strike"])
            if not (lo_oi <= k <= hi_oi): continue
            oi = 0 if str(row.get("openInterest", "nan")) == "nan" else int(row.get("openInterest") or 0)
            oi_map.setdefault(k, {"call": 0, "put": 0})
            oi_map[k]["put"] += oi

    import pandas as pd
    oi_df = pd.DataFrame([
        {"strike": k, "call_oi": v["call"], "put_oi": v["put"],
         "total_oi": v["call"] + v["put"]} for k, v in oi_map.items()
    ]).sort_values("strike").reset_index(drop=True)

    pin_strike  = float(oi_df.loc[oi_df["total_oi"].idxmax(), "strike"]) if not oi_df.empty else spot
    top3_oi     = oi_df.nlargest(3, "total_oi")[["strike","call_oi","put_oi","total_oi"]].copy()
    top3_oi["strike_spx"] = (top3_oi["strike"] * ratio).round(0).astype(int)

    # Max pain desde OI filtrado
    pain_scores = {}
    for s_c in oi_df["strike"]:
        pain_scores[s_c] = sum(
            max(s_c - r["strike"], 0) * r["call_oi"] +
            max(r["strike"] - s_c, 0) * r["put_oi"]
            for _, r in oi_df.iterrows())
    oi_max_pain = min(pain_scores, key=pain_scores.get) if pain_scores else spot

    # Distancia del pin al spot (para medir su fiabilidad)
    pin_dist_pct = abs(pin_strike - spot) / spot

    # ── MODELO 1: GEX ────────────────────────────────────────────────────────
    # En gamma positivo: Call/Put Wall y Gamma Flip pesan más que el pin OI
    # En gamma negativo: pin OI y max pain pesan más
    m1_targets, m1_weights = [], []

    if net_gex >= 0:
        # Gamma positivo: precio gravita hacia muros de gamma
        if cw: m1_targets.append(cw); m1_weights.append(0.35)
        if pw: m1_targets.append(pw); m1_weights.append(0.20)
        if gf: m1_targets.append(gf); m1_weights.append(0.20)
        # Pin OI pesa menos si está lejos del spot
        pin_w = max(0.05, 0.15 * (1 - pin_dist_pct * 10))
        m1_targets.append(oi_max_pain); m1_weights.append(pin_w)
        m1_targets.append(pin_strike);  m1_weights.append(pin_w * 0.5)
        m1_targets.append(spot);        m1_weights.append(0.05)
    else:
        # Gamma negativo: precio tiende a romper, pin OI más relevante
        m1_targets.append(oi_max_pain); m1_weights.append(0.30)
        m1_targets.append(pin_strike);  m1_weights.append(0.25)
        if gf: m1_targets.append(gf);  m1_weights.append(0.20)
        if cw: m1_targets.append(cw);  m1_weights.append(0.10)
        if pw: m1_targets.append(pw);  m1_weights.append(0.10)
        m1_targets.append(spot);        m1_weights.append(0.05)

    tw = sum(m1_weights)
    m1_spy = sum(t * w for t, w in zip(m1_targets, m1_weights)) / tw
    m1_spx = round(m1_spy * ratio, 0)

    # ── MODELO 2: TECNICO ────────────────────────────────────────────────────
    try:
        h5  = ticker.history(period="1d",  interval="5m")
        hd  = ticker.history(period="30d", interval="1d")
        delta = hd["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = float(100 - (100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])
        ema9  = float(hd["Close"].ewm(span=9,  adjust=False).mean().iloc[-1])
        ema21 = float(hd["Close"].ewm(span=21, adjust=False).mean().iloc[-1])
        if not h5.empty:
            h5["tp"] = (h5["High"] + h5["Low"] + h5["Close"]) / 3
            vwap = float((h5["tp"] * h5["Volume"]).cumsum().iloc[-1] / h5["Volume"].cumsum().iloc[-1])
        else:
            vwap = spot
        bias = 0.0; notes = []
        # RSI: sobrecompra/sobreventa con umbrales más amplios (antes 60/40)
        if rsi > 65:   bias += 0.003; notes.append("RSI {:.0f} sobrecompra".format(rsi))
        elif rsi < 35: bias -= 0.003; notes.append("RSI {:.0f} sobreventa".format(rsi))
        else:          notes.append("RSI {:.0f} neutro".format(rsi))
        if ema9 > ema21: bias += 0.002; notes.append("EMA9>21 alcista")
        else:            bias -= 0.002; notes.append("EMA9<21 bajista")
        dv = (spot - vwap) / vwap
        if dv > 0.003:    bias -= 0.001; notes.append("Sobre VWAP")
        elif dv < -0.003: bias += 0.001; notes.append("Bajo VWAP")
        else:             notes.append("En VWAP")
        m2_spy = spot * (1 + bias)
        m2_note = " | ".join(notes)
        m2_rsi  = rsi
        m2_vwap_spx = round(vwap * ratio, 0)
    except Exception:
        m2_spy = spot; m2_note = "Sin datos técnicos"; m2_rsi = None; m2_vwap_spx = None
    m2_spx = round(m2_spy * ratio, 0)

    # ── MODELO 3: VOLATILIDAD ────────────────────────────────────────────────
    try:
        vix = float(yf.Ticker("^VIX").fast_info["last_price"])
    except Exception:
        vix = 20.0
    try:
        hv = ticker.history(period="25d", interval="1d")
        rel_vol = float(hv["Volume"].iloc[-1]) / float(hv["Volume"].iloc[:-1].mean())
    except Exception:
        rel_vol = 1.0

    v_bias = 0.0; v_notes = []
    if vix > 25:   v_bias -= 0.004; v_notes.append("VIX {:.1f} alto".format(vix))
    elif vix > 18: v_bias -= 0.001; v_notes.append("VIX {:.1f} moderado".format(vix))
    else:          v_bias += 0.001; v_notes.append("VIX {:.1f} bajo".format(vix))

    if net_gex >= 0:
        v_notes.append("GEX+ comprime"); range_pct = max(0.002, 0.005 - net_gex / 200)
    else:
        v_notes.append("GEX- amplifica"); range_pct = min(0.015, 0.005 + abs(net_gex) / 200)

    if rel_vol > 1.4:  v_bias *= 1.2; range_pct *= 1.2; v_notes.append("Vol {:.1f}x alto".format(rel_vol))
    elif rel_vol < 0.7: range_pct *= 0.8; v_notes.append("Vol {:.1f}x bajo".format(rel_vol))

    m3_spy = spot * (1 + v_bias)
    m3_spx = round(m3_spy * ratio, 0)
    m3_note = " | ".join(v_notes)

    # ── MODELO 4: CONSENSO PONDERADO ────────────────────────────────────────
    # Gamma positivo: M1 (GEX/muros) pesa más. Gamma negativo: más equilibrado.
    if net_gex >= 0:
        w1 = 0.50 + (0.05 if is_expiry else 0)   # GEX pesa más en gamma+
        w2 = 0.28
        w3 = 0.22 + (0.05 if vix > 22 else 0)
    else:
        w1 = 0.38 + (0.07 if is_expiry else 0)
        w2 = 0.32
        w3 = 0.30 + (0.07 if vix > 22 else 0)

    tw2 = w1 + w2 + w3
    w1, w2, w3 = w1/tw2, w2/tw2, w3/tw2
    m4_spy = m1_spy * w1 + m2_spy * w2 + m3_spy * w3
    m4_spx = round(m4_spy * ratio, 0)

    all_spx = [m1_spx, m2_spx, m3_spx]
    spx_lo  = round(min(all_spx) * (1 - range_pct * 0.5), 0)
    spx_hi  = round(max(all_spx) * (1 + range_pct * 0.5), 0)
    div     = max(all_spx) - min(all_spx)
    conf    = "Alta" if div < 30 else "Media" if div < 80 else "Baja"

    return {
        "m1_spx": m1_spx, "m1_spy": round(m1_spy, 2), "m1_note": "Muros GEX + Pin OI filtrado ±5%",
        "m2_spx": m2_spx, "m2_spy": round(m2_spy, 2), "m2_note": m2_note,
        "m3_spx": m3_spx, "m3_spy": round(m3_spy, 2), "m3_note": m3_note,
        "m4_spx": m4_spx, "m4_spy": round(m4_spy, 2),
        "spx_low": spx_lo, "spx_high": spx_hi,
        "pin_spx": round(pin_strike * ratio, 0),
        "maxpain_spx": round(oi_max_pain * ratio, 0),
        "pin_dist_pct": round(pin_dist_pct * 100, 1),
        "vix": vix, "rel_vol": rel_vol, "rsi": m2_rsi,
        "vwap_spx": m2_vwap_spx,
        "is_expiry": is_expiry,
        "top_oi": top3_oi,
        "confidence": conf, "divergence": div,
        "weights": (round(w1, 2), round(w2, 2), round(w3, 2)),
        "exps_used": nearby[:3],
    }

def build_narrative(levels, ratio):
    spot    = levels.get("spot", 0)
    net_gex = levels.get("net_gex", 0) or 0
    gf      = levels.get("gamma_flip")
    cw      = levels.get("call_wall")
    pw      = levels.get("put_wall")
    lines   = []

    if net_gex >= 0:
        lines.append("🟢 Gamma POSITIVO: dealers compran caídas y venden subidas → volatilidad amortiguada, precio anclado.")
    else:
        lines.append("🔴 Gamma NEGATIVO: dealers siguen la dirección del precio → movimientos amplificados, mayor volatilidad.")

    if gf:
        spx_gf = gf * ratio
        if spot > gf:
            lines.append("📍 Precio SOBRE Gamma Flip ({:.0f} SPX): zona estable, sesgo alcista a corto plazo.".format(spx_gf))
        else:
            lines.append("⚠️ Precio BAJO Gamma Flip ({:.0f} SPX): zona de riesgo, posible aceleración bajista.".format(spx_gf))

    if cw and pw:
        lines.append("📊 Rango: {:.0f}–{:.0f} SPX (Put Wall soporte / Call Wall resistencia).".format(pw * ratio, cw * ratio))

    if cw and pw and spot:
        du = (cw - spot) / spot
        dd = (spot - pw) / spot
        if dd < du * 0.6:
            lines.append("🔻 Sesgo bajista: precio cerca del soporte → atención a ruptura.")
        elif du < dd * 0.6:
            lines.append("🔺 Sesgo alcista: precio cerca de resistencia → posible rechazo.")
        else:
            lines.append("⚖️ Equilibrio: precio centrado entre muros → esperar ruptura de rango.")

    return lines


# ── GRAFICOS PLOTLY ───────────────────────────────────────────────────────────
COLORS = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "green":   "#2ea043",
    "red":     "#da3633",
    "blue":    "#58a6ff",
    "yellow":  "#e3b341",
    "purple":  "#bc8cff",
    "gray":    "#8b949e",
    "white":   "#f0f6fc",
    "border":  "#30363d",
}

def make_gex_chart(df, levels, ratio, zoom_pct=0.04, threshold=0.02):
    spot = levels["spot"]
    lo   = spot * (1 - zoom_pct)
    hi   = spot * (1 + zoom_pct)
    dfp  = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
    if dfp.empty: dfp = df.copy()

    max_abs = dfp[["call","put"]].abs().max().max()
    if max_abs > 0:
        thr = max_abs * threshold
        dfp = dfp[(dfp["call"].abs() >= thr) | (dfp["put"].abs() >= thr) |
                  ((dfp["strike"] - spot).abs() < spot * 0.005)]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.14,
        shared_xaxes=True,
        subplot_titles=["Gamma Exposure por Strike (USD Bn)", "Net GEX por Strike"],
    )

    # Calls
    dfp["spx_strike"] = (dfp["strike"] * ratio).round(0).astype(int)
    fig.add_trace(go.Bar(
        x=dfp["strike"], y=dfp["call"],
        name="Call GEX", marker_color=COLORS["green"],
        opacity=0.85,
        customdata=dfp["spx_strike"],
        hovertemplate="SPY: $%{x}  (SPX: %{customdata:,})<br>Call GEX: %{y:.3f}B<extra></extra>",
    ), row=1, col=1)

    # Puts
    fig.add_trace(go.Bar(
        x=dfp["strike"], y=dfp["put"],
        name="Put GEX", marker_color=COLORS["red"],
        opacity=0.85,
        customdata=dfp["spx_strike"],
        hovertemplate="SPY: $%{x}  (SPX: %{customdata:,})<br>Put GEX: %{y:.3f}B<extra></extra>",
    ), row=1, col=1)

    # Lineas de niveles
    def add_vline(fig, x, color, name, row):
        fig.add_vline(x=x, line_color=color, line_width=1.5,
                      line_dash="dot", annotation_text=name,
                      annotation_font_color=color,
                      annotation_font_size=10, row=row, col=1)

    add_vline(fig, spot, COLORS["blue"], f"Spot ${spot:.2f} ({spot*ratio:.0f})", 1)
    add_vline(fig, spot, COLORS["blue"], "", 2)

    lvls = [
        ("gamma_flip", COLORS["yellow"], "GFlip"),
        ("call_wall",  COLORS["green"],  "CWall"),
        ("put_wall",   COLORS["red"],    "PWall"),
    ]
    for key, color, label in lvls:
        val = levels.get(key)
        if val and lo <= val <= hi:
            add_vline(fig, val, color, f"{label} ${val:.0f}", 1)
            add_vline(fig, val, color, "", 2)

    # Eje SPX: segundo eje X arriba usando ticktext
    spx_tickvals = dfp["strike"].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).round(0).values
    spx_ticktext = ["{:,.0f}".format(v * ratio) for v in spx_tickvals]

    # Net GEX (fila 2)
    net_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in dfp["net"]]
    fig.add_trace(go.Bar(
        x=dfp["strike"], y=dfp["net"],
        name="Net GEX", marker_color=net_colors,
        opacity=0.82,
        customdata=dfp["spx_strike"],
        hovertemplate="SPY: $%{x}  (SPX: %{customdata:,})<br>Net GEX: %{y:.3f}B<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=620,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["white"], family="JetBrains Mono"),
        legend=dict(
            bgcolor=COLORS["panel"], bordercolor=COLORS["border"],
            orientation="h", y=1.01, x=0,
        ),
        margin=dict(l=50, r=20, t=60, b=40),
        bargap=0.15,
        barmode="relative",
        xaxis=dict(
            tickvals=spx_tickvals.tolist(),
            ticktext=[f"${v:.0f}" for v in spx_tickvals],
            gridcolor=COLORS["border"],
            tickfont=dict(color=COLORS["gray"]),
        ),
        xaxis2=dict(
            gridcolor=COLORS["border"],
            tickfont=dict(color=COLORS["gray"]),
            matches="x",
        ),
        # Eje SPX encima del grafico principal
        xaxis3=dict(
            overlaying="x",
            side="top",
            tickvals=spx_tickvals.tolist(),
            ticktext=spx_ticktext,
            tickfont=dict(color=COLORS["yellow"], size=9),
            title=dict(text="SPX", font=dict(color=COLORS["yellow"], size=10)),
            showgrid=False,
            zeroline=False,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        title_font=dict(color=COLORS["gray"]),
    )
    fig.update_yaxes(
        gridcolor=COLORS["border"], showgrid=True,
        tickfont=dict(color=COLORS["gray"]),
        ticksuffix="B",
    )
    for ann in fig.layout.annotations:
        ann.font.color = COLORS["gray"]
        ann.font.size  = 11

    return fig


def make_model_bars(close, spx_now):
    """Gráfico de barras horizontales con los 3 modelos."""
    models = [
        ("M3 Volatilidad", close["m3_spx"]),
        ("M2 Técnico",     close["m2_spx"]),
        ("M1 GEX",         close["m1_spx"]),
    ]
    colors = [
        COLORS["green"] if v >= spx_now else COLORS["red"]
        for _, v in models
    ]
    fig = go.Figure(go.Bar(
        y=[m[0] for m in models],
        x=[m[1] for m in models],
        orientation="h",
        marker_color=colors,
        opacity=0.8,
        text=["{:.0f}".format(m[1]) for m in models],
        textposition="outside",
        textfont=dict(color=COLORS["white"], size=12, family="JetBrains Mono"),
        hovertemplate="%{y}: %{x:.0f} SPX<extra></extra>",
    ))
    # Linea "now"
    fig.add_vline(x=spx_now, line_color=COLORS["blue"],
                  line_width=2, line_dash="dash",
                  annotation_text="Now {:.0f}".format(spx_now),
                  annotation_font_color=COLORS["blue"])
    # Consenso
    fig.add_vline(x=close["m4_spx"], line_color=COLORS["yellow"],
                  line_width=2, line_dash="dot",
                  annotation_text="Consenso {:.0f}".format(close["m4_spx"]),
                  annotation_font_color=COLORS["yellow"])

    fig.update_layout(
        height=200,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["white"], family="JetBrains Mono", size=11),
        margin=dict(l=100, r=80, t=10, b=10),
        showlegend=False,
        xaxis=dict(
            gridcolor=COLORS["border"],
            tickfont=dict(color=COLORS["gray"]),
            range=[min(close["m1_spx"], close["m2_spx"], close["m3_spx"]) - 50,
                   max(close["m1_spx"], close["m2_spx"], close["m3_spx"]) + 80],
        ),
        yaxis=dict(tickfont=dict(color=COLORS["gray"])),
    )
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-header">GEX</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">GAMMA EXPOSURE DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown("---")

    symbol = st.selectbox(
        "Ticker", ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN"],
        index=0,
        help="SPY replica el S&P 500. Los niveles se convierten a SPX."
    )
    num_exp = st.slider("Expiraciones", min_value=1, max_value=6, value=4,
                        help="1=cierre hoy, 2-3=próximos días, 4=semana, 5-6=mensual")
    zoom_pct = st.slider("Zoom gráfico ±%", min_value=1, max_value=10, value=3,
                         help="Rango de strikes visible alrededor del spot") / 100

    st.markdown("---")
    actualizar = st.button("🔄  Actualizar datos")
    if actualizar:
        st.cache_data.clear()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#8b949e; font-family:'JetBrains Mono',monospace;">
    Datos: Yahoo Finance (delayed)<br>
    Cache: 5 minutos<br>
    GEX: Black-Scholes propio<br><br>
    ⚠️ Solo informativo.<br>
    No es consejo de inversión.
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d  %H:%M")

with st.spinner("Cargando datos de opciones..."):
    try:
        df, levels, ratio, exps, close = fetch_all_data(symbol, num_exp, 0.053)
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

spot    = levels["spot"]
net_gex = levels.get("net_gex", 0) or 0
spx_now = spot * ratio

# ── METRICAS SUPERIORES ───────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
regime_delta = "🟢 Positivo" if net_gex >= 0 else "🔴 Negativo"

c1.metric("SPY Spot",    f"${spot:,.2f}")
c2.metric("SPX Equiv.",  f"{spx_now:,.0f}")
c3.metric("Net GEX",     f"{net_gex:+.2f}B", regime_delta)
c4.metric("Gamma Flip",  f"{levels['gamma_flip']*ratio:,.0f} SPX" if levels.get("gamma_flip") else "N/A")
c5.metric("Call Wall",   f"{levels['call_wall']*ratio:,.0f} SPX"  if levels.get("call_wall")  else "N/A")
c6.metric("Put Wall",    f"{levels['put_wall']*ratio:,.0f} SPX"   if levels.get("put_wall")   else "N/A")

st.markdown("---")

# ── GRAFICO PRINCIPAL ─────────────────────────────────────────────────────────
fig_gex = make_gex_chart(df, levels, ratio, zoom_pct)
st.plotly_chart(fig_gex, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# ── FILA INFERIOR: Niveles | Análisis | Estimaciones ─────────────────────────
col_lvl, col_ana, col_est = st.columns([1, 1.2, 1.5])

# Columna 1: Niveles clave
with col_lvl:
    def fmt(val):
        if not val: return "—"
        return f"${val:,.2f}  ({val*ratio:,.0f})"

    st.markdown(f"""
    <div class="gex-card">
      <h4>📊 Niveles Clave</h4>
      <div class="level-row"><span class="level-label">Spot (SPY/SPX)</span>
        <span style="color:#f0f6fc;font-family:'JetBrains Mono',monospace;font-weight:600">
          ${spot:,.2f} / {spx_now:,.0f}</span></div>
      <div class="level-row"><span class="level-label">Net GEX</span>
        <span style="color:{'#2ea043' if net_gex>=0 else '#da3633'};font-family:'JetBrains Mono',monospace;font-weight:600">
          {net_gex:+.3f}B</span></div>
      <div class="level-row"><span class="level-label">Gamma Flip</span>
        <span style="color:#e3b341;font-family:'JetBrains Mono',monospace;font-weight:600">
          {fmt(levels.get('gamma_flip'))}</span></div>
      <div class="level-row"><span class="level-label">Call Wall</span>
        <span style="color:#2ea043;font-family:'JetBrains Mono',monospace;font-weight:600">
          {fmt(levels.get('call_wall'))}</span></div>
      <div class="level-row"><span class="level-label">Put Wall</span>
        <span style="color:#da3633;font-family:'JetBrains Mono',monospace;font-weight:600">
          {fmt(levels.get('put_wall'))}</span></div>
      <div class="level-row"><span class="level-label">Max Pain</span>
        <span style="color:#bc8cff;font-family:'JetBrains Mono',monospace;font-weight:600">
          {fmt(levels.get('max_pain'))}</span></div>
    </div>
    """, unsafe_allow_html=True)

# Columna 2: Análisis GEX
with col_ana:
    narr = build_narrative(levels, ratio)
    items = "".join([f'<div class="analysis-item">{l}</div>' for l in narr])
    st.markdown(f"""
    <div class="gex-card">
      <h4>🧠 Análisis GEX</h4>
      {items}
    </div>
    """, unsafe_allow_html=True)

    # VIX y datos técnicos
    rsi_str  = f"{close['rsi']:.0f}" if close.get("rsi") else "—"
    vwap_str = f"{close['vwap_spx']:,.0f}" if close.get("vwap_spx") else "—"
    st.markdown(f"""
    <div class="gex-card">
      <h4>📈 Indicadores Técnicos</h4>
      <div class="level-row"><span class="level-label">VIX</span>
        <span style="color:{'#da3633' if close['vix']>25 else '#e3b341' if close['vix']>18 else '#2ea043'};font-family:'JetBrains Mono',monospace;font-weight:600">{close['vix']:.1f}</span></div>
      <div class="level-row"><span class="level-label">RSI (14d)</span>
        <span style="color:#f0f6fc;font-family:'JetBrains Mono',monospace;font-weight:600">{rsi_str}</span></div>
      <div class="level-row"><span class="level-label">VWAP intradía</span>
        <span style="color:#58a6ff;font-family:'JetBrains Mono',monospace;font-weight:600">{vwap_str} SPX</span></div>
      <div class="level-row"><span class="level-label">Vol. Relativo</span>
        <span style="color:#f0f6fc;font-family:'JetBrains Mono',monospace;font-weight:600">{close['rel_vol']:.1f}x</span></div>
      <div class="level-row"><span class="level-label">0DTE hoy</span>
        <span style="color:{'#e3b341' if close['is_expiry'] else '#8b949e'};font-family:'JetBrains Mono',monospace;font-weight:600">{'SI ⚡' if close['is_expiry'] else 'NO'}</span></div>
    </div>
    """, unsafe_allow_html=True)

# Columna 3: Estimaciones
with col_est:
    conf_color = {"Alta": "#2ea043", "Media": "#e3b341", "Baja": "#da3633"}.get(close["confidence"], "#8b949e")
    est_color  = "#2ea043" if close["m4_spx"] >= spx_now else "#da3633"
    dir_arrow  = "▲" if close["m4_spx"] >= spx_now else "▼"

    st.markdown(f"""
    <div class="gex-card">
      <h4>🎯 Estimación Cierre SPX</h4>
      <div style="text-align:center;padding:8px 0 4px 0;">
        <span style="color:{est_color};font-size:2.2rem;font-weight:700;font-family:'JetBrains Mono',monospace;">
          {dir_arrow} {close['m4_spx']:,.0f}</span><br>
        <span style="color:#8b949e;font-size:0.8rem;font-family:'JetBrains Mono',monospace;">
          Rango: {close['spx_low']:,.0f} – {close['spx_high']:,.0f} SPX</span>
      </div>
      <hr style="margin:8px 0;border-color:#30363d">
      <div class="level-row"><span class="level-label">Confianza</span>
        <span style="color:{conf_color};font-family:'JetBrains Mono',monospace;font-weight:600">
          {close['confidence']} (div. {close['divergence']:.0f}pts)</span></div>
      <div class="level-row"><span class="level-label">Pin mayor OI</span>
        <span style="color:#58a6ff;font-family:'JetBrains Mono',monospace;font-weight:600">
          {close['pin_spx']:,.0f} SPX</span></div>
      <div class="level-row"><span class="level-label">Max Pain OI</span>
        <span style="color:#bc8cff;font-family:'JetBrains Mono',monospace;font-weight:600">
          {close['maxpain_spx']:,.0f} SPX</span></div>
      <div class="level-row"><span class="level-label">Pesos M1/M2/M3</span>
        <span style="color:#8b949e;font-family:'JetBrains Mono',monospace;">
          {close['weights'][0]:.0%} / {close['weights'][1]:.0%} / {close['weights'][2]:.0%}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Barras de modelos
    st.markdown('<div class="gex-card"><h4>📉 Modelos Individuales</h4>', unsafe_allow_html=True)
    st.plotly_chart(make_model_bars(close, spx_now),
                    use_container_width=True,
                    config={"displayModeBar": False})

    # Notas por modelo
    for label, note in [("M1 GEX", close["m1_note"]),
                         ("M2 Técnico", close["m2_note"]),
                         ("M3 Volatil.", close["m3_note"])]:
        st.markdown(
            f'<div style="font-size:0.72rem;color:#8b949e;font-family:JetBrains Mono,monospace;'
            f'padding:2px 0"><b style="color:#e3b341">{label}:</b> {note}</div>',
            unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── TOP OI ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sub-header" style="margin-bottom:8px">TOP STRIKES POR OPEN INTEREST (0-2 DTE)</div>', unsafe_allow_html=True)
top = close["top_oi"].copy()
top.columns = ["Strike SPY", "Call OI", "Put OI", "Total OI", "Strike SPX"]
top = top[["Strike SPX", "Strike SPY", "Call OI", "Put OI", "Total OI"]]
top["Strike SPY"] = top["Strike SPY"].apply(lambda x: f"${x:.0f}")
top["Strike SPX"] = top["Strike SPX"].apply(lambda x: f"{x:,.0f}")
for col in ["Call OI", "Put OI", "Total OI"]:
    top[col] = top[col].apply(lambda x: f"{x:,.0f}")
st.dataframe(top, hide_index=True, use_container_width=True)

st.markdown(
    '<div class="warning-box">⚠️ Dashboard informativo basado en datos de Yahoo Finance (delayed). '
    'El GEX se calcula con Black-Scholes propio. Las estimaciones de cierre son probabilísticas '
    'y NO constituyen consejo de inversión.</div>',
    unsafe_allow_html=True
)
