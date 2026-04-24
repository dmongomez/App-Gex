# GEX Dashboard

Dashboard de Gamma Exposure para SPY/SPX con estimación de cierre.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run gex_app.py
```

## Desplegar gratis en Streamlit Cloud (acceso desde móvil)

1. Crea una cuenta en https://github.com y sube estos 3 ficheros a un repositorio nuevo:
   - `gex_app.py`
   - `requirements.txt`
   - `README.md`

2. Ve a https://streamlit.io/cloud → **New app**

3. Conecta tu repositorio de GitHub y selecciona `gex_app.py` como fichero principal

4. Pulsa **Deploy** — en 2-3 minutos tendrás una URL pública que puedes abrir desde el móvil

## Características

- Gráfico interactivo de GEX por strike con zoom y hover
- Eje doble SPY / SPX
- Niveles clave: Gamma Flip, Call Wall, Put Wall, Max Pain
- 3 modelos de estimación de cierre: GEX, Técnico, Volatilidad
- Consenso ponderado adaptativo (0DTE, VIX)
- VIX, RSI, VWAP, Volumen relativo
- Cache de 5 minutos, botón de actualización manual
- Diseño oscuro optimizado para móvil y escritorio
