import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from prophet import Prophet

from datos import cargar_datos

# ---------------------------------------------------------------------------
# Configuracion general de la pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pasajeros de avion",
    page_icon="✈️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Carga de datos (con cache para no bajar el CSV en cada recarga)
# ---------------------------------------------------------------------------
@st.cache_data
def obtener_datos():
    return cargar_datos()

df = obtener_datos()

# ---------------------------------------------------------------------------
# TITULO
# ---------------------------------------------------------------------------
st.title("¿Cuántos pasajeros tendremos el próximo año?")
st.markdown("---")

# ---------------------------------------------------------------------------
# INDICADORES SUPERIORES
# ---------------------------------------------------------------------------
pasajeros_1949 = df[df["Month"].dt.year == 1949]["Passengers"].sum()
pasajeros_1960 = df[df["Month"].dt.year == 1960]["Passengers"].sum()
veces_crecio   = round(pasajeros_1960 / pasajeros_1949, 1)

col1, col2, col3 = st.columns(3)
col1.metric(label="Pasajeros en 1949", value=f"{pasajeros_1949:,}")
col2.metric(label="Pasajeros en 1960", value=f"{pasajeros_1960:,}")
col3.metric(label="Cuántas veces creció", value=f"{veces_crecio}x")

st.markdown("---")

# ---------------------------------------------------------------------------
# PESTANAS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "✈️ La historia",
    "📅 La temporada",
    "🔬 El examen",
    "🔮 El pronóstico",
])


# ===========================================================================
# HELPERS — funciones de entrenamiento (con cache para no recalcular)
# ===========================================================================

@st.cache_data
def entrenar_todos(serie_train_values, n_prueba):
    """
    Entrena los 6 modelos con los datos de entrenamiento y devuelve
    un diccionario con las predicciones para el periodo de prueba.
    """
    serie = pd.Series(serie_train_values)
    preds = {}

    # 1. Copiar el año pasado (naive estacional)
    ultimos_12 = serie.iloc[-12:].values
    preds["Copiar el año pasado"] = np.tile(ultimos_12, int(np.ceil(n_prueba / 12)))[:n_prueba]

    # 2. Suavizamiento Exponencial simple (Holt-Winters aditivo)
    hw = ExponentialSmoothing(
        serie, trend="add", seasonal="mul", seasonal_periods=12
    ).fit(optimized=True)
    preds["Holt-Winters"] = hw.forecast(n_prueba).values

    # 3. SARIMA(1,1,1)(1,1,1,12)
    sarima = SARIMAX(
        serie,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    preds["SARIMA"] = sarima.forecast(n_prueba).values

    # 4. Regresion Lineal (variable: numero de mes 1,2,3,...)
    X_train = np.arange(len(serie)).reshape(-1, 1)
    y_train = serie.values
    rl = LinearRegression().fit(X_train, y_train)
    X_pred = np.arange(len(serie), len(serie) + n_prueba).reshape(-1, 1)
    preds["Regresión Lineal"] = rl.predict(X_pred)

    # 5. Prophet
    df_p = pd.DataFrame({
        "ds": pd.date_range(start="1949-01-01", periods=len(serie), freq="MS"),
        "y": serie.values,
    })
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    futuro = m.make_future_dataframe(periods=n_prueba, freq="MS")
    forecast_p = m.predict(futuro)
    preds["Prophet"] = forecast_p.iloc[-n_prueba:]["yhat"].values

    return preds


@st.cache_data
def pronosticar_todos(serie_completa_values, n_forecast):
    """
    Entrena los 6 modelos con TODOS los datos y devuelve
    un diccionario con el pronóstico para n_forecast meses.
    """
    serie = pd.Series(serie_completa_values)
    preds = {}

    # 1. Copiar el año pasado
    ultimos_12 = serie.iloc[-12:].values
    preds["Copiar el año pasado"] = np.tile(ultimos_12, int(np.ceil(n_forecast / 12)))[:n_forecast]

    # 2. Holt-Winters
    hw = ExponentialSmoothing(
        serie, trend="add", seasonal="mul", seasonal_periods=12
    ).fit(optimized=True)
    preds["Holt-Winters"] = hw.forecast(n_forecast).values

    # 3. SARIMA
    sarima = SARIMAX(
        serie,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    preds["SARIMA"] = sarima.forecast(n_forecast).values

    # 4. Regresion Lineal
    X_train = np.arange(len(serie)).reshape(-1, 1)
    y_train = serie.values
    rl = LinearRegression().fit(X_train, y_train)
    X_pred = np.arange(len(serie), len(serie) + n_forecast).reshape(-1, 1)
    preds["Regresión Lineal"] = rl.predict(X_pred)

    # 5. Prophet
    df_p = pd.DataFrame({
        "ds": pd.date_range(start="1949-01-01", periods=len(serie), freq="MS"),
        "y": serie.values,
    })
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    futuro = m.make_future_dataframe(periods=n_forecast, freq="MS")
    forecast_p = m.predict(futuro)
    preds["Prophet"] = forecast_p.iloc[-n_forecast:]["yhat"].values

    return preds


# ---------------------------------------------------------------------------
# PALETA AZULES Y MORADOS
# ---------------------------------------------------------------------------
# Azul marino oscuro para la línea histórica principal
COLOR_HISTORICO = "#1a237e"

# Colores fijos para cada modelo (gama azul-violeta-celeste)
COLORES = {
    "Copiar el año pasado": "#7986cb",   # azul medio
    "Holt-Winters":         "#ce93d8",   # lila suave
    "SARIMA":               "#4fc3f7",   # celeste brillante
    "Regresión Lineal":     "#b39ddb",   # violeta claro
    "Prophet":              "#4dd0e1",   # turquesa azulado
}

# Temporada alta y baja en la paleta
COLOR_ALTA = "#5c6bc0"   # azul índigo (temporada alta)
COLOR_BAJA = "#b39ddb"   # violeta claro (temporada baja)
COLOR_LINEA_NORMAL = "#90a4ae"  # gris azulado para la línea "mes normal"


# ===========================================================================
# PESTANA 1 — La historia
# ===========================================================================
with tab1:
    st.subheader("La historia: pasajeros mes a mes")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Passengers"],
        mode="lines+markers",
        line=dict(color=COLOR_HISTORICO, width=2),
        marker=dict(size=4, color="#5c6bc0"),
        name="Pasajeros",
        fill="tozeroy",
        fillcolor="rgba(92, 107, 192, 0.10)",
    ))
    fig1.update_layout(
        xaxis_title="Mes",
        yaxis_title="Número de pasajeros",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.info(
        "El gráfico muestra cuántos pasajeros viajaron cada mes entre 1949 y 1960. "
        "Se ve que la cantidad crece con los años: el negocio aéreo estuvo en expansión "
        "durante toda esa época. Además, cada año se repite un patrón de picos en verano "
        "y valles en invierno, lo que llamamos 'estacionalidad'."
    )


# ===========================================================================
# PESTANA 2 — La temporada
# ===========================================================================
with tab2:
    st.subheader("La temporada: ¿qué meses son los más ocupados?")

    df["mes_num"] = df["Month"].dt.month
    promedio_mes  = df.groupby("mes_num")["Passengers"].mean().reset_index()
    promedio_mes.columns = ["mes_num", "promedio"]

    nombres_mes = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]
    promedio_mes["nombre"] = promedio_mes["mes_num"].apply(lambda x: nombres_mes[x - 1])
    promedio_general = df["Passengers"].mean()

    colores_bar = [
        COLOR_ALTA if v >= promedio_general else COLOR_BAJA
        for v in promedio_mes["promedio"]
    ]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=promedio_mes["nombre"],
        y=promedio_mes["promedio"],
        marker_color=colores_bar,
        name="Promedio del mes",
    ))
    fig2.add_hline(
        y=promedio_general,
        line_dash="dash",
        line_color=COLOR_LINEA_NORMAL,
        annotation_text="Mes normal",
        annotation_position="top left",
    )
    fig2.update_layout(
        xaxis_title="Mes",
        yaxis_title="Promedio de pasajeros",
        height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "Las barras verdes son los meses con más pasajeros que un mes normal (temporada alta) "
        "y las barras rojas los de menos (temporada baja). "
        "Julio y agosto son los meses pico: coinciden con las vacaciones de verano en el "
        "hemisferio norte. Esta información ayuda a planificar tripulaciones, precios y capacidad."
    )


# ===========================================================================
# PESTANA 3 — El examen
# ===========================================================================
with tab3:
    st.subheader("El examen: ¿qué modelo adivina mejor?")
    st.markdown(
        "Tapamos los **últimos 24 meses** y dejamos que cada modelo los adivine. "
        "Luego medimos el **error promedio en porcentaje** (MAPE): cuanto más bajo, mejor."
    )

    n_prueba  = 24
    df_train  = df.iloc[:-n_prueba].copy()
    df_test   = df.iloc[-n_prueba:].copy()

    with st.spinner("Entrenando modelos... un momento"):
        preds_exam = entrenar_todos(df_train["Passengers"].values, n_prueba)

    # Calcular MAPE de cada modelo
    real = df_test["Passengers"].values
    mapes = {}
    for nombre, pred in preds_exam.items():
        mapes[nombre] = float(np.mean(np.abs((real - pred) / real) * 100))

    ganador = min(mapes, key=mapes.get)

    # --- Grafico ---
    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=df_train["Month"],
        y=df_train["Passengers"],
        mode="lines",
        line=dict(color=COLOR_HISTORICO, width=2),
        name="Datos conocidos",
    ))
    fig3.add_trace(go.Scatter(
        x=df_test["Month"],
        y=df_test["Passengers"],
        mode="lines",
        line=dict(color="#3949ab", width=2, dash="dot"),
        name="Real (tapado)",
    ))
    for nombre, pred in preds_exam.items():
        fig3.add_trace(go.Scatter(
            x=df_test["Month"],
            y=pred,
            mode="lines+markers",
            line=dict(color=COLORES[nombre], width=2, dash="dash"),
            marker=dict(size=5),
            name=nombre,
        ))

    fig3.add_vline(
        x=df_train["Month"].iloc[-1],
        line_dash="dash",
        line_color="#7986cb",
        annotation_text="Aquí se tapa",
        annotation_position="top right",
    )
    fig3.update_layout(
        xaxis_title="Mes",
        yaxis_title="Número de pasajeros",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- Tabla de errores ordenada de mejor a peor ---
    st.markdown("#### Error promedio de cada modelo (menor es mejor)")
    df_mapes = pd.DataFrame(
        [(n, f"{v:.1f}%") for n, v in sorted(mapes.items(), key=lambda x: x[1])],
        columns=["Modelo", "Error promedio (MAPE)"],
    )
    # Resaltar ganador
    df_mapes.insert(0, "Posición", [
        "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else str(i + 1)))
        for i in range(len(df_mapes))
    ])
    st.table(df_mapes)

    st.markdown(f"### Ganador: {ganador}")

    st.info(
        "Cada modelo tiene una estrategia distinta para adivinar el futuro. "
        "SARIMA y Holt-Winters entienden la tendencia y la estacionalidad de la serie. "
        "Prophet fue creado por Meta y es muy bueno con patrones de calendario. "
        "La Regresión Lineal solo ve si los números suben o bajan en línea recta, "
        "sin capturar la temporada, por eso suele quedar más atrás en este tipo de datos."
    )


# ===========================================================================
# PESTANA 4 — El pronostico
# ===========================================================================
with tab4:
    st.subheader("El pronóstico: pasajeros esperados en 1961")
    st.markdown(
        "Ahora entrenamos **todos los modelos con los 144 meses completos** "
        "y proyectamos los 12 meses de 1961."
    )

    with st.spinner("Calculando pronósticos... un momento"):
        preds_fc = pronosticar_todos(df["Passengers"].values, 12)

    fechas_1961 = pd.date_range(start="1961-01-01", periods=12, freq="MS")

    # --- Grafico ---
    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Passengers"],
        mode="lines",
        line=dict(color=COLOR_HISTORICO, width=2),
        name="Histórico (1949–1960)",
        fill="tozeroy",
        fillcolor="rgba(26, 35, 126, 0.08)",
    ))
    for nombre, pred in preds_fc.items():
        fig4.add_trace(go.Scatter(
            x=fechas_1961,
            y=pred.astype(int),
            mode="lines+markers",
            line=dict(color=COLORES[nombre], width=2),
            marker=dict(size=7),
            name=nombre,
        ))

    fig4.add_vline(
        x=df["Month"].iloc[-1],
        line_dash="dash",
        line_color=COLOR_LINEA_NORMAL,
        annotation_text="Inicio del pronóstico",
        annotation_position="top left",
    )
    fig4.update_layout(
        xaxis_title="Mes",
        yaxis_title="Número de pasajeros",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # --- Tabla mes a mes con todos los modelos ---
    st.markdown("#### Pasajeros esperados mes a mes en 1961")

    nombres_mes_es = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]
    tabla_fc = pd.DataFrame({"Mes": nombres_mes_es})
    for nombre, pred in preds_fc.items():
        tabla_fc[nombre] = pred.astype(int)

    st.table(tabla_fc)

    st.info(
        "Cada modelo da una estimación diferente para 1961 según cómo entiende los datos. "
        "Los modelos que captaron mejor la temporada en el examen (pestaña anterior) "
        "suelen dar pronósticos más creíbles. "
        "Usar varios modelos a la vez ayuda a ver en qué rango es probable que caiga "
        "el número real de pasajeros, lo que da más confianza para tomar decisiones de negocio."
    )
