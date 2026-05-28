"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODELO M1 — SCORING DE CANCELACIONES PROBLEMÁTICAS                        ║
║  Hotel Portoalegre · Golfo de Morrosquillo                                  ║
║  Maestría Analítica Inteligencia de Negocios · PUJ · 2026                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Para correr:
    pip install streamlit pandas numpy scikit-learn imbalanced-learn openpyxl xlrd plotly
    streamlit run app_m1_portoalegre.py

Archivo de datos esperado: reservas_canceladas_2026-03-18.xls
(el mismo usado en el notebook Modelo_M1_Reservas_Canceladas.ipynb)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# ── scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DEL NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
MOTIVOS_PROBLEMATICOS = ["No show", "Pago rechazado", "Cliente sin comunicación"]
MESES_ALTA = [6, 7, 8, 12, 1]
MESES_PRECURSOR = [5, 11]

UMBRAL_ALTA    = 0.25
UMBRAL_PRECUR  = 0.30
UMBRAL_BAJA    = 0.45

FEATURES_NUM = [
    "noches_reserva", "dias_hasta_entrada", "dias_anticipacion_cancelacion",
    "mes_entrada", "dia_semana_entrada", "es_fin_de_semana",
    "hora_cancelacion", "total_cop", "es_temporada_alta",
]
FEATURES_CAT = ["Canal", "Cancelada por", "Habitación"]
FEATURES_GRUPALES = ["cancelaciones_previas_huesped", "score_riesgo_canal"]
TARGET = "target"

COLORES = {
    "azul":   "#1E2761",
    "azul2":  "#4FA3E0",
    "teal":   "#1AAE9F",
    "ambar":  "#F5B642",
    "rojo":   "#E24B4A",
    "gris":   "#6B7280",
    "fondo":  "#F8FAFC",
    "borde":  "#D6DEEC",
}

PARAM_GRID = {
    "clf__n_estimators"     : [300, 500],
    "clf__max_depth"        : [5, 6],
    "clf__min_samples_leaf" : [20, 30],
    "clf__min_samples_split": [30, 50],
    "clf__max_features"     : ["sqrt", 0.5],
    "clf__max_samples"      : [0.7, 0.8],
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="M1 · Portoalegre",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1,h2,h3 { font-family: 'DM Serif Display', serif; }

    .block-container { padding: 2rem 3rem; max-width: 1400px; }

    .metric-card {
        background: white;
        border: 1px solid #D6DEEC;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(30,39,97,.06);
    }
    .metric-card .value  { font-size: 2.2rem; font-weight: 700; color: #1E2761; line-height:1.1; }
    .metric-card .label  { font-size: 0.75rem; font-weight: 600; color: #6B7280;
                           text-transform: uppercase; letter-spacing:.08em; margin-top:.3rem; }
    .metric-card .sub    { font-size: 0.72rem; color: #9CB4D9; margin-top:.2rem; }
    .metric-ok  { border-top: 4px solid #1AAE9F; }
    .metric-warn{ border-top: 4px solid #F5B642; }
    .metric-info{ border-top: 4px solid #4FA3E0; }

    .chip-alta  { background:#FFF3CD; color:#7A4300; border:1px solid #F5B642;
                  border-radius:20px; padding:.15rem .75rem; font-size:.78rem; font-weight:600; }
    .chip-prec  { background:#FEF0E2; color:#7A4300; border:1px solid #EF9F27;
                  border-radius:20px; padding:.15rem .75rem; font-size:.78rem; font-weight:600; }
    .chip-baja  { background:#CFEDE6; color:#0C5C4F; border:1px solid #1AAE9F;
                  border-radius:20px; padding:.15rem .75rem; font-size:.78rem; font-weight:600; }

    .alert-alta { background:#FFF3CD; border-left:4px solid #F5B642;
                  padding:.6rem 1rem; border-radius:0 8px 8px 0; }
    .alert-ok   { background:#CFEDE6; border-left:4px solid #1AAE9F;
                  padding:.6rem 1rem; border-radius:0 8px 8px 0; }
    .alert-info { background:#E7F1FB; border-left:4px solid #4FA3E0;
                  padding:.6rem 1rem; border-radius:0 8px 8px 0; }

    [data-testid="stMetricValue"] { color: #1E2761; font-family: 'DM Sans'; }
    div[data-testid="stSidebar"] { background: #F0F4FB; }
    .stButton > button { background: #1E2761; color: white; border-radius: 8px;
                         border: none; font-weight: 600; padding: .5rem 1.5rem; }
    .stButton > button:hover { background: #4FA3E0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES CORE (fiel al notebook)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cargar_datos(file_bytes: bytes) -> pd.DataFrame:
    """Carga el archivo XLS/XLSX del PMS Lobbybookings.
    Maneja variaciones en nombres de columnas y formatos de exportación.
    """
    try:
        dfs = pd.read_html(io.BytesIO(file_bytes), encoding="latin-1")
        df = dfs[0]
    except Exception:
        df = pd.read_excel(io.BytesIO(file_bytes))

    # ── Limpiar nombres de columna: quitar espacios y saltos de línea ─────────
    df.columns = [str(c).strip().replace("\n", " ").replace("\r", "") for c in df.columns]

    # ── Mapa de renombrado flexible (variantes del PMS Lobbybookings) ─────────
    renombrar = {
        # Motivo — variantes posibles
        "Motivo de cancelación": "Motivo",
        "Motivo Cancelación":    "Motivo",
        "motivo":                "Motivo",
        "MOTIVO":                "Motivo",
        # Total reserva
        "Total de la reserva":   "total_cop",
        "Total Reserva":         "total_cop",
        "Valor reserva":         "total_cop",
        "total reserva":         "total_cop",
        # Canal
        "Canal de venta":        "Canal",
        "Canal Venta":           "Canal",
        # Cancelada por
        "Cancelado por":         "Cancelada por",
        # Habitación
        "Habitacion":            "Habitación",
        "habitación":            "Habitación",
        # Huésped
        "Huesped":               "huésped",
        "Huésped":               "huésped",
        "Cliente":               "huésped",
        # Fechas
        "Fecha Cancelación":     "Fecha cancelación",
        "Fecha de cancelación":  "Fecha cancelación",
        "Fecha Creación":        "Fecha creación",
        "Fecha de creación":     "Fecha creación",
        "Fecha Entrada":         "Entrada",
        "Fecha entrada":         "Entrada",
        "Fecha Salida":          "Salida",
        "Fecha salida":          "Salida",
    }
    df = df.rename(columns=renombrar)

    # ── Si "Motivo" sigue sin existir, buscar columna que contenga "motivo" ───
    if "Motivo" not in df.columns:
        for col in df.columns:
            if "motivo" in col.lower() or "cancelac" in col.lower():
                df = df.rename(columns={col: "Motivo"})
                break

    # ── Si aún no hay "Motivo", crear columna vacía para no romper el flujo ───
    if "Motivo" not in df.columns:
        df["Motivo"] = "Cancelación oportuna del cliente"

    # ── Limpiar total_cop ──────────────────────────────────────────────────────
    if "total_cop" in df.columns and df["total_cop"].dtype == object:
        df["total_cop"] = (
            df["total_cop"].astype(str)
            .str.replace("COP ", "", regex=False)
            .str.replace("$",   "", regex=False)
            .str.replace(".",   "", regex=False)
            .str.replace(",",   ".", regex=False)
            .str.strip()
        )
        df["total_cop"] = pd.to_numeric(df["total_cop"], errors="coerce")

    return df


def parsear_fechas(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Fecha cancelación", "Entrada", "Salida", "Fecha creación"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df


def construir_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = df["Motivo"].apply(
        lambda x: 1 if str(x).strip() in MOTIVOS_PROBLEMATICOS else 0
    )
    return df


def feature_engineering_base(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering base — fiel a celda 22 y 25 del notebook."""
    df = df.sort_values("Fecha creación").reset_index(drop=True)

    df["noches_reserva"]               = (df["Salida"] - df["Entrada"]).dt.days
    df["dias_hasta_entrada"]           = (df["Entrada"] - df["Fecha cancelación"]).dt.days
    df["dias_anticipacion_cancelacion"]= (df["Fecha cancelación"] - df["Fecha creación"]).dt.days
    df["mes_entrada"]                  = df["Entrada"].dt.month
    df["dia_semana_entrada"]           = df["Entrada"].dt.dayofweek
    df["es_fin_de_semana"]             = df["dia_semana_entrada"].isin([4,5,6]).astype(int)
    df["mes_creacion"]                 = df["Fecha creación"].dt.month
    df["hora_cancelacion"]             = df["Fecha cancelación"].dt.hour
    df["es_temporada_alta"]            = df["mes_entrada"].isin(MESES_ALTA).astype(int)

    # Imputar NaN en variables numéricas derivadas de fechas
    # (ocurre cuando Entrada/Salida/Fecha cancelación no parsean)
    for col in ["noches_reserva", "dias_hasta_entrada", "dias_anticipacion_cancelacion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    for col in ["mes_entrada", "dia_semana_entrada", "mes_creacion", "hora_cancelacion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(float)
    df["es_fin_de_semana"]  = df["es_fin_de_semana"].fillna(0).astype(int)
    df["es_temporada_alta"] = df["es_temporada_alta"].fillna(0).astype(int)

    # cancelaciones_previas_huesped — cumcount anticausal (celda 22)
    if "huésped" in df.columns:
        df["cancelaciones_previas_huesped"] = df.groupby("huésped").cumcount()
    else:
        df["cancelaciones_previas_huesped"] = 0

    # Imputar nulos en total_cop con mediana
    if "total_cop" in df.columns:
        df["total_cop"] = pd.to_numeric(df["total_cop"], errors="coerce")
        mediana = df["total_cop"].median()
        df["total_cop"] = df["total_cop"].fillna(mediana if not pd.isna(mediana) else 0)
    else:
        df["total_cop"] = 0

    # Imputar canal desconocido
    for col in ["Canal", "Cancelada por", "Habitación"]:
        if col in df.columns:
            df[col] = df[col].fillna("Desconocido").astype(str)

    return df


def calcular_score_canal(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Score riesgo canal — solo desde train (celda 36, anti-leakage)."""
    canal_riesgo = train_df.groupby("Canal")[TARGET].mean()
    q33 = canal_riesgo.quantile(0.33)
    q66 = canal_riesgo.quantile(0.66)

    def asignar(tasa):
        if tasa >= q66: return 3
        elif tasa >= q33: return 2
        return 1

    canal_map = {c: asignar(t) for c, t in canal_riesgo.items()}
    canal_map["Desconocido"] = 2

    train_df["score_riesgo_canal"] = train_df["Canal"].map(canal_map).fillna(2).astype(int)
    test_df["score_riesgo_canal"]  = test_df["Canal"].map(canal_map).fillna(2).astype(int)
    return train_df, test_df, canal_map


def preparar_xy(train_df, test_df):
    """One-Hot Encoding + alineación de columnas — celda 37."""
    FEATURES_FINAL = FEATURES_NUM + FEATURES_GRUPALES + FEATURES_CAT

    # Imputar NaN en columnas numéricas ANTES del encoding
    # (pueden aparecer cuando fechas no parsean bien en el archivo del PMS)
    for col in FEATURES_NUM + FEATURES_GRUPALES:
        if col in train_df.columns:
            mediana = train_df[col].median()
            mediana = 0 if pd.isna(mediana) else mediana
            train_df[col] = train_df[col].fillna(mediana)
        if col in test_df.columns:
            mediana = train_df[col].median() if col in train_df.columns else 0
            mediana = 0 if pd.isna(mediana) else mediana
            test_df[col] = test_df[col].fillna(mediana)

    # Imputar NaN en columnas categóricas
    for col in FEATURES_CAT:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("Desconocido").astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("Desconocido").astype(str)

    train_enc = pd.get_dummies(train_df[FEATURES_FINAL + [TARGET]], columns=FEATURES_CAT, drop_first=True)
    test_enc  = pd.get_dummies(test_df[FEATURES_FINAL  + [TARGET]], columns=FEATURES_CAT, drop_first=True)

    X_train = train_enc.drop(columns=[TARGET])
    y_train = train_enc[TARGET]
    X_test  = test_enc.drop(columns=[TARGET])
    y_test  = test_enc[TARGET]

    # Alinear columnas y rellenar cualquier NaN residual con 0
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # Reemplazar infinitos si los hubiera (ej. divisiones en features derivadas)
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test  = X_test.replace([np.inf, -np.inf], 0)

    return X_train, y_train, X_test, y_test


@st.cache_data(show_spinner=False)
def entrenar_modelo(_X_train: pd.DataFrame, _y_train: pd.Series,
                    modo: str, column_hash: str):
    """Entrena RF con SMOTE pipeline. modo='rapido' o 'optimizado'.
    column_hash: hash de los nombres de columnas de X_train para invalidar
    el caché cuando cambia la estructura del archivo cargado.
    """
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    if modo == "optimizado":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs  = GridSearchCV(pipe, PARAM_GRID, cv=skf, scoring="f1_weighted",
                           n_jobs=-1, verbose=0)
        gs.fit(_X_train, _y_train)
        modelo = gs.best_estimator_
        params = gs.best_params_
    else:
        params_base = {
            "clf__n_estimators": 300, "clf__max_depth": 6,
            "clf__min_samples_leaf": 20, "clf__min_samples_split": 50,
            "clf__max_features": 0.5, "clf__max_samples": 0.8,
        }
        pipe.set_params(**params_base)
        pipe.fit(_X_train, _y_train)
        modelo = pipe
        params = params_base

    return modelo, params


def umbral_por_temporada(mes: int) -> float:
    if mes in MESES_ALTA:    return UMBRAL_ALTA
    if mes in MESES_PRECURSOR: return UMBRAL_PRECUR
    return UMBRAL_BAJA


def clasificar_riesgo(score: float, mes: int) -> str:
    umbral = umbral_por_temporada(mes)
    if score >= umbral:
        if score >= 0.60: return "Alto"
        return "Medio"
    return "Bajo"


def color_riesgo(r: str) -> str:
    return {"Alto": "#E24B4A", "Medio": "#F5B642", "Bajo": "#1AAE9F"}.get(r, "#6B7280")


def formatear_cop(v: float) -> str:
    return f"$ {v:,.0f}".replace(",", ".")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏨 M1 · Portoalegre")
    st.markdown("**Propensión a Cancelación Problemática**")
    st.markdown("---")

    archivo = st.file_uploader(
        "📂 Cargar reservas_canceladas.xls",
        type=["xls", "xlsx"],
        help="Exporta desde Lobbybookings: Reservas → Canceladas → Exportar a Excel",
    )

    st.markdown("### ⚙️ Configuración del modelo")
    modo_entrenamiento = st.radio(
        "Modo de entrenamiento",
        ["Rápido (params fijos)", "Optimizado (GridSearchCV 5-fold)"],
        index=0,
        help="Rápido usa los hiperparámetros del modelo final del notebook. Optimizado hace GridSearchCV completo (~3 min).",
    )
    modo = "rapido" if "Rápido" in modo_entrenamiento else "optimizado"

    st.markdown("### 📅 Proyección")
    anio_proyeccion = st.number_input("Año proyección", min_value=2025, max_value=2030, value=2026)
    inflacion       = st.slider("Inflación anual (%)", 0.0, 10.0, 2.5, 0.1) / 100

    st.markdown("---")
    st.markdown("""
    <small>
    Maestría Analítica para BI · PUJ<br>
    Higuera · Ibarra · Balen · Jerez<br>
    Tutor: Tirado Cifuentes · 2026
    </small>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:1.5rem;">
  <p style="color:#4FA3E0;font-size:.85rem;font-weight:600;letter-spacing:.12em;margin:0;">
    MODELO 1 · HOTEL PORTOALEGRE · COVEÑAS
  </p>
  <h1 style="margin:0;font-size:2.2rem;color:#1E2761;">
    Scoring de Cancelaciones Problemáticas
  </h1>
  <p style="color:#6B7280;margin:.3rem 0 0 0;font-size:.95rem;">
    Random Forest · SMOTE · Split temporal 70/30 · Umbrales dinámicos por temporada
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FLUJO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
if archivo is None:
    st.markdown("""
    <div class="alert-info">
    <b>📋 Cómo usar esta app</b><br>
    1. Exporta las reservas canceladas desde <b>Lobbybookings → Reservas → Canceladas → Exportar Excel</b><br>
    2. Sube el archivo .xls en el panel izquierdo<br>
    3. Selecciona el modo de entrenamiento<br>
    4. El modelo se entrena, evalúa y puntúa todas las reservas automáticamente
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Umbrales de intervención por temporada")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card metric-warn">
        <div class="value">0.25</div>
        <div class="label">Temporada Alta</div>
        <div class="sub">Jun·Jul·Ago·Dic·Ene — máxima sensibilidad</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card metric-warn">
        <div class="value">0.30</div>
        <div class="label">Meses Precursores</div>
        <div class="sub">Nov · May — umbral intermedio</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card metric-ok">
        <div class="value">0.45</div>
        <div class="label">Media / Baja</div>
        <div class="sub">Resto del año — umbral conservador</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Cargar y preparar datos ────────────────────────────────────────────────────
with st.spinner("📂 Cargando datos del PMS..."):
    df_raw = cargar_datos(archivo.read())

    # ── Diagnóstico: mostrar columnas si falta alguna clave ───────────────────
    cols_requeridas = ["Motivo", "Entrada", "Salida", "Fecha cancelación", "Fecha creación"]
    cols_faltantes  = [c for c in cols_requeridas if c not in df_raw.columns]
    if cols_faltantes:
        st.warning(f"⚠️ Columnas no encontradas: **{cols_faltantes}**  \n"
                   f"Columnas detectadas en el archivo: `{list(df_raw.columns)}`  \n"
                   f"Verifica que el archivo sea el export correcto de Lobbybookings.")

    # ── Si no hay Fecha creación, usar Fecha cancelación como proxy ───────────
    if "Fecha creación" not in df_raw.columns and "Fecha cancelación" in df_raw.columns:
        df_raw["Fecha creación"] = df_raw["Fecha cancelación"]

    df = parsear_fechas(df_raw.copy())
    df = construir_target(df)
    df = feature_engineering_base(df)

n_total  = len(df)
n_prob   = df["target"].sum()
pct_prob = n_prob / n_total * 100

st.success(f"✅ Datos cargados: **{n_total:,} reservas** | **{n_prob:,} problemáticas** ({pct_prob:.1f}%)")

# ── Split temporal 70/30 ───────────────────────────────────────────────────────
df_sorted = df.sort_values("Fecha creación").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.70)
train_df  = df_sorted.iloc[:split_idx].copy()
test_df   = df_sorted.iloc[split_idx:].copy()
fecha_corte = test_df["Fecha creación"].min().date()

# ── Features grupales post-split (anti-leakage) ────────────────────────────────
# cancelaciones_previas_huesped en test = máximo acumulado en train
if "huésped" in train_df.columns:
    conteo_train = train_df.groupby("huésped")["cancelaciones_previas_huesped"].max()
    test_df["cancelaciones_previas_huesped"] = test_df["huésped"].map(conteo_train).fillna(0)

train_df, test_df, canal_map = calcular_score_canal(train_df, test_df)

# ── Preparar X/y ──────────────────────────────────────────────────────────────
X_train, y_train, X_test, y_test = preparar_xy(train_df, test_df)

with st.spinner(f"🌲 Entrenando Random Forest ({modo})... esto puede tomar unos segundos."):
    # column_hash invalida el caché si cambia la estructura de columnas del archivo
    column_hash = str(sorted(X_train.columns.tolist()))
    modelo, params_usados = entrenar_modelo(X_train, y_train, modo, column_hash)

# ── Predicciones en test ───────────────────────────────────────────────────────
# Forzar que X_test tenga exactamente las columnas con las que el modelo fue entrenado
# (el modelo sklearn guarda feature_names_in_ internamente)
try:
    feature_names_model = modelo.named_steps["clf"].feature_names_in_
    X_test  = X_test.reindex(columns=feature_names_model, fill_value=0)
    X_train = X_train.reindex(columns=feature_names_model, fill_value=0)
except AttributeError:
    # Si el modelo no tiene feature_names_in_, alinear con X_train directamente
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

X_test  = X_test.fillna(0).replace([np.inf, -np.inf], 0)
X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)

y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

auc    = roc_auc_score(y_test, y_proba)
recall = recall_score(y_test, y_pred)
f1w    = f1_score(y_test, y_pred, average="weighted")
acc    = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0

# Gap train
y_train_pred  = modelo.predict(X_train)
y_train_proba = modelo.predict_proba(X_train)[:, 1]
auc_train = roc_auc_score(y_train, y_train_proba)
gap_auc   = auc_train - auc

# Umbral óptimo Youden
fpr_r, tpr_r, thr_r = roc_curve(y_test, y_proba)
j_stat   = tpr_r - fpr_r
opt_idx  = np.argmax(j_stat)
opt_thr  = thr_r[opt_idx]

# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Evaluación del modelo",
    "🎯 Scoring de reservas",
    "💰 Impacto financiero",
    "📈 Análisis exploratorio",
    "⚙️ Detalle técnico",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — EVALUACIÓN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("### Métricas en conjunto de prueba (30% más reciente)")
    st.caption(f"Fecha de corte: **{fecha_corte}** · Train: {len(X_train):,} reg. · Test: {len(X_test):,} reg.")

    c1,c2,c3,c4,c5 = st.columns(5)
    def mcard(col, val, lbl, sub, cls="metric-info"):
        col.markdown(f"""<div class="metric-card {cls}">
        <div class="value">{val}</div>
        <div class="label">{lbl}</div>
        <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    mcard(c1, f"{auc:.4f}", "AUC-ROC",     "meta ≥ 0.75 ✓" if auc>=0.75 else "meta ≥ 0.75 ✗",
          "metric-ok" if auc>=0.75 else "metric-warn")
    mcard(c2, f"{recall:.4f}", "Recall",   "meta ≥ 0.70 ✓" if recall>=0.70 else "meta ≥ 0.70 ✗",
          "metric-ok" if recall>=0.70 else "metric-warn")
    mcard(c3, f"{precision_pos:.4f}", "Precisión", f"{fp} FP / {len(y_test):,} reservas", "metric-info")
    mcard(c4, f"{f1w:.4f}", "F1 pond.", "ponderado por clase", "metric-info")
    mcard(c5, f"{gap_auc:.4f}", "Gap AUC",
          "✓ generaliza bien" if gap_auc<0.10 else "⚠ overfitting moderado",
          "metric-ok" if gap_auc<0.10 else "metric-warn")

    st.markdown("---")
    col_roc, col_cm = st.columns([1.1, 1])

    with col_roc:
        st.markdown("#### Curva ROC")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_r, y=tpr_r, mode="lines",
            line=dict(color=COLORES["azul2"], width=2.5),
            name=f"RF · AUC = {auc:.4f}"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#CCCCCC", dash="dash", width=1), name="Aleatorio", showlegend=False))
        fig_roc.add_trace(go.Scatter(
            x=[fpr_r[opt_idx]], y=[tpr_r[opt_idx]], mode="markers",
            marker=dict(color=COLORES["rojo"], size=10),
            name=f"Umbral Youden = {opt_thr:.3f}"))
        fig_roc.update_layout(
            xaxis_title="Falsos Positivos", yaxis_title="Verdaderos Positivos",
            height=340, margin=dict(l=40,r=20,t=20,b=40),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(x=0.55, y=0.05, bgcolor="rgba(0,0,0,0)")
        )
        fig_roc.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
        fig_roc.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown("#### Matriz de Confusión")
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp],[fn, tp]],
            x=["Pred: No prob.", "Pred: Problemática"],
            y=["Real: No prob.", "Real: Problemática"],
            colorscale=[[0,"#E7F1FB"],[1,COLORES["azul"]]],
            text=[[f"TN={tn}", f"FP={fp}"],[f"FN={fn}", f"TP={tp}"]],
            texttemplate="%{text}", textfont=dict(size=16, color="white"),
            showscale=False,
        ))
        fig_cm.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=40),
                              plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown(f"""
        <div class="alert-ok">
        El modelo identifica <b>{tp}</b> de {tp+fn} cancelaciones problemáticas reales.<br>
        Solo <b>{fp}</b> falsos positivos en {len(y_test):,} reservas evaluadas.
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — SCORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("### Scoring de reservas del conjunto de prueba")
    st.caption("Clasificación con umbrales dinámicos por temporada: Alta 0.25 · Precursor 0.30 · Media/Baja 0.45")

    # Construir tabla de scoring
    df_score = test_df[["Fecha cancelación", "Entrada", "Canal",
                         "Habitación", "total_cop", "mes_entrada",
                         "target"]].copy().reset_index(drop=True)
    df_score["score"]  = np.round(y_proba, 4)
    df_score["umbral"] = df_score["mes_entrada"].apply(umbral_por_temporada)
    df_score["riesgo"] = df_score.apply(
        lambda r: clasificar_riesgo(r["score"], r["mes_entrada"]), axis=1)
    df_score["real"]   = df_score["target"].map({1:"Problemática", 0:"No prob."})

    # Filtros
    f1, f2, f3 = st.columns(3)
    filtro_riesgo = f1.multiselect("Nivel de riesgo", ["Alto","Medio","Bajo"],
                                    default=["Alto","Medio"])
    filtro_canal  = f2.multiselect("Canal",
                                    sorted(df_score["Canal"].dropna().unique().tolist()),
                                    default=[])
    orden_score   = f3.radio("Ordenar por", ["Score ↓","COP ↓"], horizontal=True)

    df_vis = df_score.copy()
    if filtro_riesgo:
        df_vis = df_vis[df_vis["riesgo"].isin(filtro_riesgo)]
    if filtro_canal:
        df_vis = df_vis[df_vis["Canal"].isin(filtro_canal)]
    if orden_score == "Score ↓":
        df_vis = df_vis.sort_values("score", ascending=False)
    else:
        df_vis = df_vis.sort_values("total_cop", ascending=False)

    # KPIs de scoring
    n_alto = (df_score["riesgo"]=="Alto").sum()
    n_medio= (df_score["riesgo"]=="Medio").sum()
    n_bajo = (df_score["riesgo"]=="Bajo").sum()
    ka,km,kb,ktot = st.columns(4)
    ka.metric("🔴 Riesgo Alto",   f"{n_alto}",  help="Score ≥ umbral + 0.35")
    km.metric("🟡 Riesgo Medio",  f"{n_medio}", help="Score ≥ umbral de temporada")
    kb.metric("🟢 Riesgo Bajo",   f"{n_bajo}",  help="Score < umbral de temporada")
    ktot.metric("Total evaluadas", f"{len(df_score):,}")

    # Tabla
    df_tabla = df_vis[["Entrada","Canal","Habitación","total_cop","score","umbral","riesgo","real"]].copy()
    df_tabla.columns = ["Entrada","Canal","Habitación","COP reserva","Score","Umbral","Riesgo","Real"]
    df_tabla["COP reserva"] = df_tabla["COP reserva"].apply(
        lambda x: f"$ {x:,.0f}".replace(",","."))
    df_tabla["Score"] = df_tabla["Score"].map("{:.3f}".format)
    df_tabla["Umbral"]= df_tabla["Umbral"].map("{:.2f}".format)
    df_tabla["Entrada"] = pd.to_datetime(df_tabla["Entrada"]).dt.strftime("%Y-%m-%d")

    st.dataframe(
        df_tabla.reset_index(drop=True),
        use_container_width=True,
        height=400,
        column_config={
            "Riesgo": st.column_config.TextColumn("Riesgo"),
            "Score":  st.column_config.TextColumn("Score M1"),
        }
    )

    # Exportar
    csv = df_vis.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Descargar scoring completo (.csv)", csv,
                       "scoring_m1_portoalegre.csv", "text/csv")

    # Distribución de scores por riesgo
    st.markdown("#### Distribución de scores")
    fig_hist = go.Figure()
    for nivel, color in [("Alto",COLORES["rojo"]),("Medio",COLORES["ambar"]),("Bajo",COLORES["teal"])]:
        sub = df_score[df_score["riesgo"]==nivel]["score"]
        fig_hist.add_trace(go.Histogram(x=sub, name=nivel,
            marker_color=color, opacity=0.75, nbinsx=30))
    fig_hist.add_vline(x=UMBRAL_ALTA,   line_dash="dot", line_color=COLORES["rojo"],
                       annotation_text="Alta 0.25", annotation_position="top right")
    fig_hist.add_vline(x=UMBRAL_PRECUR, line_dash="dot", line_color=COLORES["ambar"],
                       annotation_text="Precursor 0.30")
    fig_hist.add_vline(x=UMBRAL_BAJA,   line_dash="dot", line_color=COLORES["teal"],
                       annotation_text="Baja 0.45")
    fig_hist.update_layout(barmode="overlay", height=300,
        xaxis_title="Score", yaxis_title="Reservas",
        margin=dict(l=40,r=20,t=30,b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=1.12))
    fig_hist.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
    st.plotly_chart(fig_hist, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — IMPACTO FINANCIERO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### Simulación contrafactual — Impacto económico")
    st.caption("Metodología: Fórmula F1 del documento · COP_histórico vs COP_intervenido · Fiel a celda 71 del notebook")

    # ── Simulación sobre test (fiel a celda 71) ────────────────────────────────
    df_test_ob3 = test_df.copy().reset_index(drop=True)
    df_test_ob3["target_r"] = y_test.values
    df_test_ob3["pred_r"]   = y_pred
    df_test_ob3["prob_r"]   = y_proba

    prob_mask = df_test_ob3["target_r"] == 1
    df_prob   = df_test_ob3[prob_mask].copy()

    cop_en_riesgo        = df_prob["total_cop"].sum()
    cop_fn               = df_prob[df_prob["pred_r"] == 0]["total_cop"].sum()   # no detectados
    cop_tp               = df_prob[df_prob["pred_r"] == 1]["total_cop"].sum()   # detectados
    reduccion_pct        = (cop_tp / cop_en_riesgo * 100) if cop_en_riesgo > 0 else 0
    recall_ob3           = recall_score(df_test_ob3["target_r"], df_test_ob3["pred_r"])

    # ── Proyección inflacionaria (celda 73) ────────────────────────────────────
    anio_base = test_df["Fecha cancelación"].dt.year.mode()[0] if "Fecha cancelación" in test_df else 2025
    df_base_proy = df[
        (df["target"]==1) & (df["Fecha cancelación"].dt.year==anio_base)
    ]["total_cop"]
    cop_base_proy   = df_base_proy.sum()
    cop_proyectado  = cop_base_proy * (1 + inflacion)
    oportunidad_proy= cop_proyectado * recall_ob3
    cop_no_id_proy  = cop_proyectado * (1 - recall_ob3)

    # KPIs financieros
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("COP en riesgo (test)",    formatear_cop(cop_en_riesgo))
    k2.metric("COP detectado (TP)",      formatear_cop(cop_tp),
              delta=f"+{reduccion_pct:.1f}% recuperable")
    k3.metric("COP no detectado (FN)",   formatear_cop(cop_fn))
    k4.metric(f"Oportunidad {anio_proyeccion}", formatear_cop(oportunidad_proy),
              delta=f"Inflación +{inflacion*100:.1f}%")

    st.markdown("---")
    col_contra, col_proy = st.columns([1,1])

    with col_contra:
        st.markdown("#### Contrafactual 2025 (test)")
        fig_contra = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","relative","total"],
            x=["COP en riesgo","Detectado (TP)","No detectado (FN)","Total en riesgo"],
            y=[cop_en_riesgo, -cop_tp, cop_fn, 0],
            text=[formatear_cop(cop_en_riesgo), f"−{formatear_cop(cop_tp)}",
                  formatear_cop(cop_fn), formatear_cop(cop_fn)],
            textposition="outside",
            connector=dict(line=dict(color="#D6DEEC")),
            decreasing=dict(marker=dict(color=COLORES["teal"])),
            increasing=dict(marker=dict(color=COLORES["rojo"])),
            totals=dict(marker=dict(color=COLORES["azul"])),
        ))
        fig_contra.update_layout(height=320, margin=dict(l=20,r=20,t=30,b=40),
                                  plot_bgcolor="white", paper_bgcolor="white",
                                  showlegend=False)
        st.plotly_chart(fig_contra, use_container_width=True)
        st.markdown(f"""
        <div class="alert-ok">
        <b>Reducción estimada del COP perdido: {reduccion_pct:.1f}%</b><br>
        Umbral de éxito del proyecto: ≥ 5% · 
        Resultado: <b>{reduccion_pct/5:.1f}×</b> el umbral mínimo
        </div>
        """, unsafe_allow_html=True)

    with col_proy:
        st.markdown(f"#### Proyección {anio_proyeccion} (inflación {inflacion*100:.1f}%)")
        fig_proy = go.Figure(go.Bar(
            x=["COP proyectado\nen riesgo", f"Oportunidad\npotencial {anio_proyeccion}",
               "COP no\nidentificado"],
            y=[cop_proyectado, oportunidad_proy, cop_no_id_proy],
            marker_color=[COLORES["azul"], COLORES["teal"], COLORES["rojo"]],
            text=[formatear_cop(v) for v in [cop_proyectado, oportunidad_proy, cop_no_id_proy]],
            textposition="outside",
        ))
        fig_proy.update_layout(height=320, margin=dict(l=20,r=20,t=30,b=60),
                                plot_bgcolor="white", paper_bgcolor="white",
                                yaxis_title="COP", showlegend=False)
        fig_proy.update_xaxes(showgrid=False)
        fig_proy.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
        st.plotly_chart(fig_proy, use_container_width=True)
        st.markdown(f"""
        <div class="alert-info">
        <b>Estimación de potencial — no ingreso garantizado</b><br>
        Recall validado en test ({recall_ob3:.3f}) × COP proyectado con inflación {inflacion*100:.1f}%
        (meta Banco de la República)
        </div>
        """, unsafe_allow_html=True)

    # Análisis por temporada
    st.markdown("#### Impacto financiero por temporada — Hallazgo H3")
    df_temp_fin = df[df["target"]==1].copy()
    df_temp_fin["temporada"] = df_temp_fin["mes_entrada"].apply(
        lambda m: "Alta" if m in MESES_ALTA else "Media/Baja")
    temp_stats = df_temp_fin.groupby("temporada").agg(
        n=("total_cop","count"),
        cop_total=("total_cop","sum"),
        cop_promedio=("total_cop","mean")
    ).reset_index()
    temp_stats["tasa_%"] = (df.groupby(
        df["mes_entrada"].apply(lambda m: "Alta" if m in MESES_ALTA else "Media/Baja")
    )["target"].mean().values * 100)

    fig_temp = make_subplots(rows=1, cols=2,
        subplot_titles=["Tasa problemática (%)", "COP perdido promedio por reserva"])
    colores_temp = [COLORES["rojo"], COLORES["azul2"]]
    for i, (_, row) in enumerate(temp_stats.iterrows()):
        color = colores_temp[i % len(colores_temp)]
        tasa_val     = row["tasa_%"]
        cop_prom_val = row["cop_promedio"]
        fig_temp.add_trace(go.Bar(x=[row["temporada"]], y=[tasa_val],
            marker_color=color, name=row["temporada"],
            text=[f"{tasa_val:.1f}%"], textposition="outside"), row=1, col=1)
        fig_temp.add_trace(go.Bar(x=[row["temporada"]], y=[cop_prom_val],
            marker_color=color, showlegend=False,
            text=[formatear_cop(cop_prom_val)], textposition="outside"), row=1, col=2)
    fig_temp.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=40),
                            plot_bgcolor="white", paper_bgcolor="white",
                            showlegend=False, barmode="group")
    for ax in ["xaxis","xaxis2"]:
        fig_temp.update_layout(**{ax: dict(showgrid=False)})
    st.plotly_chart(fig_temp, use_container_width=True)
    st.markdown("""
    <div class="alert-alta">
    <b>Hallazgo H3:</b> Temporada Alta tiene <i>menor tasa</i> de cancelación problemática pero
    <b>~42% más COP perdido por evento</b>. La priorización de intervenciones debe guiarse
    por impacto financiero, no por frecuencia.
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — EDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("### Análisis exploratorio")
    ea1, ea2 = st.columns(2)

    with ea1:
        st.markdown("#### Distribución de motivos de cancelación")
        motivos = df["Motivo"].value_counts().reset_index()
        motivos.columns = ["Motivo","n"]
        motivos["es_prob"] = motivos["Motivo"].isin(MOTIVOS_PROBLEMATICOS)
        fig_mot = px.bar(motivos.head(10), x="n", y="Motivo", orientation="h",
            color="es_prob",
            color_discrete_map={True: COLORES["rojo"], False: COLORES["azul2"]},
            text="n")
        fig_mot.update_layout(height=320, margin=dict(l=10,r=20,t=20,b=40),
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_mot, use_container_width=True)

    with ea2:
        st.markdown("#### Tasa problemática por canal (score riesgo)")
        if "Canal" in df.columns:
            canal_r = df.groupby("Canal")["target"].agg(["mean","count"]).reset_index()
            canal_r.columns = ["Canal","tasa","n"]
            canal_r = canal_r[canal_r["n"]>=5].sort_values("tasa", ascending=False).head(12)
            canal_r["score"] = canal_r["Canal"].map(canal_map).fillna(2)
            canal_r["color"] = canal_r["score"].map(
                {3:COLORES["rojo"], 2:COLORES["ambar"], 1:COLORES["teal"]})
            fig_canal = go.Figure(go.Bar(
                x=(canal_r["tasa"]*100).round(1),
                y=canal_r["Canal"],
                orientation="h",
                marker_color=canal_r["color"].tolist(),
                text=(canal_r["tasa"]*100).round(1).astype(str)+"%",
                textposition="outside",
            ))
            fig_canal.update_layout(height=320, margin=dict(l=10,r=40,t=20,b=40),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="% problemáticas",
                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_canal, use_container_width=True)

    # Estacionalidad de cancelaciones
    st.markdown("#### Estacionalidad mensual de cancelaciones")
    df["mes_cancel"] = df["Fecha cancelación"].dt.to_period("M").astype(str)
    estac = df.groupby("mes_cancel").agg(
        total=("target","count"), prob=("target","sum")).reset_index()
    estac["tasa"] = (estac["prob"]/estac["total"]*100).round(1)

    fig_estac = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Volumen mensual","Tasa problemática (%)"],
        row_heights=[0.6,0.4])
    fig_estac.add_trace(go.Bar(x=estac["mes_cancel"], y=estac["total"],
        name="No prob.", marker_color=COLORES["azul2"], opacity=0.7), row=1, col=1)
    fig_estac.add_trace(go.Bar(x=estac["mes_cancel"], y=estac["prob"],
        name="Problemática", marker_color=COLORES["rojo"], opacity=0.85), row=1, col=1)
    fig_estac.add_trace(go.Scatter(x=estac["mes_cancel"], y=estac["tasa"],
        mode="lines+markers", line=dict(color=COLORES["ambar"], width=2),
        marker=dict(size=5), name="Tasa %"), row=2, col=1)
    fig_estac.add_hline(y=estac["tasa"].mean(), line_dash="dash",
        line_color=COLORES["gris"], annotation_text=f"Prom. {estac['tasa'].mean():.0f}%",
        row=2, col=1)
    fig_estac.update_layout(height=400, barmode="stack",
        margin=dict(l=20,r=20,t=40,b=40),
        plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_estac, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — DETALLE TÉCNICO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("### Detalle técnico del modelo")

    dt1, dt2 = st.columns(2)
    with dt1:
        st.markdown("#### Hiperparámetros usados")
        params_df = pd.DataFrame(
            list({k.replace("clf__",""):v for k,v in params_usados.items()}.items()),
            columns=["Parámetro","Valor"]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        st.markdown("#### Features del modelo")
        feat_df = pd.DataFrame({
            "Variable": FEATURES_NUM + FEATURES_GRUPALES,
            "Tipo":     ["Numérica"]*len(FEATURES_NUM) + ["Grupal (post-split)"]*len(FEATURES_GRUPALES),
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with dt2:
        st.markdown("#### Importancia de variables (Gini — top 20)")
        rf_clf = modelo.named_steps["clf"]
        feat_names = X_train.columns.tolist()
        gini_imp = pd.DataFrame({
            "feature":    feat_names,
            "importance": rf_clf.feature_importances_
        }).sort_values("importance", ascending=False).head(20)

        fig_gini = go.Figure(go.Bar(
            x=gini_imp["importance"],
            y=gini_imp["feature"],
            orientation="h",
            marker_color=COLORES["azul2"],
        ))
        fig_gini.update_layout(height=460, margin=dict(l=10,r=20,t=10,b=40),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Importancia Gini",
            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_gini, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Resumen CRISP-DM — Criterios de éxito")
    resumen = pd.DataFrame([
        {"Objetivo": "OB1 · AUC-ROC ≥ 0.75", "Resultado": f"{auc:.4f}",
         "Estado": "✅ Cumple" if auc>=0.75 else "❌ No cumple",
         "Margen": f"+{(auc-0.75)*100:.1f} pp"},
        {"Objetivo": "OB2 · Recall ≥ 0.70", "Resultado": f"{recall:.4f}",
         "Estado": "✅ Cumple" if recall>=0.70 else "❌ No cumple",
         "Margen": f"+{(recall-0.70)*100:.1f} pp"},
        {"Objetivo": "OB3 · Reducción COP ≥ 5%", "Resultado": f"{reduccion_pct:.1f}%",
         "Estado": "✅ Cumple" if reduccion_pct>=5 else "❌ No cumple",
         "Margen": f"{reduccion_pct/5:.1f}× el umbral"},
        {"Objetivo": "Gap AUC < 0.10",      "Resultado": f"{gap_auc:.4f}",
         "Estado": "✅ Cumple" if gap_auc<0.10 else "⚠ Zona alerta",
         "Margen": f"{'Generaliza bien' if gap_auc<0.10 else 'Moderado'}"},
    ])
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.markdown("#### Score riesgo por canal (mapa aprendido desde train)")
    canal_tasa = train_df.groupby("Canal")[TARGET].mean().reset_index()
    canal_tasa.columns = ["Canal","tasa"]
    canal_tasa["score"] = canal_tasa["Canal"].map(canal_map).fillna(2).astype(int)
    canal_tasa["score_label"] = canal_tasa["score"].map(
        {1:"🟢 Bajo (1)", 2:"🟡 Medio (2)", 3:"🔴 Alto (3)"})
    canal_tasa["tasa_%"] = (canal_tasa["tasa"]*100).round(1)
    st.dataframe(
        canal_tasa[["Canal","tasa_%","score_label"]].sort_values("tasa_%",ascending=False),
        use_container_width=True, hide_index=True,
        column_config={"tasa_%": "Tasa problemática (%)", "score_label": "Score riesgo canal"}
    )
