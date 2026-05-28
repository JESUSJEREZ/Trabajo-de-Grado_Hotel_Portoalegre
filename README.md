# Modelo M1 — Scoring de Cancelaciones Problemáticas
### Hotel Portoalegre · Golfo de Morrosquillo · Coveñas, Sucre

> **Trabajo de Grado Aplicado**  
> Maestría en Analítica para Inteligencia de Negocios  
> Pontificia Universidad Javeriana · Bogotá · Primer Semestre 2026

**Equipo:** Johan Esteban Higuera Hurtado · Juan Camilo Ibarra Cifuentes · Daniel Balen Giancola · Jesús Eduardo Jerez Rojas  
**Tutor:** Cristian Camilo Tirado Cifuentes

---

## ¿Qué hace esta app?

Convierte el notebook de modelado (`Modelo_M1_Reservas_Canceladas.ipynb`) en una
herramienta operativa que el Hotel Portoalegre puede usar diariamente.

Carga el archivo de reservas canceladas exportado desde el PMS **Lobbybookings**,
entrena el modelo Random Forest con SMOTE exactamente como se describe en el
documento CRISP-DM, y produce:

- Score de probabilidad de cancelación problemática para cada reserva
- Clasificación en riesgo **Alto / Medio / Bajo** con umbrales dinámicos por temporada
- Evaluación técnica: AUC, Recall, Gap AUC, matriz de confusión, curva ROC
- Simulación contrafactual del impacto financiero (Fórmula F1 del documento)
- Proyección de oportunidad potencial al año configurado (ajuste inflacionario 2.5%)

---

## Estructura del repositorio

```
Trabajo-de-Grado_Hotel_Portoalegre/
│
├── app_m1_portoalegre.py          ← App Streamlit (este archivo)
├── requirements.txt               ← Dependencias Python
├── README.md                      ← Este archivo
│
├── Modelo_M1_Reservas_Canceladas.ipynb   ← Notebook fuente
├── Pronostico_Ocupacion_v12.ipynb        ← Notebook M2/M4
├── Modelo_Segmentación_de_Huéspedes.ipynb ← Notebook M3
│
└── data/
    └── reservas_canceladas_2026-03-18.xls  ← Datos fuente (no versionados)
```

> **Nota:** El archivo de datos no se versiona en el repositorio por protección
> de datos personales (Ley 1581 de 2012). Se carga localmente desde la app.

---

## Instalación rápida

### 1. Clonar el repositorio

```bash
git clone https://github.com/JESUSJEREZ/Trabajo-de-Grado_Hotel_Portoalegre.git
cd Trabajo-de-Grado_Hotel_Portoalegre
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Correr la app

```bash
streamlit run app_m1_portoalegre.py
```

La app abre automáticamente en `http://localhost:8501`.

---

## Uso paso a paso

| Paso | Acción |
|------|--------|
| 1 | Exportar desde Lobbybookings: **Reservas → Canceladas → Exportar a Excel** |
| 2 | Subir el archivo `.xls` en el panel izquierdo de la app |
| 3 | Seleccionar modo de entrenamiento (Rápido para operación diaria) |
| 4 | Explorar las 5 pestañas de resultados |
| 5 | Descargar el scoring completo en CSV desde la pestaña 🎯 |

---

## Umbrales de intervención por temporada

El modelo asigna un score de 0 a 1 a cada reserva.
El umbral que define cuándo se activa una alerta cambia según la temporada:

| Temporada | Meses | Umbral | Justificación |
|-----------|-------|--------|---------------|
| **Alta** | Jun · Jul · Ago · Dic · Ene | **0.25** | COP perdido por evento ~42% mayor — máxima sensibilidad |
| **Precursor** | Nov · May | **0.30** | Comportamiento similar a Alta (Hallazgo H2) |
| **Media / Baja** | Resto | **0.45** | Menor impacto financiero — evita saturar recepción |

> Mismo modelo, distinta sensibilidad. El umbral no reentrena el modelo,
> solo mueve la línea de corte sobre el mismo score.

---

## Métricas del modelo final (Random Forest)

| Métrica | Resultado | Umbral de éxito | Estado |
|---------|-----------|-----------------|--------|
| AUC-ROC | 0.9170 | ≥ 0.75 | ✅ Cumple |
| Recall | 0.7719 | ≥ 0.70 | ✅ Cumple |
| Precisión | 0.95 | — | 19 FP / 1.068 reservas |
| Gap AUC | 0.0651 | < 0.10 | ✅ Generaliza bien |
| Reducción COP | 33.3% | ≥ 5% | ✅ 6.6× el umbral |

---

## Decisiones metodológicas clave

**Anti data leakage:**
- Split temporal estricto (70% train / 30% test) ordenado por `Fecha creación`
- `cancelaciones_previas_huesped` calculada con `cumcount()` anticausal (no `transform('count')`)
- `score_riesgo_canal` calculado exclusivamente desde train post-split
- SMOTE aplicado solo dentro del fold de entrenamiento (ImbPipeline)

**Selección del modelo:**
Random Forest fue seleccionado sobre XGBoost y LightGBM por su menor Gap AUC
(0.0651 vs 0.0690 y 0.0800). Los tres modelos cumplen los criterios técnicos,
pero RF generaliza mejor a datos nuevos — propiedad estructural del ajuste,
no ajustable por umbral.

**Hallazgos independientes del modelo:**
- **H1:** Booking Engine concentra el 89.7% de tasa problemática por un
  problema técnico de pasarela de cobro, no por riesgo del huésped.
  Acción recomendada: retry policy / pre-autorización, independiente del modelo.
- **H3:** Temporada Alta tiene menor tasa de cancelación pero mayor COP por
  evento (~$2.3M vs $1.6M). La priorización debe guiarse por impacto
  financiero, no por frecuencia.

---

## Arquitectura de despliegue recomendada

```
Lobbybookings PMS
      │  (export .xlsx diario)
      ▼
app_m1_portoalegre.py   ←── modelo .pkl (serializado con joblib)
      │
      ├── Score + categoría por reserva
      ├── Dashboard operativo (Streamlit)
      └── Reporte gerencial (Power BI / exportar CSV)
```

**Siguiente paso:** cuando Lobbybookings exponga una API REST, el job diario
puede automatizarse con Airflow o cron, eliminando la carga manual del archivo.

---

## Plan de monitoreo

| Modelo | Frecuencia reentrenamiento | Alerta de drift |
|--------|--------------------------|-----------------|
| M1 | Trimestral | Caída > 5 pp en Recall |
| M2 | Mensual | MAPE rolling 30d > 18% |
| M3 | Semestral | Desplazamiento centroides > 15% |

Cada reentrenamiento debe documentarse en el repositorio: fecha, dataset,
hiperparámetros, métricas de validación y autor responsable.

---

## Marco normativo

Tratamiento de datos bajo **Ley 1581 de 2012** y **Decreto 1377 de 2013**.

- Identificadores directos (nombre, documento, email) excluidos del entrenamiento
- Campo `huésped` usado solo de forma agregada y anonimizada
- Nacionalidad omitida deliberadamente en M1 para evitar perfilamiento por origen
- Datos brutos no versionados en repositorio público

---

## Dependencias principales

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
plotly>=5.20.0
openpyxl>=3.1.0
xlrd>=2.0.1
```

---

## Solución a problemas comunes

**`ModuleNotFoundError: No module named 'streamlit'`**  
El entorno virtual no está activo.  
Windows: `venv\Scripts\activate` · Mac/Linux: `source venv/bin/activate`

**`xlrd.biffh.XLRDError`**  
El archivo `.xls` está en formato HTML (exportación por defecto de Lobbybookings).
La app lo maneja automáticamente. Si persiste, ábrelo en Excel y guarda como `.xlsx`.

**Puerto 8501 ocupado**  
```bash
streamlit run app_m1_portoalegre.py --server.port 8502
```

**El modelo tarda mucho**  
Cambia a modo **Rápido** en el sidebar. Mismos hiperparámetros finales del notebook,
sin GridSearchCV.

---

*Estimación de potencial — no ingreso garantizado.  
La recuperación efectiva depende de la capacidad operativa del hotel para
reasignar habitaciones alertadas antes del check-in.*
