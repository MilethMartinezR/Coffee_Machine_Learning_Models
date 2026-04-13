"""
data_preprocessing.py
======================
Ciclo de vida ML — etapas cubiertas en este módulo:

  ETAPA 1 · Recopilación de datos
    - Carga el CSV y muestra su estructura (filas, columnas, tipos).

  ETAPA 2 · Elección de la medida de éxito
    - Genera el reporte EDA (ydata-profiling) para entender la distribución
      del target y decidir métricas apropiadas (MAE, RMSE, R²).

  ETAPA 3 · Establecimiento del protocolo de evaluación
    - Define la división train / validation / test (70 / 15 / 15).
    - El scaler se ajusta SOLO en train para evitar data leakage.

  ETAPA 4 · Preparación de los datos
    - Elimina duplicados.
    - Imputa valores faltantes con la mediana.
    - Codifica variables categóricas con one-hot encoding.
    - Normaliza con StandardScaler.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from ydata_profiling import ProfileReport
except Exception:
    ProfileReport = None

from utils import DATA_PATH, RESULTS_DIR, save_figure, print_section


# ── ETAPA 1 · Recopilación de datos ──────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print_section("ETAPA 1 · Recopilación de datos")
    df = pd.read_csv(DATA_PATH)
    print(f"  Filas   : {df.shape[0]}")
    print(f"  Columnas: {df.shape[1]}")
    print(f"\n{df.dtypes.to_string()}")
    return df


# ── ETAPA 2 · Elección de la medida de éxito (EDA) ───────────────────────────
def exploratory_analysis(df: pd.DataFrame) -> None:
    print_section("ETAPA 2 · Elección de la medida de éxito — EDA")

    # Estadísticas descriptivas: permite decidir si usar MAE, RMSE o R²
    print(df.describe().to_string())

    # Detectar valores faltantes que afectarían el entrenamiento
    missing = df.isnull().sum()
    print(f"\nValores faltantes:\n{missing[missing > 0]}")

    # Reporte automático completo — distribuciones, correlaciones, outliers
    # Esto justifica la elección de métricas de regresión para el parcial
    if ProfileReport is not None:
        print("\n  Generando reporte HTML con ydata-profiling ...")
        profile = ProfileReport(df, title="Coffee Shop — EDA Report", explorative=True)
        report_path = f"{RESULTS_DIR}/eda_report.html"
        profile.to_file(report_path)
        print(f"  [OK] Reporte guardado: {report_path}")
    else:
        print("\n  [WARN] ydata-profiling no disponible; se omite el reporte HTML.")

    # Visualizar distribución del target (ingresos)
    # Si es simétrica → RMSE es adecuado; si tiene sesgo → preferir MAE
    target_col = _get_target(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Distribución de ingresos diarios", fontsize=13)

    sns.histplot(df[target_col], kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Histograma")
    axes[0].set_xlabel(target_col)

    sns.boxplot(y=df[target_col], ax=axes[1], color="#4C72B0")
    axes[1].set_title("Boxplot")

    save_figure(fig, "01_distribucion_objetivo.png")

    # Matriz de correlación: identifica qué features influyen más en el target
    num_df = df.select_dtypes(include=np.number)
    fig2, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax
    )
    ax.set_title("Matriz de correlación")
    save_figure(fig2, "02_correlacion.png")


# ── ETAPA 3 y 4 · Protocolo de evaluación + Preparación de datos ─────────────
def preprocess(df: pd.DataFrame):
    """
    Devuelve:
        X_train, X_val, X_test  — arrays NumPy normalizados
        y_train, y_val, y_test  — Series de pandas
        scaler                  — StandardScaler ajustado al train
        feature_names           — lista de nombres de columnas
    """
    print_section("ETAPAS 3-4 · Protocolo de evaluación + Preparación de datos")

    df = df.copy()

    # ETAPA 4 — Eliminar duplicados que sesgarían el entrenamiento
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Duplicados eliminados: {before - len(df)}")

    # ETAPA 4 — Imputar nulos numéricos con la mediana
    # Se usa mediana (robusta a outliers) en lugar de la media
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # ETAPA 4 — Codificar variables categóricas con one-hot encoding
    # Los algoritmos de ML requieren entradas numéricas
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    target_col = _get_target(df)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"  Columnas codificadas: {cat_cols}")

    # Separar features / target
    target_col = _get_target(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    print(f"  Features : {len(feature_names)}")
    print(f"  Objetivo : {target_col}")

    # ETAPA 3 — División 70 / 15 / 15
    # Train: aprende patrones | Val: ajuste de hiperparámetros | Test: evaluación final
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ETAPA 4 — Normalización con StandardScaler
    # fit SOLO en train para evitar data leakage hacia val y test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names


# ── Auxiliar ──────────────────────────────────────────────────────────────────
def _get_target(df: pd.DataFrame) -> str:
    """
    Detecta automáticamente la columna objetivo.
    Ajusta 'revenue' al nombre real del CSV si fuera distinto.
    """
    candidates = [c for c in df.columns if "revenue" in c.lower()]
    if candidates:
        return candidates[0]
    # Fallback: última columna numérica
    return df.select_dtypes(include=np.number).columns[-1]