"""
train_models.py
===============
Ciclo de vida ML — etapas cubiertas en este módulo:

  ETAPA 5 · Desarrollo de un punto de referencia del modelo.
    - Se definen los hiperparámetros iniciales de cada modelo.
    - El baseline más simple es LogisticRegression (clasificación binaria).

  ETAPA 6 · Desarrollo de un buen modelo y ajuste fino de sus parámetros.
    - Se entrenan los 5 algoritmos requeridos por el parcial:
        · Regresión Logística  (clasificación: ingresos alto/bajo)
        · SVM — SVR con kernel RBF
        · Árbol de Decisión    (max_depth=8)
        · Random Forest        (200 árboles)
        · Red Neuronal MLP     (capas 128→64→32, early stopping)
    - Cada modelo se guarda en models/ con joblib para reutilización.

  Nota sobre Regresión Logística:
    Los ingresos son un valor continuo → se binariza según la mediana
    (>= mediana = "Alto", < mediana = "Bajo") para poder usar el clasificador.
    El resto de modelos predicen el valor exacto de ingresos (regresión).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVR
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from utils import MODELS_DIR, print_section


# ── ETAPA 5 · Desarrollo de un punto de referencia del modelo ──────────────
def build_models() -> dict:
    """
    Devuelve un diccionario {nombre: estimador} con los 5 algoritmos del parcial.
    Los hiperparámetros son valores iniciales razonables (baseline).
    En ETAPA 5 se pueden ajustar con GridSearchCV si se desea afinar.
    """
    models = {
        # Clasificador binario (ingresos alto/bajo) — sirve de baseline simple
        "Regresión Logística": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        # Regresor con kernel RBF — bueno para relaciones no lineales
        "SVM": SVR(
            kernel="rbf", C=10, epsilon=0.1
        ),
        # Regresor de árbol único — interpretable, propenso a overfitting profundo
        "Árbol de Decisión": DecisionTreeRegressor(
            max_depth=8, random_state=42
        ),
        # Ensemble de 200 árboles — reduce varianza respecto al árbol solo
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        # Red neuronal multicapa con early stopping para evitar overfitting
        "Red Neuronal": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }
    return models


# ── ETAPA 6 · Desarrollo de un buen modelo y ajuste fino de sus parámetros ─
def train_all(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val:   np.ndarray,
    y_val:   pd.Series,
) -> dict:
    """
    Entrena todos los modelos sobre X_train/y_train.
    Evalúa en X_val/y_val para un primer chequeo de rendimiento.
    Guarda cada modelo en models/ con joblib para no reentrenar cada vez.
    """
    print_section("ETAPA 6 · Desarrollo de un buen modelo y ajuste fino de sus parámetros")

    models = build_models()
    trained = {}

    # Umbral para binarizar y → usado solo por Regresión Logística
    threshold = y_train.median()
    y_train_bin = (y_train >= threshold).astype(int)
    y_val_bin   = (y_val   >= threshold).astype(int)

    for name, model in models.items():
        print(f"\n  Entrenando: {name} ...", end=" ")

        if name == "Regresión Logística":
            # ETAPA 5 — fit sobre etiquetas binarias (alto=1 / bajo=0)
            model.fit(X_train, y_train_bin)
            val_score = model.score(X_val, y_val_bin)
            print(f"Accuracy val = {val_score:.4f}")
        else:
            # ETAPA 5 — fit sobre el valor continuo de ingresos
            model.fit(X_train, y_train)
            # R² en validación: primer indicador de si el modelo generaliza
            val_score = model.score(X_val, y_val)
            print(f"R² val = {val_score:.4f}")

        # Persistir modelo entrenado en disco (reutilizable sin reentrenar)
        model_path = f"{MODELS_DIR}/{name.replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        trained[name] = model

    print("\n  Todos los modelos entrenados y guardados.")
    return trained, threshold