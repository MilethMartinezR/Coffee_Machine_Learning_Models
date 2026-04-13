"""
evaluate_models.py
==================
Ciclo de vida ML — este módulo aplica la evaluación final posterior a la etapa 6:

  Evaluación de los modelos afinados:
    - Se calculan las métricas definidas en la etapa 2:
        · Modelos de regresión : MAE, RMSE, R²
        · Regresión Logística  : Accuracy, F1-Score
    - Se aplica el protocolo de evaluación de la etapa 3 sobre el conjunto de TEST
      (datos que los modelos nunca vieron durante el entrenamiento).
    - Se genera la tabla comparativa metrics.csv para el informe.

  Visualizaciones generadas:
    · Real vs Predicho por modelo (regresión)
    · Matriz de confusión (Regresión Logística)
    · Gráfico comparativo de R² y RMSE entre modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

from utils import RESULTS_DIR, save_figure, print_section, MODEL_NAMES, COLORS


# ── Evaluación final posterior a la etapa 6 — conjunto de test ───────────
def evaluate_all(
    trained_models: dict,
    X_test:  np.ndarray,
    y_test:  pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """
    Evalúa cada modelo sobre X_test (conjunto nunca visto en entrenamiento).
    Calcula las métricas de éxito definidas en la ETAPA 2 del ciclo ML.
    Exporta la tabla comparativa a results/metrics.csv.
    """
    print_section("Evaluación final posterior a la etapa 6")

    # Binarizar target para Regresión Logística (mismo umbral del entrenamiento)
    y_test_bin = (y_test >= threshold).astype(int)
    rows = []

    for name, model in trained_models.items():
        if name == "Regresión Logística":
            # ETAPA 6 — Métricas de clasificación definidas en etapa 2: Accuracy y F1
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test_bin, y_pred)
            f1  = f1_score(y_test_bin, y_pred, zero_division=0)
            row = {
                "Modelo": name,
                "Tipo": "Clasificación",
                "Accuracy": round(acc, 4),
                "F1-Score": round(f1, 4),
                "MAE": "-",
                "RMSE": "-",
                "R²": "-",
            }
            print(f"  {name:<25}  Acc={acc:.4f}  F1={f1:.4f}")
            # ETAPA 6 — Visualización: matriz de confusión
            _plot_confusion_matrix(name, model, X_test, y_test_bin)
        else:
            # ETAPA 6 — Métricas de regresión definidas en etapa 2: MAE, RMSE, R²
            y_pred = model.predict(X_test)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)
            row = {
                "Modelo": name,
                "Tipo": "Regresión",
                "Accuracy": "-",
                "F1-Score": "-",
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R²": round(r2, 4),
            }
            print(f"  {name:<25}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
            # ETAPA 6 — Visualización: real vs predicho
            _plot_pred_vs_real(name, y_test, y_pred)

        rows.append(row)

    # ETAPA 6 — Exportar tabla comparativa para el informe del parcial
    df_metrics = pd.DataFrame(rows)
    metrics_path = f"{RESULTS_DIR}/metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\n  Métricas guardadas en: {metrics_path}")

    # Gráfico comparativo entre todos los modelos de regresión
    _plot_metrics_comparison(df_metrics)
    return df_metrics


# ── Gráficas ──────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(name: str, model, X_test, y_test_bin) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test_bin,
        display_labels=["Bajo", "Alto"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Matriz de confusión — {name}")
    safe = name.replace(" ", "_")
    save_figure(fig, f"cm_{safe}.png")


def _plot_pred_vs_real(name: str, y_real, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_real, y_pred, alpha=0.4, s=20, color="#4C72B0")
    lims = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predicho")
    ax.set_title(f"Real vs Predicho — {name}")
    ax.legend()
    safe = name.replace(" ", "_")
    save_figure(fig, f"pred_{safe}.png")


def _plot_metrics_comparison(df_metrics: pd.DataFrame) -> None:
    """Gráfica de barras comparando R² y RMSE de los modelos de regresión."""
    reg = df_metrics[df_metrics["Tipo"] == "Regresión"].copy()
    if reg.empty:
        return

    reg["R²"]   = pd.to_numeric(reg["R²"],   errors="coerce")
    reg["RMSE"] = pd.to_numeric(reg["RMSE"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Comparación de modelos de regresión", fontsize=13)

    sns.barplot(data=reg, x="Modelo", y="R²", ax=axes[0], palette="muted")
    axes[0].set_title("R² (mayor es mejor)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha="right")
    axes[0].set_ylim(0, 1)

    sns.barplot(data=reg, x="Modelo", y="RMSE", ax=axes[1], palette="muted")
    axes[1].set_title("RMSE (menor es mejor)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")

    save_figure(fig, "comparacion_modelos.png")