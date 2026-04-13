"""
main.py — Punto de entrada del proyecto.
Ejecuta el ciclo completo de ML:
  1. Carga y análisis exploratorio
  2. Preprocesamiento
  3. Entrenamiento de 5 modelos
  4. Evaluación y generación de reportes

Uso:
    python main.py
"""

import sys
import os

# Añadir src/ al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_data, exploratory_analysis, preprocess
from train_models        import train_all
from evaluate_models     import evaluate_all
from utils               import print_section


def main():
    print("\n" + "█" * 60)
    print("  PROYECTO ML — PREDICCIÓN DE INGRESOS DE CAFETERÍAS")
    print("█" * 60)

    # ── ETAPA 1 · Recopilación de datos ───────────────────────────
    # Carga el CSV y muestra estructura general del dataset
    df = load_data()

    # ── ETAPA 2 · Elección de la medida de éxito ──────────────────
    # EDA para entender la distribución del target y justificar
    # el uso de MAE, RMSE y R² como métricas de evaluación
    exploratory_analysis(df)

    # ── ETAPAS 3-4 · Protocolo de evaluación + Preparación ────────
    # Limpieza, codificación, normalización y división 70/15/15
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = preprocess(df)

    # ── ETAPAS 5-6 · Baseline y entrenamiento ─────────────────────
    # Entrena los 5 algoritmos y los guarda en models/
    trained_models, threshold = train_all(X_train, y_train, X_val, y_val)

    # ── ETAPAS 2-3 · Evaluación final ─────────────────────────────
    # Calcula métricas sobre test, genera gráficas y exporta metrics.csv
    metrics_df = evaluate_all(trained_models, X_test, y_test, threshold)

    # ── Resumen final ─────────────────────────────────────────────
    print_section("Resumen de métricas")
    print(metrics_df.to_string(index=False))

    print("\n  Proyecto completado. Revisa results/ para gráficas y métricas.\n")


if __name__ == "__main__":
    main()