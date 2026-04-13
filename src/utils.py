"""
utils.py — Funciones de apoyo compartidas entre módulos.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ── Directorios ───────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT_DIR, "data", "coffee_shop_revenue.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
GRAPHS_DIR  = os.path.join(RESULTS_DIR, "graphs")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")

for _dir in [RESULTS_DIR, GRAPHS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Estilo global de gráficas ─────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = sns.color_palette("muted", 5)

MODEL_NAMES = [
    "Regresión Logística",
    "SVM",
    "Árbol de Decisión",
    "Random Forest",
    "Red Neuronal",
]


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Guarda una figura en results/graphs/."""
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figura guardada: {path}")


def print_section(title: str) -> None:
    """Imprime un encabezado de sección en consola."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)