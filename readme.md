# Proyecto ML — Predicción de Ingresos de Cafeterías

## Estructura
```
coffee_ml_project/
├── data/
│   └── coffee_shop_revenue.csv   ← coloca aquí el CSV de Kaggle
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── utils.py
├── results/
│   ├── metrics.csv               ← generado al ejecutar
│   └── graphs/                   ← imágenes generadas
├── models/                       ← modelos .pkl guardados
├── main.py
└── requirements.txt
```

## Instalación
Recomendado: **Python 3.11 o 3.12** en Windows o macOS.

```bash
pip install -r requirements.txt
```

## Quickstart (Windows y macOS)
Desde la raíz del proyecto, crea y activa un entorno virtual:

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS (Terminal):

```bash
source .venv/bin/activate
```

Luego instala dependencias y ejecuta el pipeline:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Uso
1. Descarga el CSV de Kaggle y colócalo en `data/`
2. Ejecuta:
```bash
python main.py
```

## Compatibilidad
- Soportado: Windows y macOS.
- No objetivo de ejecución: iOS/iPadOS (el stack de `scikit-learn` no está orientado a ese entorno).

## Modelos
| Modelo               | Tipo           | Métrica principal |
|----------------------|----------------|-------------------|
| Regresión Logística  | Clasificación  | Accuracy, F1      |
| SVM                  | Regresión      | MAE, RMSE, R²     |
| Árbol de Decisión    | Regresión      | MAE, RMSE, R²     |
| Random Forest        | Regresión      | MAE, RMSE, R²     |
| Red Neuronal (MLP)   | Regresión      | MAE, RMSE, R²     |

## Nota sobre Regresión Logística
Como los ingresos son un valor continuo, la Regresión Logística
se aplica sobre una versión binaria (alto/bajo según la mediana).
Los demás modelos predicen el valor exacto de ingresos.