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
```bash
pip install -r requirements.txt
```

## Uso
1. Descarga el CSV de Kaggle y colócalo en `data/`
2. Ejecuta:
```bash
python main.py
```

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