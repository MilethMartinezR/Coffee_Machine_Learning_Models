# ☕ Coffee Shop Revenue Prediction

Proyecto de **Machine Learning** para la predicción de ingresos de cafeterías utilizando diferentes algoritmos de aprendizaje supervisado. El proyecto implementa un pipeline completo de procesamiento de datos, entrenamiento, evaluación y almacenamiento de modelos, permitiendo comparar el desempeño de múltiples técnicas de regresión y clasificación.

---

# 🚀 Características

- Preprocesamiento automático del dataset.
- Entrenamiento de múltiples modelos de Machine Learning.
- Comparación de métricas de desempeño.
- Generación automática de gráficas.
- Exportación de métricas a CSV.
- Almacenamiento de modelos entrenados (.pkl).
- Pipeline reproducible con una sola ejecución.

---

# 🛠 Tecnologías

- Python 3.11 / 3.12
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

# 📂 Estructura del proyecto

```
coffee_ml_project/
│
├── data/
│   └── coffee_shop_revenue.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── utils.py
│
├── models/
│   └── *.pkl
│
├── results/
│   ├── metrics.csv
│   └── graphs/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 📋 Descripción de los módulos

| Archivo | Descripción |
|----------|-------------|
| `data_preprocessing.py` | Limpieza, transformación y preparación del dataset. |
| `train_models.py` | Entrena todos los modelos de Machine Learning. |
| `evaluate_models.py` | Calcula las métricas de evaluación y genera los resultados. |
| `utils.py` | Funciones auxiliares utilizadas por el proyecto. |
| `main.py` | Punto de entrada del pipeline completo. |

---

# ⚙️ Instalación

Se recomienda utilizar **Python 3.11** o **Python 3.12**.

## 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd coffee_ml_project
```

---

## 2. Crear un entorno virtual

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 📥 Dataset

El proyecto utiliza el dataset **Coffee Shop Revenue** obtenido desde Kaggle.

Una vez descargado, ubique el archivo dentro de:

```
data/
└── coffee_shop_revenue.csv
```

---

# ▶️ Ejecución

Ejecute el proyecto desde la raíz:

```bash
python main.py
```

El pipeline realizará automáticamente:

1. Carga del dataset.
2. Preprocesamiento.
3. División de entrenamiento y prueba.
4. Entrenamiento de modelos.
5. Evaluación.
6. Guardado de modelos.
7. Generación de métricas.
8. Generación de gráficas.

---

# 🤖 Modelos implementados

| Modelo | Tipo |
|---------|------|
| Logistic Regression | Clasificación |
| Support Vector Machine (SVR) | Regresión |
| Decision Tree Regressor | Regresión |
| Random Forest Regressor | Regresión |
| Multi-Layer Perceptron (MLP Regressor) | Regresión |

---

# 📈 Métricas de evaluación

## Clasificación

- Accuracy
- Precision
- Recall
- F1 Score

## Regresión

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Todas las métricas son exportadas automáticamente a:

```
results/
└── metrics.csv
```

---

# 📊 Resultados

Al finalizar la ejecución se generan:

```
results/

├── metrics.csv
└── graphs/
    ├── comparison.png
    ├── errors.png
    └── ...
```

Los modelos entrenados se almacenan en:

```
models/
├── random_forest.pkl
├── svm.pkl
├── decision_tree.pkl
├── mlp.pkl
└── logistic.pkl
```

---

# 🧠 Consideraciones sobre la Regresión Logística

La **Regresión Logística** es un algoritmo de clasificación y no puede predecir directamente valores continuos como los ingresos de una cafetería.

Para incluir este modelo en la comparación, los ingresos fueron transformados en una variable binaria utilizando la mediana del conjunto de datos como umbral:

- **0:** ingresos bajos
- **1:** ingresos altos

De esta manera, la Regresión Logística permite clasificar las cafeterías según su nivel de ingresos, mientras que los demás modelos estiman el valor numérico exacto.

---

# 💻 Compatibilidad

| Sistema Operativo | Soporte |
|-------------------|---------|
| Windows | ✅ |
| macOS | ✅ |
| Linux | ✅ |
| iOS / iPadOS | ❌ No soportado |

---

# 📦 Dependencias principales

- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

Todas las dependencias se encuentran en:

```
requirements.txt
```

---

# 🎯 Objetivos del proyecto

- Comparar diferentes algoritmos de Machine Learning.
- Evaluar modelos de clasificación y regresión.
- Analizar el rendimiento mediante métricas estándar.
- Automatizar el pipeline de entrenamiento y evaluación.
- Facilitar la reproducibilidad de los experimentos.

---

# 📄 Licencia

Este proyecto fue desarrollado con fines académicos y de aprendizaje en Machine Learning y Ciencia de Datos.
