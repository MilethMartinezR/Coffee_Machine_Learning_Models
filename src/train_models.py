from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np


# ===============================
# 🔹 3. INGENIERÍA DE MODELOS
# Transformación para clasificación
# ===============================
def create_classes(y, threshold=None):
    """
    Convierte la variable continua (ingresos) en clases:
    0 = ingresos bajos
    1 = ingresos altos

    Si no se define threshold, usa la media.
    """
    if threshold is None:
        threshold = np.mean(y)

    y_class = (y > threshold).astype(int)
    return y_class, threshold


# ===============================
# 🔹 3. INGENIERÍA DE MODELOS
# Regresión Logística (Clasificación)
# ===============================
def train_logistic_regression(X_train, y_train):
    """
    Modelo de clasificación basado en regresión logística.
    Se aplica sobre datos transformados en clases.
    """
    y_train_class, threshold = create_classes(y_train)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train_class)

    return model, threshold


# ===============================
# 🔹 3. INGENIERÍA DE MODELOS
# SVM (Regresión)
# ===============================
def train_svm(X_train, y_train):
    """
    SVM para regresión (SVR)
    """
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    return model


# ===============================
# 🔹 3. INGENIERÍA DE MODELOS
# Árbol de decisión
# ===============================
def train_decision_tree(X_train, y_train):
    """
    Árbol de decisión para regresión
    """
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)
    return model


# ===============================
# 🔹 3. INGENIERÍA DE MODELOS
# Random Forest
# ===============================
def train_random_forest(X_train, y_train):
    """
    Random Forest (ensamble de árboles)
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# ===============================
# 🔹 3. INGENIERÍA DE MODELOS
# Red neuronal
# ===============================
def train_neural_network(X_train, y_train):
    """
    Red neuronal (MLP)
    """
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model


# ===============================
# 🔹 ENTRENAR TODOS LOS MODELOS
# ===============================
def train_all_models(X_train, y_train):
    """
    Entrena todos los modelos separando:
    - Regresión
    - Clasificación (logística)
    """

    models = {}

    # 🔹 Modelos de regresión
    models["SVM"] = train_svm(X_train, y_train)
    models["Decision Tree"] = train_decision_tree(X_train, y_train)
    models["Random Forest"] = train_random_forest(X_train, y_train)
    models["Neural Network"] = train_neural_network(X_train, y_train)

    # 🔹 Modelo de clasificación (separado)
    logistic_model, threshold = train_logistic_regression(X_train, y_train)
    models["Logistic Regression"] = (logistic_model, threshold)

    return models