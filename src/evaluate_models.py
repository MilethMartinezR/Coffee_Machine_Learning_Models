import joblib
import os


# ===============================
# 🔹 SELECCIONAR Y GUARDAR MEJOR MODELO
# ===============================
def save_best_model(models, df_regression):
    """
    Selecciona el mejor modelo de regresión basado en RMSE
    y lo guarda en la carpeta models/
    """

    # Crear carpeta si no existe
    os.makedirs("models", exist_ok=True)

    # Ordenar por menor RMSE
    best_row = df_regression.loc[df_regression["RMSE"].idxmin()]
    best_model_name = best_row["Model"]

    # Obtener el modelo correspondiente
    best_model = models[best_model_name]

    # Guardar modelo
    path = f"models/best_model_{best_model_name.replace(' ', '_')}.pkl"
    joblib.dump(best_model, path)

    print(f"\n🏆 Mejor modelo: {best_model_name}")
    print(f"💾 Guardado en: {path}")