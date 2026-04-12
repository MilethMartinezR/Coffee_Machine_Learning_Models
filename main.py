# ===============================
# 🔹 IMPORTACIONES
# ===============================
from src.data_preprocessing import preprocess_pipeline
from src.train_models import train_all_models
from src.evaluate_models import evaluate_all_models, save_results, print_results


# ===============================
# 🔹 CONFIGURACIÓN
# ===============================
DATA_PATH = "data/coffee_shop_revenue.csv"

# ⚠️ Cambia esto según el dataset real
TARGET_COLUMN = "Revenue"


# ===============================
# 🔹 FUNCIÓN PRINCIPAL
# ===============================
def main():

    # ===============================
    # 🔹 1. PLANIFICACIÓN Y PREPARACIÓN DE DATOS
    # ===============================
    print("📊 Cargando y preprocesando datos...")

    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        DATA_PATH,
        TARGET_COLUMN
    )

    print("✅ Datos listos")

    # ===============================
    # 🔹 2. INGENIERÍA DE MODELOS
    # ===============================
    print("\n🤖 Entrenando modelos...")

    models = train_all_models(X_train, y_train)

    print("✅ Modelos entrenados")

    # ===============================
    # 🔹 3. EVALUACIÓN DEL MODELO
    # ===============================
    print("\n📊 Evaluando modelos...")

    df_reg, df_clf = evaluate_all_models(models, X_test, y_test)

    print("✅ Evaluación completada")

    # ===============================
    # 🔹 4. RESULTADOS
    # ===============================
    print_results(df_reg, df_clf)
    save_results(df_reg, df_clf)

    print("\n🎯 Proceso finalizado correctamente")


# ===============================
# 🔹 EJECUCIÓN
# ===============================
if __name__ == "__main__":
    main()