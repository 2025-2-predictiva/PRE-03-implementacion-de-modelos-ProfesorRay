"""Ejemplo de uso del modelo de regresión lineal."""

from homework import LinearRegression, load_data, train_and_evaluate_model
import numpy as np


def main():
    """Función principal que demuestra el uso del modelo."""
    print("=== Ejemplo de Regresión Lineal ===\n")
    
    # Ejemplo 1: Usar la función principal
    print("1. Ejecutando función principal:")
    model, X, y, y_pred = train_and_evaluate_model()
    
    print("\n" + "="*50 + "\n")
    
    # Ejemplo 2: Uso manual paso a paso
    print("2. Ejemplo de uso manual:")
    
    # Cargar datos
    X, y = load_data()
    print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
    
    # Crear modelo
    model = LinearRegression()
    print("Modelo creado")
    
    # Entrenar modelo
    model.fit(X, y)
    print("Modelo entrenado")
    
    # Hacer predicción para una nueva muestra
    new_sample = np.array([[0.5, 0.3]])  # x1=0.5, x2=0.3
    prediction = model.predict(new_sample)
    print(f"Predicción para x1=0.5, x2=0.3: {prediction[0]:.4f}")
    
    # Calcular score
    score = model.score(X, y)
    print(f"R² Score: {score:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Ejemplo 3: Comparación con datos reales
    print("3. Comparación de predicciones vs valores reales:")
    print("Índice | Real    | Predicho | Error")
    print("-" * 35)
    for i in range(min(10, len(y))):
        real = y[i]
        pred = y_pred[i]
        error = abs(real - pred)
        print(f"{i:6d} | {real:7.4f} | {pred:8.4f} | {error:.4f}")


if __name__ == "__main__":
    main()
