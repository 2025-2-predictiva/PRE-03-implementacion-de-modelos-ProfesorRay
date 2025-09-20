"""Implementación de regresión lineal desde cero."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class LinearRegression:
    """Implementación de regresión lineal desde cero."""
    
    def __init__(self):
        """Inicializar el modelo de regresión lineal."""
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.is_fitted: bool = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Entrenar el modelo de regresión lineal.
        
        Args:
            X: Matriz de características (n_samples, n_features)
            y: Vector objetivo (n_samples,)
            
        Returns:
            self: Instancia del modelo entrenado
        """
        # Agregar columna de unos para el intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calcular coeficientes usando la ecuación normal: (X^T * X)^-1 * X^T * y
        XtX = np.dot(X_with_intercept.T, X_with_intercept)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(X_with_intercept.T, y)
        
        # Obtener coeficientes
        coefficients = np.dot(XtX_inv, Xty)
        self.intercept = coefficients[0]
        self.coefficients = coefficients[1:]
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hacer predicciones usando el modelo entrenado.
        
        Args:
            X: Matriz de características (n_samples, n_features)
            
        Returns:
            Predicciones (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        return np.dot(X, self.coefficients) + self.intercept
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcular el coeficiente de determinación R².
        
        Args:
            X: Matriz de características (n_samples, n_features)
            y: Vector objetivo (n_samples,)
            
        Returns:
            Coeficiente R²
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de calcular el score")
        
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """
        Obtener parámetros del modelo.
        
        Returns:
            Diccionario con los parámetros del modelo
        """
        if not self.is_fitted:
            return {"intercept": None, "coefficients": None}
        
        return {
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist()
        }
    
    def set_params(self, intercept: float, coefficients: np.ndarray) -> None:
        """
        Establecer parámetros del modelo.
        
        Args:
            intercept: Valor del intercepto
            coefficients: Array de coeficientes
        """
        self.intercept = intercept
        self.coefficients = np.array(coefficients)
        self.is_fitted = True


def load_data(file_path: str = "files/input/data.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Cargar datos desde un archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        Tupla con (X, y) donde X son las características e y es el objetivo
    """
    df = pd.read_csv(file_path)
    X = df[['x1', 'x2']].values
    y = df['y'].values
    return X, y


def train_model_with_data():
    """
    Función para entrenar el modelo con los datos del archivo CSV.
    
    Returns:
        Tupla con (modelo, X, y, predicciones)
    """
    # Cargar datos
    X, y = load_data()
    
    # Crear y entrenar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Hacer predicciones
    y_pred = model.predict(X)
    
    return model, X, y, y_pred


if __name__ == "__main__":
    # Ejecutar entrenamiento y mostrar resultados
    model, X, y, y_pred = train_model_with_data()
    
    # Calcular métricas
    r2_score = model.score(X, y)
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    print("=== Resultados del Modelo de Regresión Lineal ===")
    print(f"Intercepto: {model.intercept:.4f}")
    print(f"Coeficientes: {model.coefficients}")
    print(f"R² Score: {r2_score:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Mostrar algunos ejemplos
    print("\n=== Ejemplos de Predicciones ===")
    for i in range(min(5, len(y))):
        print(f"Muestra {i+1}: x1={X[i,0]:.3f}, x2={X[i,1]:.3f}")
        print(f"  Real: {y[i]:.4f}, Predicho: {y_pred[i]:.4f}")
        print(f"  Error: {abs(y[i] - y_pred[i]):.4f}\n")
