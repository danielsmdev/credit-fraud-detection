"""
Script para ejecutar el preprocesamiento de datos.
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path de Python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import preprocessing_pipeline
from src.config import RAW_DATA_FILE

def main():
    """Función principal para ejecutar el preprocesamiento de datos."""
    print("Iniciando preprocesamiento de datos...")
    
    # Verificar que el archivo de datos existe
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Error: No se encontró el archivo {RAW_DATA_FILE}")
        print("Por favor, descarga el dataset desde Kaggle y colócalo en la carpeta 'data/raw/'")
        return
    
    # Ejecutar pipeline de preprocesamiento
    X_train, X_test, y_train, y_test, feature_names = preprocessing_pipeline(RAW_DATA_FILE)
    
    print("Preprocesamiento completado exitosamente.")
    print(f"Características de entrenamiento: {X_train.shape}")
    print(f"Características de prueba: {X_test.shape}")
    print(f"Etiquetas de entrenamiento: {y_train.shape}")
    print(f"Etiquetas de prueba: {y_test.shape}")
    print(f"Nombres de características: {len(feature_names)}")

if __name__ == "__main__":
    main()
