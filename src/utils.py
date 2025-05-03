"""
Utilidades generales para el proyecto de detección de fraude.
"""

import time
import os
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from src.config import FIGURES_DIR

def timer_decorator(func: Callable) -> Callable:
    """
    Decorador para medir el tiempo de ejecución de una función.
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Función {func.__name__} ejecutada en {execution_time:.2f} segundos")
        return result
    return wrapper

def save_figure(filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> str:
    """
    Guarda la figura actual de matplotlib en el directorio de figuras.
    
    Args:
        filename: Nombre del archivo
        dpi: Resolución de la imagen
        bbox_inches: Ajuste de bordes
        
    Returns:
        Ruta completa donde se guardó la figura
    """
    # Crear directorio si no existe
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Construir ruta completa
    filepath = os.path.join(FIGURES_DIR, filename)
    
    # Guardar figura
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figura guardada en {filepath}")
    
    return filepath

def detect_outliers_iqr(data: Union[pd.Series, np.ndarray], threshold: float = 1.5) -> Dict[str, Any]:
    """
    Detecta outliers usando el método IQR.
    
    Args:
        data: Serie o array con datos
        threshold: Multiplicador para IQR
        
    Returns:
        Diccionario con información sobre outliers
    """
    # Convertir a numpy array si es necesario
    if isinstance(data, pd.Series):
        data_array = data.values
    else:
        data_array = data
    
    # Calcular cuartiles e IQR
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    
    # Calcular límites
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Identificar outliers
    outliers = np.logical_or(data_array < lower_bound, data_array > upper_bound)
    outliers_count = np.sum(outliers)
    outliers_percentage = outliers_count / len(data_array) * 100
    
    return {
        'count': outliers_count,
        'percentage': outliers_percentage,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'mask': outliers
    }

def print_section_header(title: str, width: int = 80) -> None:
    """
    Imprime un encabezado de sección formateado.
    
    Args:
        title: Título de la sección
        width: Ancho del encabezado
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_step(step_number: int, step_description: str) -> None:
    """
    Imprime un paso numerado en un proceso.
    
    Args:
        step_number: Número del paso
        step_description: Descripción del paso
    """
    print(f"\n[Paso {step_number}] {step_description}")
    print("-" * 80)

def format_time(seconds: float) -> str:
    """
    Formatea un tiempo en segundos a un formato legible.
    
    Args:
        seconds: Tiempo en segundos
        
    Returns:
        Tiempo formateado
    """
    if seconds < 60:
        return f"{seconds:.2f} segundos"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)} minutos y {remaining_seconds:.2f} segundos"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{int(hours)} horas, {int(minutes)} minutos y {seconds:.2f} segundos"

def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Calcula y formatea el uso de memoria de un DataFrame.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Uso de memoria formateado
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    # Convertir a unidades legibles
    if memory_bytes < 1024:
        return f"{memory_bytes} bytes"
    elif memory_bytes < 1024**2:
        return f"{memory_bytes/1024:.2f} KB"
    elif memory_bytes < 1024**3:
        return f"{memory_bytes/1024**2:.2f} MB"
    else:
        return f"{memory_bytes/1024**3:.2f} GB"