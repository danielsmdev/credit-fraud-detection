"""
Configuración para el proyecto de detección de fraude en tarjetas de crédito.
Contiene constantes, rutas y parámetros de configuración.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Rutas de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODELS_REPORT_DIR = os.path.join(REPORTS_DIR, 'models')
EVALUATION_DIR = os.path.join(REPORTS_DIR, 'evaluation')

# Crear directorios si no existen
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                 FIGURES_DIR, MODELS_REPORT_DIR, EVALUATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Rutas de archivos
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'creditcard.csv')
X_TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'X_train.csv')
X_TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'X_test.csv')
Y_TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')
Y_TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')
SCALER_FILE = os.path.join(PROCESSED_DATA_DIR, 'standard_scaler.pkl')
BEST_PARAMS_FILE = os.path.join(MODELS_REPORT_DIR, 'best_params.json')
OPTIMAL_THRESHOLD_FILE = os.path.join(MODELS_REPORT_DIR, 'optimal_threshold.json')
MODEL_COMPARISON_FILE = os.path.join(EVALUATION_DIR, 'model_comparison.csv')
THRESHOLD_SENSITIVITY_FILE = os.path.join(EVALUATION_DIR, 'threshold_sensitivity.csv')
OPTIMAL_THRESHOLDS_FILE = os.path.join(EVALUATION_DIR, 'optimal_thresholds.json')
FALSE_POSITIVES_FILE = os.path.join(EVALUATION_DIR, 'false_positives.csv')
FALSE_NEGATIVES_FILE = os.path.join(EVALUATION_DIR, 'false_negatives.csv')

# Parámetros generales
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5  # Umbral por defecto para clasificación

# Parámetros para tratamiento de outliers
OUTLIER_THRESHOLD = 1.5  # Para método IQR

# Parámetros para costo de clasificación errónea
FN_COST = 10  # Costo de no detectar un fraude (alto)
FP_COST = 1   # Costo de falsa alarma (bajo)

# Configuración de visualización
def set_plot_style():
    """Configura el estilo de visualización para gráficos."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Paleta de colores personalizada
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
        '#1abc9c', '#34495e', '#e67e22', '#7f8c8d', '#27ae60'
    ])

# Función para generar nombre de archivo con timestamp
def get_timestamped_filename(base_name, extension):
    """Genera un nombre de archivo con timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

# Parámetros para grids de hiperparámetros
def get_param_grids():
    """Retorna los grids de parámetros para búsqueda de hiperparámetros."""
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [1, 5, 10]  # Para manejar desbalanceo
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6, -1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 1.0],
            'class_weight': [None, 'balanced']
        }
    }
    return param_grids