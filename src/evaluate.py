"""
Funciones para la evaluación de modelos en el proyecto de detección de fraude.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Añade esta línea
import joblib
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import glob

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve,
    fbeta_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import shap

from src.config import (
    RANDOM_STATE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE
)
from src.utils import timer_decorator, save_figure
from src.data_prep import load_processed_data
from src.model_training import load_model

@timer_decorator
def load_models(model_dir=MODELS_DIR, pattern='*.pkl', exclude_patterns=None):
    """
    Carga todos los modelos guardados en el directorio especificado.
    
    Args:
        model_dir: Directorio donde se encuentran los modelos
        pattern: Patrón para buscar archivos de modelos
        exclude_patterns: Lista de patrones para excluir archivos
        
    Returns:
        models: Diccionario con los modelos cargados
    """
    print(f"Cargando modelos desde {model_dir}...")
    
    # Configurar patrones de exclusión por defecto
    if exclude_patterns is None:
        exclude_patterns = ['best_metrics', 'best_params', 'ensemble_weights']
    
    # Buscar archivos de modelos
    model_files = glob.glob(os.path.join(model_dir, pattern))
    
    # También buscar archivos .joblib si existen
    model_files.extend(glob.glob(os.path.join(model_dir, "*.joblib")))
    
    # Filtrar archivos excluidos
    filtered_files = []
    for file in model_files:
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern in file:
                should_exclude = True
                break
        if not should_exclude:
            filtered_files.append(file)
    
    model_files = filtered_files
    
    if not model_files:
        print(f"ADVERTENCIA: No se encontraron archivos de modelo en {model_dir}")
        return {}
    
    models = {}
    for model_file in model_files:
        # Extraer nombre del modelo del nombre de archivo
        model_name = os.path.basename(model_file).split('.')[0]
        if '_20' in model_name:  # Remover timestamp si existe
            model_name = model_name.split('_20')[0]
        
        try:
            # Cargar modelo
            if model_file.endswith('.joblib'):
                model = joblib.load(model_file)
            else:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            
            # Verificar si es un modelo válido
            if hasattr(model, "predict") or hasattr(model, "predict_proba"):
                models[model_name] = model
                print(f"  Modelo '{model_name}' cargado desde {model_file}")
            else:
                print(f"  ADVERTENCIA: El archivo {model_file} no contiene un modelo válido")
        except Exception as e:
            print(f"  ERROR al cargar {model_file}: {e}")
    
    print(f"Total de modelos cargados: {len(models)}")
    return models

@timer_decorator
def evaluate_model_detailed(model, X_test, y_test, model_name=None, threshold=0.5):
    """
    Evalúa un modelo con métricas detalladas.
    
    Args:
        model: Modelo a evaluar
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        model_name: Nombre del modelo
        threshold: Umbral de probabilidad para clasificación
        
    Returns:
        metrics: Diccionario con métricas detalladas
    """
    if model_name is None:
        model_name = type(model).__name__
    
    print(f"Evaluando modelo: {model_name}")
    
    # Verificar si el objeto es realmente un modelo
    if not hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        print(f"ERROR: El objeto '{model_name}' no parece ser un modelo válido (es de tipo {type(model)})")
        print(f"Contenido del objeto: {model}")
        return None
    
    # Obtener predicciones y probabilidades
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
        else:
            y_prob = None
            y_pred = model.predict(X_test)
    except Exception as e:
        print(f"ERROR al hacer predicciones con el modelo '{model_name}': {e}")
        return None
    
    # Calcular métricas básicas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'f2': fbeta_score(y_test, y_pred, beta=2),  # F2 da más peso al recall
        'mcc': matthews_corrcoef(y_test, y_pred),  # Coeficiente de correlación de Matthews
        'kappa': cohen_kappa_score(y_test, y_pred),  # Kappa de Cohen
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude']),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # Imprimir métricas principales
    print(f"Métricas para {model_name}:")
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'classification_report', 'y_pred', 'y_prob']:
            print(f"  {metric}: {value:.4f}")
    
    return metrics

@timer_decorator
def evaluate_all_models(models, X_test, y_test, threshold=0.5):
    """
    Evalúa todos los modelos y compara su rendimiento.
    
    Args:
        models: Diccionario con los modelos a evaluar
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        threshold: Umbral de probabilidad para clasificación
        
    Returns:
        all_metrics: Diccionario con métricas para todos los modelos
        comparison_df: DataFrame con comparación de modelos
    """
    print("Evaluando todos los modelos...")
    
    all_metrics = {}
    
    for name, model in models.items():
        metrics = evaluate_model_detailed(model, X_test, y_test, name, threshold)
        if metrics is not None:  # Solo añadir si la evaluación fue exitosa
            all_metrics[name] = metrics
    
    # Verificar si hay modelos evaluados
    if not all_metrics:
        print("ERROR: No se pudo evaluar ningún modelo correctamente")
        return {}, pd.DataFrame()
    
    # Comparar modelos
    comparison_df = compare_models_detailed(all_metrics)
    
    return all_metrics, comparison_df

def compare_models_detailed(metrics_dict):
    """
    Compara el rendimiento de varios modelos con métricas detalladas.
    
    Args:
        metrics_dict: Diccionario con métricas de rendimiento por modelo
        
    Returns:
        comparison_df: DataFrame con comparación de modelos
    """
    # Crear DataFrame con métricas
    comparison_data = []
    for model_name, metrics in metrics_dict.items():
        model_metrics = {
            'Modelo': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'F2': metrics['f2'],
            'MCC': metrics['mcc'],
            'Kappa': metrics['kappa']
        }
        
        if 'roc_auc' in metrics:
            model_metrics['AUC'] = metrics['roc_auc']
        
        if 'avg_precision' in metrics:
            model_metrics['AP'] = metrics['avg_precision']
        
        comparison_data.append(model_metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Ordenar por F1 score (descendente)
    comparison_df = comparison_df.sort_values('F1', ascending=False)
    
    print("\nComparación detallada de modelos:")
    print(comparison_df)
    
    # Guardar comparación
    os.makedirs(REPORTS_DIR, exist_ok=True)
    comparison_path = os.path.join(REPORTS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparación guardada en {comparison_path}")
    
    # Visualizar comparación
    plt.figure(figsize=(14, 8))
    
    # Seleccionar columnas numéricas
    metric_cols = [col for col in comparison_df.columns if col != 'Modelo']
    
    # Crear heatmap
    sns.heatmap(comparison_df.set_index('Modelo')[metric_cols], annot=True, 
                cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('Comparación Detallada de Modelos')
    plt.tight_layout()
    
    # Guardar figura
    save_figure('model_comparison_heatmap.png')
    
    plt.show()
    
    return comparison_df

def plot_all_confusion_matrices(metrics_dict, figsize=(12, 10)):
    """
    Grafica las matrices de confusión para todos los modelos.
    
    Args:
        metrics_dict: Diccionario con métricas de rendimiento por modelo
        figsize: Tamaño de la figura
    """
    n_models = len(metrics_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        cm = metrics['confusion_matrix']
        
        # Normalizar matriz
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
        axes[i].set_title(f'Matriz de Confusión - {name}')
        axes[i].set_ylabel('Etiqueta Real')
        axes[i].set_xlabel('Etiqueta Predicha')
        axes[i].set_xticks([0.5, 1.5])
        axes[i].set_xticklabels(['No Fraude', 'Fraude'])
        axes[i].set_yticks([0.5, 1.5])
        axes[i].set_yticklabels(['No Fraude', 'Fraude'])
    
    # Ocultar ejes vacíos
    for i in range(n_models, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Guardar figura
    save_figure('all_confusion_matrices.png')
    
    plt.show()

@timer_decorator
def plot_all_roc_curves(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualiza curvas ROC para múltiples modelos en una sola gráfica.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    for name, metrics in metrics_dict.items():
        if metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                print(f"Advertencia: No se encontraron etiquetas verdaderas para el modelo {name}. Saltando.")
                continue
                
            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_true, metrics['y_prob'])
            roc_auc = metrics.get('roc_auc', roc_auc_score(y_true, metrics['y_prob']))
            
            # Graficar curva ROC
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Graficar línea de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Configurar gráfica
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC para Múltiples Modelos')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    save_figure('all_roc_curves.png')
    
    plt.show()

@timer_decorator
def plot_all_precision_recall_curves(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualiza curvas de precisión-recall para múltiples modelos en una sola gráfica.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    for name, metrics in metrics_dict.items():
        if metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                print(f"Advertencia: No se encontraron etiquetas verdaderas para el modelo {name}. Saltando.")
                continue
                
            # Calcular curva de precisión-recall
            precision, recall, _ = precision_recall_curve(y_true, metrics['y_prob'])
            
            # Obtener o calcular average precision
            if 'avg_precision' in metrics:
                avg_precision = metrics['avg_precision']
            else:
                avg_precision = average_precision_score(y_true, metrics['y_prob'])
            
            # Graficar curva de precisión-recall
            plt.plot(recall, precision, lw=2, label=f'{name} (AP = {avg_precision:.4f})')
    
    # Configurar gráfica
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas Precision-Recall para Múltiples Modelos')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    save_figure('all_precision_recall_curves.png')
    
    plt.show()

@timer_decorator
def plot_calibration_curves(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, n_bins: int = 10, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualiza curvas de calibración para múltiples modelos en una sola gráfica.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        n_bins: Número de bins para la curva de calibración
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Graficar línea de referencia (calibración perfecta)
    plt.plot([0, 1], [0, 1], 'k--', label='Calibración perfecta')
    
    for name, metrics in metrics_dict.items():
        if metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                print(f"Advertencia: No se encontraron etiquetas verdaderas para el modelo {name}. Saltando.")
                continue
                
            # Calcular curva de calibración
            prob_true, prob_pred = calibration_curve(
                y_true, 
                metrics['y_prob'], 
                n_bins=n_bins
            )
            
            # Graficar curva de calibración
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=name)
    
    # Configurar gráfica
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Probabilidad media predicha')
    plt.ylabel('Fracción de positivos')
    plt.title('Curvas de Calibración para Múltiples Modelos')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    save_figure('calibration_curves.png')
    
    plt.show()

@timer_decorator
def analyze_feature_importance(model, X_test, y_test, feature_names=None, n_repeats=10, top_n=20, figsize=(12, 8)):
    """
    Analiza la importancia de las características usando permutation importance.
    
    Args:
        model: Modelo a analizar
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        feature_names: Nombres de las características
        n_repeats: Número de repeticiones para permutation importance
        top_n: Número de características principales a mostrar
        figsize: Tamaño de la figura
        
    Returns:
        importance_df: DataFrame con importancia de características
    """
    print("Analizando importancia de características...")
    
    # Obtener nombres de características si no se proporcionan
    if feature_names is None:
        if hasattr(X_test, 'columns'):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
    
    # Calcular permutation importance
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=n_repeats, 
        random_state=RANDOM_STATE,
        scoring='f1'
    )
    
    # Crear DataFrame con resultados
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })
    
    # Ordenar por importancia
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Guardar resultados
    os.makedirs(REPORTS_DIR, exist_ok=True)
    importance_path = os.path.join(REPORTS_DIR, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Importancia de características guardada en {importance_path}")
    
    # Visualizar top N características
    plt.figure(figsize=figsize)
    top_features = importance_df.head(top_n)
    
    # Crear gráfico de barras horizontales
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title(f'Top {top_n} Características Más Importantes')
    plt.xlabel('Importancia')
    plt.tight_layout()
    
    # Guardar figura
    save_figure('feature_importance.png')
    
    plt.show()
    
    return importance_df

@timer_decorator
def analyze_shap_values(model, X_test, feature_names=None, max_display=20, figsize=(12, 8)):
    """
    Analiza los valores SHAP para interpretabilidad del modelo.
    
    Args:
        model: Modelo a analizar
        X_test: Características de prueba
        feature_names: Nombres de las características
        max_display: Número máximo de características a mostrar
        figsize: Tamaño de la figura
        
    Returns:
        shap_values: Valores SHAP calculados
    """
    print("Analizando valores SHAP...")
    
    # Convertir a numpy array si es DataFrame
    if hasattr(X_test, 'values'):
        X_test_array = X_test.values
        if feature_names is None:
            feature_names = X_test.columns.tolist()
    else:
        X_test_array = X_test
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
    
    # Seleccionar una muestra aleatoria si X_test es grande
    if X_test_array.shape[0] > 1000:
        np.random.seed(RANDOM_STATE)
        sample_indices = np.random.choice(X_test_array.shape[0], 1000, replace=False)
        X_test_sample = X_test_array[sample_indices]
    else:
        X_test_sample = X_test_array
    
    try:
        # Crear explainer SHAP
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model.predict_proba, X_test_sample)
            shap_values = explainer(X_test_sample)
            shap_values = shap_values[:, :, 1]  # Seleccionar valores para clase positiva
        else:
            explainer = shap.Explainer(model.predict, X_test_sample)
            shap_values = explainer(X_test_sample)
        
        # Graficar resumen de valores SHAP
        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, 
                          max_display=max_display, show=False)
        plt.title('Resumen de Valores SHAP')
        plt.tight_layout()
        
        # Guardar figura
        save_figure('shap_summary.png')
        
        plt.show()
        
        # Graficar gráfico de dependencia para las características más importantes
        top_features = np.argsort(-np.abs(shap_values.values).mean(0))[:3]
        
        for i, feature_idx in enumerate(top_features):
            plt.figure(figsize=figsize)
            shap.dependence_plot(feature_idx, shap_values.values, X_test_sample, 
                                feature_names=feature_names, show=False)
            plt.title(f'Gráfico de Dependencia SHAP para {feature_names[feature_idx]}')
            plt.tight_layout()
            
            # Guardar figura
            save_figure(f'shap_dependence_{feature_names[feature_idx]}.png')
            
            plt.show()
        
        return shap_values
    
    except Exception as e:
        print(f"Error al calcular valores SHAP: {e}")
        return None

def analyze_threshold_impact(y_test, y_prob, thresholds=None, figsize=(12, 8)):
    """
    Analiza el impacto del umbral de clasificación en las métricas.
    
    Args:
        y_test: Variable objetivo de prueba
        y_prob: Probabilidades predichas
        thresholds: Lista de umbrales a evaluar
        figsize: Tamaño de la figura
        
    Returns:
        threshold_df: DataFrame con métricas para cada umbral
    """
    print("Analizando impacto del umbral de clasificación...")
    
    # Definir umbrales si no se proporcionan
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    # Calcular métricas para cada umbral
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        
        results.append({
            'Threshold': threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'F2': f2
        })
    
    # Crear DataFrame con resultados
    threshold_df = pd.DataFrame(results)
    
    # Guardar resultados
    os.makedirs(REPORTS_DIR, exist_ok=True)
    threshold_path = os.path.join(REPORTS_DIR, 'threshold_analysis.csv')
    threshold_df.to_csv(threshold_path, index=False)
    print(f"Análisis de umbrales guardado en {threshold_path}")
    
    # Visualizar impacto del umbral
    plt.figure(figsize=figsize)
    
    # Graficar métricas vs umbral
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'F2']:
        plt.plot(threshold_df['Threshold'], threshold_df[metric], marker='o', label=metric)
    
    plt.xlabel('Umbral de Clasificación')
    plt.ylabel('Puntuación')
    plt.title('Impacto del Umbral de Clasificación en las Métricas')
    plt.legend()
    plt.grid(True)
    
    # Guardar figura
    save_figure('threshold_impact.png')
    
    plt.show()
    
    # Encontrar umbral óptimo para F1
    best_f1_idx = threshold_df['F1'].idxmax()
    best_f1_threshold = threshold_df.loc[best_f1_idx, 'Threshold']
    print(f"Umbral óptimo para F1: {best_f1_threshold:.2f} (F1 = {threshold_df.loc[best_f1_idx, 'F1']:.4f})")
    
    # Encontrar umbral óptimo para F2
    best_f2_idx = threshold_df['F2'].idxmax()
    best_f2_threshold = threshold_df.loc[best_f2_idx, 'Threshold']
    print(f"Umbral óptimo para F2: {best_f2_threshold:.2f} (F2 = {threshold_df.loc[best_f2_idx, 'F2']:.4f})")
    
    return threshold_df

def analyze_cost_benefit(y_test, y_prob, cost_matrix=None, thresholds=None, figsize=(12, 8)):
    """
    Analiza el costo-beneficio para diferentes umbrales de clasificación.
    
    Args:
        y_test: Variable objetivo de prueba
        y_prob: Probabilidades predichas
        cost_matrix: Matriz de costos [TN, FP, FN, TP]
        thresholds: Lista de umbrales a evaluar
        figsize: Tamaño de la figura
        
    Returns:
        cost_df: DataFrame con costos para cada umbral
    """
    print("Analizando costo-beneficio...")
    
    # Definir matriz de costos si no se proporciona
    if cost_matrix is None:
        # Valores por defecto: [TN, FP, FN, TP]
        # TN = 0 (no costo por clasificar correctamente no fraude)
        # FP = -10 (costo por falsa alarma)
        # FN = -100 (costo por no detectar fraude)
        # TP = 50 (beneficio por detectar fraude)
        cost_matrix = [0, -10, -100, 50]
    
    # Definir umbrales si no se proporcionan
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    # Calcular costo-beneficio para cada umbral
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calcular matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calcular costo total
        total_cost = (tn * cost_matrix[0] + fp * cost_matrix[1] + 
                      fn * cost_matrix[2] + tp * cost_matrix[3])
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Threshold': threshold,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'Total_Cost': total_cost,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
    
    # Crear DataFrame con resultados
    cost_df = pd.DataFrame(results)
    
    # Guardar resultados
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cost_path = os.path.join(REPORTS_DIR, 'cost_benefit_analysis.csv')
    cost_df.to_csv(cost_path, index=False)
    print(f"Análisis de costo-beneficio guardado en {cost_path}")
    
    # Visualizar costo-beneficio
    plt.figure(figsize=figsize)
    
    # Graficar costo total vs umbral
    plt.plot(cost_df['Threshold'], cost_df['Total_Cost'], marker='o', color='red', linewidth=2)
    
    plt.xlabel('Umbral de Clasificación')
    plt.ylabel('Costo Total')
    plt.title('Análisis de Costo-Beneficio')
    plt.grid(True)
    
    # Guardar figura
    save_figure('cost_benefit_analysis.png')
    
    plt.show()
    
    # Encontrar umbral óptimo para costo
    best_cost_idx = cost_df['Total_Cost'].idxmax()
    best_cost_threshold = cost_df.loc[best_cost_idx, 'Threshold']
    print(f"Umbral óptimo para costo: {best_cost_threshold:.2f} (Costo = {cost_df.loc[best_cost_idx, 'Total_Cost']:.2f})")
    
    return cost_df

@timer_decorator
def generate_evaluation_report(metrics_dict, output_file=None):
    """
    Genera un informe de evaluación detallado.
    
    Args:
        metrics_dict: Diccionario con métricas de rendimiento por modelo
        output_file: Ruta del archivo de salida
        
    Returns:
        report_text: Texto del informe
    """
    print("Generando informe de evaluación...")
    
    # Crear directorio si no existe
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Definir archivo de salida si no se proporciona
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(REPORTS_DIR, f'evaluation_report_{timestamp}.txt')
    
    # Crear informe
    report_text = "INFORME DE EVALUACIÓN DE MODELOS\n"
    report_text += "=" * 50 + "\n\n"
    
    # Fecha y hora
    report_text += f"Fecha y hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Resumen de modelos
    report_text += "RESUMEN DE MODELOS\n"
    report_text += "-" * 50 + "\n"
    
    # Crear tabla de comparación
    comparison_data = []
    for name, metrics in metrics_dict.items():
        model_metrics = {
            'Modelo': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        }
        
        if 'roc_auc' in metrics:
            model_metrics['AUC'] = metrics['roc_auc']
        
        if 'avg_precision' in metrics:
            model_metrics['AP'] = metrics['avg_precision']
        
        comparison_data.append(model_metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1', ascending=False)
    
    # Añadir tabla de comparación al informe
    report_text += comparison_df.to_string() + "\n\n"
    
    # Detalles por modelo
    report_text += "DETALLES POR MODELO\n"
    report_text += "-" * 50 + "\n\n"
    
    for name, metrics in metrics_dict.items():
        report_text += f"Modelo: {name}\n"
        report_text += "=" * len(f"Modelo: {name}") + "\n"
        
        # Métricas principales
        for metric, value in metrics.items():
            if metric not in ['confusion_matrix', 'classification_report', 'y_pred', 'y_prob']:
                report_text += f"{metric}: {value:.4f}\n"
        
        # Informe de clasificación
        if 'classification_report' in metrics:
            report_text += "\nInforme de Clasificación:\n"
            report_text += metrics['classification_report'] + "\n"
        
        # Matriz de confusión
        if 'confusion_matrix' in metrics:
            report_text += "\nMatriz de Confusión:\n"
            cm = metrics['confusion_matrix']
            report_text += f"[[{cm[0, 0]}, {cm[0, 1]}]\n"
            report_text += f" [{cm[1, 0]}, {cm[1, 1]}]]\n"
        
        report_text += "\n" + "-" * 50 + "\n\n"
    
    # Conclusiones
    report_text += "CONCLUSIONES\n"
    report_text += "-" * 50 + "\n"
    
    # Encontrar mejor modelo según F1
    best_model = comparison_df.iloc[0]['Modelo']
    best_f1 = comparison_df.iloc[0]['F1']
    
    report_text += f"El mejor modelo según F1 es: {best_model} (F1 = {best_f1:.4f})\n\n"
    
    # Recomendaciones
    report_text += "Recomendaciones:\n"
    report_text += "1. Utilizar el modelo con mejor F1 para un equilibrio entre precisión y recall.\n"
    report_text += "2. Si se prioriza minimizar falsos negativos, considerar el modelo con mejor recall.\n"
    report_text += "3. Si se prioriza minimizar falsos positivos, considerar el modelo con mejor precisión.\n"
    report_text += "4. Considerar ajustar el umbral de clasificación según el análisis de costo-beneficio.\n"
    
    # Guardar informe
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"Informe guardado en {output_file}")
    
    return report_text

@timer_decorator
def evaluation_pipeline(model_dir=MODELS_DIR, threshold=0.5, cost_matrix=None):
    """
    Pipeline completo de evaluación de modelos.
    
    Args:
        model_dir: Directorio donde se encuentran los modelos
        threshold: Umbral de probabilidad para clasificación
        cost_matrix: Matriz de costos [TN, FP, FN, TP]
        
    Returns:
        all_metrics: Diccionario con métricas para todos los modelos
        comparison_df: DataFrame con comparación de modelos
    """
    print("Iniciando pipeline de evaluación de modelos...")
    
    # Cargar datos de prueba
    X_train, X_test, y_train, y_test, _ = load_processed_data()
    
    # Cargar modelos
    models = load_models(model_dir)
    
    # Evaluar todos los modelos
    all_metrics, comparison_df = evaluate_all_models(models, X_test, y_test, threshold)
    
    # Visualizar matrices de confusión
    plot_all_confusion_matrices(all_metrics)
    
    # Visualizar curvas ROC
    plot_all_roc_curves(all_metrics)
    
    # Visualizar curvas de precisión-recall
    plot_all_precision_recall_curves(all_metrics)
    
    # Visualizar curvas de calibración
    plot_calibration_curves(all_metrics)
    
    # Seleccionar mejor modelo según F1
    best_model_name = comparison_df.iloc[0]['Modelo']
    best_model = models[best_model_name]
    best_metrics = all_metrics[best_model_name]
    
    print(f"\nMejor modelo según F1: {best_model_name}")
    
    # Analizar importancia de características para el mejor modelo
    feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else None
    importance_df = analyze_feature_importance(best_model, X_test, y_test, feature_names)
    
    # Analizar valores SHAP para el mejor modelo
    shap_values = analyze_shap_values(best_model, X_test, feature_names)
    
    # Analizar impacto del umbral para el mejor modelo
    if best_metrics['y_prob'] is not None:
        threshold_df = analyze_threshold_impact(y_test, best_metrics['y_prob'])
        
        # Analizar costo-beneficio
        cost_df = analyze_cost_benefit(y_test, best_metrics['y_prob'], cost_matrix)
    
    # Generar informe de evaluación
    generate_evaluation_report(all_metrics)
    
    print("Pipeline de evaluación completado.")
    return all_metrics, comparison_df

if __name__ == "__main__":
    # Ejecutar pipeline completo
    all_metrics, comparison_df = evaluation_pipeline()
