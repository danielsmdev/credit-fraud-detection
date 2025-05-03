"""
Funciones auxiliares para visualización de resultados de modelos.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from src.utils import save_figure

def plot_all_confusion_matrices(metrics_dict: Dict[str, Dict[str, Any]], figsize=(15, 10)):
    """
    Visualiza matrices de confusión para múltiples modelos en una sola figura.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        figsize: Tamaño de la figura
    """
    # Determinar número de modelos y configurar subplots
    n_models = len(metrics_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        if i < len(axes):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
            axes[i].set_title(f'Matriz de Confusión - {name}')
            axes[i].set_ylabel('Valor Real')
            axes[i].set_xlabel('Valor Predicho')
    
    # Ocultar ejes no utilizados
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    save_figure('all_confusion_matrices.png')
    return fig

def plot_all_roc_curves(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, figsize=(10, 8)):
    """
    Visualiza curvas ROC para múltiples modelos en una sola gráfica.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Graficar línea de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador aleatorio')
    
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
            
            # Calcular AUC si no está en las métricas
            if 'roc_auc' in metrics:
                auc = metrics['roc_auc']
            else:
                auc = roc_auc_score(y_true, metrics['y_prob'])
            
            # Graficar curva ROC
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')
    
    # Configurar gráfica
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC para Múltiples Modelos')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    save_figure('all_roc_curves.png')
    
    return plt.gcf()

def plot_all_precision_recall_curves(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, figsize=(10, 8)):
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
                
            # Calcular curva precision-recall
            precision, recall, _ = precision_recall_curve(y_true, metrics['y_prob'])
            
            # Calcular AP si no está en las métricas
            if 'avg_precision' in metrics:
                ap = metrics['avg_precision']
            else:
                ap = average_precision_score(y_true, metrics['y_prob'])
            
            # Graficar curva precision-recall
            plt.plot(recall, precision, linewidth=2, label=f'{name} (AP = {ap:.4f})')
    
    # Configurar gráfica
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas Precision-Recall para Múltiples Modelos')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    save_figure('all_precision_recall_curves.png')
    
    return plt.gcf()

def create_model_comparison_dashboard(metrics_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray = None, figsize=(15, 12)):
    """
    Crea un dashboard completo con comparación de modelos.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        figsize: Tamaño de la figura
    """
    fig = plt.figure(figsize=figsize)
    
    # Definir grid para subplots
    gs = fig.add_gridspec(3, 2)
    
    # 1. Comparación de métricas principales
    ax1 = fig.add_subplot(gs[0, :])
    
    # Preparar datos para gráfico de barras
    models = list(metrics_dict.keys())
    metrics_to_plot = ['precision', 'recall', 'f1', 'roc_auc']
    metrics_data = {metric: [] for metric in metrics_to_plot}
    
    for model in models:
        for metric in metrics_to_plot:
            if metric in metrics_dict[model]:
                metrics_data[metric].append(metrics_dict[model][metric])
            else:
                metrics_data[metric].append(0)
    
    # Crear gráfico de barras agrupadas
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        ax1.bar(x + i*width - width*1.5, metrics_data[metric], width, label=metric.upper())
    
    ax1.set_title('Comparación de Métricas Principales')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Curvas ROC
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    
    for name, metrics in metrics_dict.items():
        if 'y_prob' in metrics and metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                continue
                
            fpr, tpr, _ = roc_curve(y_true, metrics['y_prob'])
            auc = metrics.get('roc_auc', roc_auc_score(y_true, metrics['y_prob']))
            ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
    
    ax2.set_title('Curvas ROC para Múltiples Modelos')
    ax2.set_xlabel('Tasa de Falsos Positivos')
    ax2.set_ylabel('Tasa de Verdaderos Positivos')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Curvas Precision-Recall
    ax3 = fig.add_subplot(gs[1, 1])
    
    for name, metrics in metrics_dict.items():
        if 'y_prob' in metrics and metrics['y_prob'] is not None:
            # Determinar las etiquetas verdaderas
            if 'y_test' in metrics:
                y_true = metrics['y_test']
            elif 'y_true' in metrics:
                y_true = metrics['y_true']
            elif y_test is not None:
                y_true = y_test
            else:
                continue
                
            precision, recall, _ = precision_recall_curve(y_true, metrics['y_prob'])
            ap = metrics.get('avg_precision', average_precision_score(y_true, metrics['y_prob']))
            ax3.plot(recall, precision, label=f'{name} (AP = {ap:.4f})')
    
    ax3.set_title('Curvas Precision-Recall para Múltiples Modelos')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Tabla de métricas detalladas
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Preparar datos para tabla
    table_data = []
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'roc_auc', 'avg_precision']
    
    for model in models:
        row = [model]
        for metric in metrics_list:
            if metric in metrics_dict[model]:
                row.append(f"{metrics_dict[model][metric]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1', 'F2', 'AUC', 'AP'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    
    # Guardar figura
    save_figure('model_comparison_dashboard.png')
    
    return fig

def plot_model_comparison_heatmap(metrics_dict: Dict[str, Dict[str, Any]], figsize=(12, 8)):
    """
    Crea un mapa de calor para comparar métricas entre modelos.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        figsize: Tamaño de la figura
    """
    # Preparar datos para el mapa de calor
    metrics_to_include = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'mcc', 'kappa', 'roc_auc', 'avg_precision']
    models = list(metrics_dict.keys())
    
    # Crear DataFrame
    data = []
    for model in models:
        row = {}
        for metric in metrics_to_include:
            if metric in metrics_dict[model]:
                row[metric] = metrics_dict[model][metric]
            else:
                row[metric] = np.nan
        data.append(row)
    
    df = pd.DataFrame(data, index=models)
    
    # Crear mapa de calor
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt='.4f', cmap='Blues', linewidths=0.5, cbar=True)
    plt.title('Comparación Detallada de Modelos')
    plt.tight_layout()
    
    # Guardar figura
    save_figure('model_comparison_heatmap.png')
    
    return plt.gcf()

def plot_cost_benefit_analysis(metrics_dict: Dict[str, Dict[str, Any]], 
                              y_test: np.ndarray = None,
                              cost_fp: float = 10,
                              cost_fn: float = 100,
                              benefit_tp: float = 50,
                              benefit_tn: float = 1,
                              model_name: str = None,
                              figsize=(10, 6)):
    """
    Realiza un análisis de costo-beneficio para diferentes umbrales de clasificación.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        y_test: Etiquetas verdaderas (opcional, si no están en metrics_dict)
        cost_fp: Costo de un falso positivo
        cost_fn: Costo de un falso negativo
        benefit_tp: Beneficio de un verdadero positivo
        benefit_tn: Beneficio de un verdadero negativo
        model_name: Nombre del modelo a analizar (si es None, se usa el mejor según AUC)
        figsize: Tamaño de la figura
    """
    # Si no se especifica un modelo, usar el mejor según AUC
    if model_name is None:
        best_model = max(metrics_dict.items(), key=lambda x: x[1].get('roc_auc', 0))
        model_name = best_model[0]
    
    # Obtener probabilidades y etiquetas verdaderas
    metrics = metrics_dict[model_name]
    if metrics['y_prob'] is None:
        print(f"El modelo {model_name} no tiene probabilidades predichas. No se puede realizar el análisis.")
        return None
    
    # Determinar las etiquetas verdaderas
    if 'y_test' in metrics:
        y_true = metrics['y_test']
    elif 'y_true' in metrics:
        y_true = metrics['y_true']
    elif y_test is not None:
        y_true = y_test
    else:
        print(f"No se encontraron etiquetas verdaderas para el modelo {model_name}.")
        return None
    
    # Calcular costo-beneficio para diferentes umbrales
    thresholds = np.linspace(0.01, 0.99, 20)
    total_costs = []
    
    for threshold in thresholds:
        y_pred = (metrics['y_prob'] >= threshold).astype(int)
        
        # Calcular matriz de confusión
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        # Calcular costo total
        total_cost = (cost_fp * fp) + (cost_fn * fn) - (benefit_tp * tp) - (benefit_tn * tn)
        total_costs.append(total_cost)
    
    # Encontrar umbral óptimo
    optimal_idx = np.argmin(total_costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = total_costs[optimal_idx]
    
    # Crear gráfico
    plt.figure(figsize=figsize)
    plt.plot(thresholds, total_costs, 'r-o')
    plt.axvline(x=optimal_threshold, color='g', linestyle='--')
    plt.axhline(y=0, color='k', linestyle=':')
    plt.title('Análisis de Costo-Beneficio')
    plt.xlabel('Umbral de Clasificación')
    plt.ylabel('Costo Total')
    plt.grid(True, alpha=0.3)
    
    # Añadir anotación para umbral óptimo
    plt.annotate(f'Umbral óptimo: {optimal_threshold:.2f}\nCosto: {optimal_cost:.2f}',
                xy=(optimal_threshold, optimal_cost),
                xytext=(optimal_threshold + 0.1, optimal_cost + 100),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    # Guardar figura
    save_figure('cost_benefit_analysis.png')
    
    return plt.gcf(), optimal_threshold

def plot_calibration_curves(metrics_dict, y_test=None, n_bins=10, figsize=(12, 8)):
    """
    Visualiza las curvas de calibración para todos los modelos.
    
    Args:
        metrics_dict (dict): Diccionario con métricas de todos los modelos.
        y_test (array): Etiquetas reales para el conjunto de prueba.
        n_bins (int): Número de bins para la calibración.
        figsize (tuple): Tamaño de la figura.
        
    Returns:
        matplotlib.figure.Figure: Figura con las curvas de calibración.
    """
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    
    # Colores para las líneas
    colors = plt.cm.get_cmap('tab10', len(metrics_dict))
    
    # Línea de referencia (calibración perfecta)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Graficar curva de calibración para cada modelo
    for i, (name, metrics) in enumerate(metrics_dict.items()):
        if 'y_prob' in metrics and metrics['y_prob'] is not None:
            y_prob = metrics['y_prob']
            
            # Si no se proporciona y_test, intentar obtenerlo de las métricas
            if y_test is None and 'y_true' in metrics:
                y_true = metrics['y_true']
            else:
                y_true = y_test
                
            if y_true is not None:
                # Calcular curva de calibración
                prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
                
                # Graficar curva
                plt.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                         color=colors(i), label=f"{name}")
    
    # Configurar gráfico
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend(loc='best')
    plt.grid(True)
    
    return plt.gcf()

