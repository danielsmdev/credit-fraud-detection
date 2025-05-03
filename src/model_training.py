"""
Funciones para el entrenamiento y optimización de modelos de detección de fraude.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve,
    fbeta_score, matthews_corrcoef, cohen_kappa_score
)

from src.config import (
    RANDOM_STATE, MODELS_DIR, FIGURES_DIR, REPORTS_DIR
)
from src.utils import timer_decorator, save_figure

@timer_decorator
def get_base_models() -> Dict[str, Any]:
    """
    Retorna un diccionario con modelos base para detección de fraude.
    
    Returns:
        Diccionario con modelos base
    """
    models = {
        'logistic_regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        'xgboost': XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=1  # Ajustado durante optimización
        ),
        'lightgbm': LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    }
    
    return models

@timer_decorator
def get_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Retorna grids de parámetros reducidos para optimización eficiente.
    
    Returns:
        Diccionario con grids de parámetros por modelo
    """
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'scale_pos_weight': [1, 5]
        },
        'lightgbm': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, -1],
            'num_leaves': [31, 50],
            'subsample': [0.8]
        }
    }
    
    return param_grids

@timer_decorator
def train_base_model(model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
    """
    Entrena un modelo base y evalúa su rendimiento.
    
    Args:
        model: Modelo a entrenar
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        Modelo entrenado y diccionario con métricas
    """
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)
    
    return model, metrics

@timer_decorator
def train_all_base_models(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Entrena todos los modelos base y evalúa su rendimiento.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        Diccionario con modelos entrenados y diccionario con métricas
    """
    # Obtener modelos base
    base_models = get_base_models()
    
    # Entrenar y evaluar cada modelo
    trained_models = {}
    metrics = {}
    
    for name, model in base_models.items():
        print(f"\nEntrenando modelo base: {name}")
        trained_model, model_metrics = train_base_model(model, X_train, y_train, X_test, y_test)
        trained_models[name] = trained_model
        metrics[name] = model_metrics
        
        # Imprimir métricas principales
        print(f"  Accuracy: {model_metrics['accuracy']:.4f}")
        print(f"  Precision: {model_metrics['precision']:.4f}")
        print(f"  Recall: {model_metrics['recall']:.4f}")
        print(f"  F1 Score: {model_metrics['f1']:.4f}")
        if 'roc_auc' in model_metrics:
            print(f"  AUC-ROC: {model_metrics['roc_auc']:.4f}")
    
    # Generar visualizaciones consolidadas
    from src.visualization_helpers import (
        plot_all_confusion_matrices,
        plot_all_roc_curves,
        plot_all_precision_recall_curves,
        create_model_comparison_dashboard,
        plot_model_comparison_heatmap
    )
    
    # Añadir y_test a las métricas para visualizaciones
    for name in metrics:
        metrics[name]['y_test'] = y_test
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones para modelos base...")
    plot_all_confusion_matrices(metrics)
    plot_all_roc_curves(metrics)
    plot_all_precision_recall_curves(metrics)
    create_model_comparison_dashboard(metrics)
    plot_model_comparison_heatmap(metrics)
    
    return trained_models, metrics

@timer_decorator
def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evalúa un modelo entrenado en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        Diccionario con métricas de evaluación
    """
    # Predecir clases
    y_pred = model.predict(X_test)
    
    # Predecir probabilidades si el modelo lo soporta
    y_prob = None
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_prob = None
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'f2': fbeta_score(y_test, y_pred, beta=2),  # Da más peso al recall
        'mcc': matthews_corrcoef(y_test, y_pred),  # Coeficiente de correlación de Matthews
        'kappa': cohen_kappa_score(y_test, y_pred),  # Kappa de Cohen
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'y_test': y_test  # Añadir etiquetas verdaderas para visualizaciones
    }
    
    # Calcular AUC-ROC si hay probabilidades
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        metrics['avg_precision'] = average_precision_score(y_test, y_prob)
    
    return metrics

@timer_decorator
def optimize_hyperparameters(model: Any, param_grid: Dict[str, List[Any]], 
                            X_train: np.ndarray, y_train: np.ndarray,
                            search_method: str = 'random', n_iter: int = 10,
                            cv: int = 3, scoring: str = 'f1',
                            sample_size: float = 0.3) -> Tuple[Any, Dict[str, Any], float]:
    """
    Optimiza hiperparámetros de un modelo usando búsqueda aleatoria.
    
    Args:
        model: Modelo base
        param_grid: Grid de parámetros
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        search_method: Método de búsqueda ('random' o 'grid')
        n_iter: Número de iteraciones para búsqueda aleatoria
        cv: Número de folds para validación cruzada
        scoring: Métrica para optimización
        sample_size: Fracción de datos a usar para optimización
        
    Returns:
        Mejor modelo, mejores parámetros y mejor puntuación
    """
    # Usar una muestra de los datos para optimización si el conjunto es grande
    if X_train.shape[0] > 10000 and sample_size < 1.0:
        print(f"Usando {sample_size:.0%} de los datos para optimización...")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=sample_size, 
            random_state=RANDOM_STATE, stratify=y_train
        )
    else:
        X_sample, y_sample = X_train, y_train
    
    print(f"Tamaño de muestra para optimización: {X_sample.shape[0]} muestras")
    
    # Configurar validación cruzada estratificada
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    # Configurar búsqueda de hiperparámetros
    if search_method.lower() == 'random':
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv_strategy, 
            scoring=scoring, random_state=RANDOM_STATE, n_jobs=-1,
            verbose=1
        )
    else:
        raise ValueError("Solo se soporta búsqueda aleatoria ('random')")
    
    # Realizar búsqueda
    search.fit(X_sample, y_sample)
    
    # Obtener mejor modelo y parámetros
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    
    print(f"Mejor puntuación de validación cruzada: {best_score:.4f}")
    print("Mejores parámetros:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Reentrenar en todos los datos si se usó una muestra
    if X_sample.shape[0] < X_train.shape[0]:
        print("Reentrenando mejor modelo en todos los datos...")
        best_model.fit(X_train, y_train)
    
    return best_model, best_params, best_score

@timer_decorator
def optimize_all_models(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       search_method: str = 'random', n_iter: int = 10,
                       cv: int = 3, scoring: str = 'f1',
                       sample_size: float = 0.3) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Optimiza hiperparámetros para todos los modelos.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        search_method: Método de búsqueda ('random' o 'grid')
        n_iter: Número de iteraciones para búsqueda aleatoria
        cv: Número de folds para validación cruzada
        scoring: Métrica para optimización
        sample_size: Fracción de datos a usar para optimización
        
    Returns:
        Diccionarios con mejores modelos, mejores parámetros y métricas
    """
    # Obtener modelos base y grids de parámetros
    base_models = get_base_models()
    param_grids = get_param_grids()
    
    # Optimizar cada modelo
    best_models = {}
    best_params = {}
    best_metrics = {}
    
    for name, model in base_models.items():
        print(f"\n{'='*50}")
        print(f"Optimizando modelo: {name}")
        print(f"{'='*50}")
        
        # Optimizar hiperparámetros
        best_model, params, _ = optimize_hyperparameters(
            model, param_grids[name], X_train, y_train,
            search_method=search_method, n_iter=n_iter,
            cv=cv, scoring=scoring, sample_size=sample_size
        )
        
        # Evaluar mejor modelo
        metrics = evaluate_model(best_model, X_test, y_test)
        
        # Guardar resultados
        best_models[name] = best_model
        best_params[name] = params
        best_metrics[name] = metrics
        
        # Imprimir métricas principales
        print(f"\nMétricas para {name} optimizado:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  AUC-ROC: {metrics['roc_auc']:.4f}")
    
    # Generar visualizaciones consolidadas
    from src.visualization_helpers import (
        plot_all_confusion_matrices,
        plot_all_roc_curves,
        plot_all_precision_recall_curves,
        create_model_comparison_dashboard,
        plot_model_comparison_heatmap,
        plot_cost_benefit_analysis
    )
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones para modelos optimizados...")
    plot_all_confusion_matrices(best_metrics)
    plot_all_roc_curves(best_metrics)
    plot_all_precision_recall_curves(best_metrics)
    create_model_comparison_dashboard(best_metrics)
    plot_model_comparison_heatmap(best_metrics)
    
    # Análisis de costo-beneficio para el mejor modelo según AUC
    best_auc_model = max(best_metrics.items(), key=lambda x: x[1].get('roc_auc', 0))[0]
    plot_cost_benefit_analysis(best_metrics, model_name=best_auc_model)
    
    return best_models, best_params, best_metrics

@timer_decorator
def compare_models(metrics_dict):
    """
    Compara el rendimiento de varios modelos.
    
    Args:
        metrics_dict (dict): Diccionario con métricas de todos los modelos.
        
    Returns:
        pandas.DataFrame: DataFrame con comparación de modelos.
    """
    import pandas as pd
    import numpy as np
    
    # Crear lista para almacenar datos de comparación
    comparison_data = []
    
    # Extraer métricas relevantes para cada modelo
    for name, metrics in metrics_dict.items():
        model_data = {'Modelo': name}
        
        # Añadir métricas relevantes
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'f2', 'mcc', 'kappa', 'roc_auc', 'avg_precision']:
            if metric in metrics:
                # Verificar si el valor es un array
                if isinstance(metrics[metric], np.ndarray):
                    model_data[metric] = np.mean(metrics[metric])
                else:
                    model_data[metric] = metrics[metric]
        
        comparison_data.append(model_data)
    
    # Crear DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Ordenar por F1 Score (descendente)
    if 'f1' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('f1', ascending=False)
    
    return comparison_df


@timer_decorator
def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """
    Visualiza matriz de confusión.
    
    Args:
        cm: Matriz de confusión
        model_name: Nombre del modelo
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.tight_layout()
    
    # Guardar figura
    save_figure(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    
    plt.show()

@timer_decorator
def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    """
    Visualiza curva ROC.
    
    Args:
        y_true: Etiquetas verdaderas
        y_prob: Probabilidades predichas
        model_name: Nombre del modelo
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar figura
    save_figure(f'roc_curve_{model_name.replace(" ", "_").lower()}.png')
    
    plt.show()

@timer_decorator
def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    """
    Visualiza curva Precision-Recall.
    
    Args:
        y_true: Etiquetas verdaderas
        y_prob: Probabilidades predichas
        model_name: Nombre del modelo
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall - {model_name}')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar figura
    save_figure(f'pr_curve_{model_name.replace(" ", "_").lower()}.png')
    
    plt.show()

@timer_decorator
def save_model(model: Any, model_name: str) -> str:
    """
    Guarda un modelo entrenado.
    
    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo
        
    Returns:
        Ruta donde se guardó el modelo
    """
    # Crear directorio si no existe
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Construir ruta
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    
    # Guardar modelo
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path

@timer_decorator
def load_model(model_path: str) -> Any:
    """
    Carga un modelo guardado.
    
    Args:
        model_path: Ruta al modelo
        
    Returns:
        Modelo cargado
    """
    # Verificar que el archivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo {model_path} no existe.")
    
    # Cargar modelo
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

@timer_decorator
def train_and_optimize_pipeline(X_train: np.ndarray, y_train: np.ndarray, 
                               X_test: np.ndarray, y_test: np.ndarray,
                               train_base: bool = True,
                               optimize: bool = True,
                               n_iter: int = 10,
                               cv: int = 3,
                               sample_size: float = 0.3) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Pipeline completo de entrenamiento y optimización de modelos.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        train_base: Si se deben entrenar modelos base
        optimize: Si se deben optimizar hiperparámetros
        n_iter: Número de iteraciones para búsqueda aleatoria
        cv: Número de folds para validación cruzada
        sample_size: Fracción de datos a usar para optimización
        
    Returns:
        Diccionarios con mejores modelos y métricas
    """
    # Entrenar modelos base
    if train_base:
        print("\n" + "="*50)
        print("ENTRENANDO MODELOS BASE")
        print("="*50)
        trained_models, base_metrics = train_all_base_models(X_train, y_train, X_test, y_test)
        
        # Comparar modelos base
        print("\nComparación de modelos base:")
        comparison_df = compare_models(base_metrics)
        print(comparison_df)
    
    # Optimizar hiperparámetros
    if optimize:
        print("\n" + "="*50)
        print("OPTIMIZANDO HIPERPARÁMETROS")
        print("="*50)
        best_models, best_params, best_metrics = optimize_all_models(
            X_train, y_train, X_test, y_test,
            search_method='random', n_iter=n_iter,
            cv=cv, scoring='f1', sample_size=sample_size
        )
        
        # Comparar modelos optimizados
        print("\nComparación de modelos optimizados:")
        comparison_df = compare_models(best_metrics)
        print(comparison_df)
        
        # Guardar mejores modelos
        print("\nGuardando mejores modelos...")
        for name, model in best_models.items():
            model_path = save_model(model, name)
            print(f"  Modelo {name} guardado en {model_path}")
        
        # Guardar métricas y parámetros para informe
        with open(os.path.join(MODELS_DIR, 'best_metrics.pkl'), 'wb') as f:
            pickle.dump(best_metrics, f)
        
        with open(os.path.join(MODELS_DIR, 'best_params.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
        
        # Generar informe
        from src.generate_report import generate_markdown_report, generate_html_report
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Informe Markdown
        md_report_path = os.path.join(REPORTS_DIR, 'model_evaluation_report.md')
        generate_markdown_report(best_metrics, best_params, md_report_path)
        print(f"\nInforme Markdown generado en {md_report_path}")
        
        # Informe HTML
        html_report_path = os.path.join(REPORTS_DIR, 'model_evaluation_report.html')
        generate_html_report(best_metrics, best_params, html_report_path)
        print(f"Informe HTML generado en {html_report_path}")
        
        return best_models, best_metrics
    
    return trained_models, base_metrics

if __name__ == "__main__":
    # Ejemplo de uso
    from sklearn.datasets import make_classification
    
    # Generar datos sintéticos
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, weights=[0.95, 0.05],
        random_state=RANDOM_STATE
    )
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Ejecutar pipeline
    best_models, best_metrics = train_and_optimize_pipeline(
        X_train, y_train, X_test, y_test,
        train_base=True, optimize=True,
        n_iter=5, cv=3, sample_size=0.5
    )
    
    # Imprimir mejores modelos
    print("\nMejores modelos:")
    for name, model in best_models.items():
        print(f"  {name}: {type(model).__name__}")