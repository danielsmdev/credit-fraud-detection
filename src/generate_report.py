"""
Funciones para generar informes de resultados de modelos.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import json

def generate_markdown_report(metrics_dict: Dict[str, Dict[str, Any]], 
                            best_params: Dict[str, Dict[str, Any]],
                            output_path: str):
    """
    Genera un informe resumido en formato Markdown.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        best_params: Diccionario con mejores parámetros por modelo
        output_path: Ruta donde guardar el informe
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Título
        f.write("# Informe de Evaluación de Modelos de Detección de Fraude\n\n")
        
        # Resumen de métricas
        f.write("## Resumen de Métricas\n\n")
        
        # Crear tabla de métricas
        f.write("| Modelo | Accuracy | Precision | Recall | F1 | F2 | AUC | AP |\n")
        f.write("|--------|----------|-----------|--------|----|----|-----|----|\n")
        
        for name, metrics in metrics_dict.items():
            accuracy = metrics.get('accuracy', 'N/A')
            precision = metrics.get('precision', 'N/A')
            recall = metrics.get('recall', 'N/A')
            f1 = metrics.get('f1', 'N/A')
            f2 = metrics.get('f2', 'N/A')
            auc = metrics.get('roc_auc', 'N/A')
            ap = metrics.get('avg_precision', 'N/A')
            
            if isinstance(accuracy, (int, float)):
                f.write(f"| {name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {f2:.4f} | {auc:.4f} | {ap:.4f} |\n")
            else:
                f.write(f"| {name} | {accuracy} | {precision} | {recall} | {f1} | {f2} | {auc} | {ap} |\n")
        
        f.write("\n")
        
        # Mejores parámetros
        f.write("## Mejores Hiperparámetros\n\n")
        
        for name, params in best_params.items():
            f.write(f"### {name}\n\n")
            f.write("```\n")
            for param, value in params.items():
                f.write(f"{param}: {value}\n")
            f.write("```\n\n")
        
        # Matrices de confusión
        f.write("## Matrices de Confusión\n\n")
        f.write("Ver imagen: `all_confusion_matrices.png`\n\n")
        
        # Curvas ROC y Precision-Recall
        f.write("## Curvas ROC y Precision-Recall\n\n")
        f.write("Ver imagen: `model_comparison_dashboard.png`\n\n")
        
        # Análisis de costo-beneficio
        f.write("## Análisis de Costo-Beneficio\n\n")
        f.write("Ver imagen: `cost_benefit_analysis.png`\n\n")
        
        # Conclusiones
        f.write("## Conclusiones\n\n")
        
        # Encontrar el mejor modelo según F1
        best_model = max(metrics_dict.items(), key=lambda x: x[1].get('f1', 0))
        best_model_name = best_model[0]
        best_model_metrics = best_model[1]
        
        f.write(f"El mejor modelo según F1 Score es **{best_model_name}** con:\n\n")
        f.write(f"- F1 Score: {best_model_metrics.get('f1', 'N/A'):.4f}\n")
        f.write(f"- Precision: {best_model_metrics.get('precision', 'N/A'):.4f}\n")
        f.write(f"- Recall: {best_model_metrics.get('recall', 'N/A'):.4f}\n")
        if 'roc_auc' in best_model_metrics:
            f.write(f"- AUC-ROC: {best_model_metrics.get('roc_auc', 'N/A'):.4f}\n")
        
        # Encontrar el mejor modelo según AUC
        auc_models = {name: metrics.get('roc_auc', 0) for name, metrics in metrics_dict.items() if 'roc_auc' in metrics}
        if auc_models:
            best_auc_model_name = max(auc_models.items(), key=lambda x: x[1])[0]
            
            if best_auc_model_name != best_model_name:
                best_auc_model_metrics = metrics_dict[best_auc_model_name]
                f.write(f"\nEl mejor modelo según AUC-ROC es **{best_auc_model_name}** con:\n\n")
                f.write(f"- AUC-ROC: {best_auc_model_metrics.get('roc_auc', 'N/A'):.4f}\n")
                f.write(f"- F1 Score: {best_auc_model_metrics.get('f1', 'N/A'):.4f}\n")
                f.write(f"- Precision: {best_auc_model_metrics.get('precision', 'N/A'):.4f}\n")
                f.write(f"- Recall: {best_auc_model_metrics.get('recall', 'N/A'):.4f}\n")
        
        # Recomendaciones
        f.write("\n## Recomendaciones\n\n")
        f.write("1. **Selección de modelo**: Basado en los resultados, se recomienda utilizar el modelo ")
        if 'roc_auc' in best_model_metrics and best_model_metrics.get('roc_auc', 0) > 0.95:
            f.write(f"**{best_model_name}** para implementación en producción, ya que ofrece el mejor equilibrio entre precisión y recall.\n\n")
        else:
            f.write("con mejor F1 Score para casos donde se requiere equilibrio, o el modelo con mejor AUC para casos donde se necesita una buena discriminación general.\n\n")
        
        f.write("2. **Umbral de clasificación**: Ajustar el umbral de clasificación según el análisis de costo-beneficio para optimizar el rendimiento en el contexto específico de negocio.\n\n")
        
        f.write("3. **Monitoreo**: Implementar un sistema de monitoreo para detectar cambios en el rendimiento del modelo a lo largo del tiempo.\n\n")
        
        f.write("4. **Reentrenamiento**: Establecer un cronograma para reentrenar el modelo periódicamente con datos nuevos.\n\n")

def generate_html_report(metrics_dict: Dict[str, Dict[str, Any]], 
                        best_params: Dict[str, Dict[str, Any]],
                        output_path: str):
    """
    Genera un informe en formato HTML.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
        best_params: Diccionario con mejores parámetros por modelo
        output_path: Ruta donde guardar el informe
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe de Evaluación de Modelos</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 30px; }
            h3 { color: #3498db; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .code { background-color: #f8f8f8; padding: 15px; border-radius: 5px; font-family: monospace; overflow-x: auto; }
            .metric-good { color: #27ae60; font-weight: bold; }
            .metric-medium { color: #f39c12; font-weight: bold; }
            .metric-bad { color: #e74c3c; font-weight: bold; }
            .conclusion { background-color: #eaf2f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Informe de Evaluación de Modelos de Detección de Fraude</h1>
        
        <h2>Resumen de Métricas</h2>
        <table>
            <tr>
                <th>Modelo</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>F2</th>
                <th>AUC</th>
                <th>AP</th>
            </tr>
    """
    
    # Añadir filas de la tabla
    for name, metrics in metrics_dict.items():
        accuracy = metrics.get('accuracy', 'N/A')
        precision = metrics.get('precision', 'N/A')
        recall = metrics.get('recall', 'N/A')
        f1 = metrics.get('f1', 'N/A')
        f2 = metrics.get('f2', 'N/A')
        auc = metrics.get('roc_auc', 'N/A')
        ap = metrics.get('avg_precision', 'N/A')
        
        html += f"""
            <tr>
                <td>{name}</td>
        """
        
        # Añadir métricas con formato condicional
        for metric in [accuracy, precision, recall, f1, f2, auc, ap]:
            if isinstance(metric, (int, float)):
                if metric > 0.9:
                    html += f'<td class="metric-good">{metric:.4f}</td>'
                elif metric > 0.7:
                    html += f'<td class="metric-medium">{metric:.4f}</td>'
                else:
                    html += f'<td class="metric-bad">{metric:.4f}</td>'
            else:
                html += f'<td>{metric}</td>'
        
        html += """
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Mejores Hiperparámetros</h2>
    """
    
    # Añadir mejores parámetros
    for name, params in best_params.items():
        html += f"""
        <h3>{name}</h3>
        <div class="code">
        """
        
        for param, value in params.items():
            html += f"{param}: {value}<br>"
        
        html += """  value in params.items():
            html += f"{param}: {value}<br>"
        <h2>Visualizaciones</h2>
        
        <h3>Matrices de Confusión</h3>
        <img src="all_confusion_matrices.png" alt="Matrices de Confusión">
        
        <h3>Curvas ROC y Precision-Recall</h3>
        <img src="model_comparison_dashboard.png" alt="Dashboard de Comparación de Modelos">
        
        <h3>Análisis de Costo-Beneficio</h3>
        <img src="cost_benefit_analysis.png" alt="Análisis de Costo-Beneficio">
        
        <h2>Conclusiones</h2>
    """
    
    # Añadir conclusiones
    best_model = max(metrics_dict.items(), key=lambda x: x[1].get('f1', 0))
    best_model_name = best_model[0]
    best_model_metrics = best_model[1]
    
    html += f"""
        <div class="conclusion">
            <p>El mejor modelo según F1 Score es <strong>{best_model_name}</strong> con:</p>
            <ul>
                <li>F1 Score: {best_model_metrics.get('f1', 'N/A'):.4f}</li>
                <li>Precision: {best_model_metrics.get('precision', 'N/A'):.4f}</li>
                <li>Recall: {best_model_metrics.get('recall', 'N/A'):.4f}</li>
    """
    
    if 'roc_auc' in best_model_metrics:
        html += f"""
                <li>AUC-ROC: {best_model_metrics.get('roc_auc', 'N/A'):.4f}</li>
        """
    
    html += """
            </ul>
        </div>
    """
    
    # Añadir mejor modelo según AUC si es diferente
    auc_models = {name: metrics.get('roc_auc', 0) for name, metrics in metrics_dict.items() if 'roc_auc' in metrics}
    if auc_models:
        best_auc_model_name = max(auc_models.items(), key=lambda x: x[1])[0]
        
        if best_auc_model_name != best_model_name:
            best_auc_model_metrics = metrics_dict[best_auc_model_name]
            html += f"""
            <div class="conclusion">
                <p>El mejor modelo según AUC-ROC es <strong>{best_auc_model_name}</strong> con:</p>
                <ul>
                    <li>AUC-ROC: {best_auc_model_metrics.get('roc_auc', 'N/A'):.4f}</li>
                    <li>F1 Score: {best_auc_model_metrics.get('f1', 'N/A'):.4f}</li>
                    <li>Precision: {best_auc_model_metrics.get('precision', 'N/A'):.4f}</li>
                    <li>Recall: {best_auc_model_metrics.get('recall', 'N/A'):.4f}</li>
                </ul>
            </div>
            """
    
    # Añadir recomendaciones
    html += """
        <h2>Recomendaciones</h2>
        <ol>
            <li><strong>Selección de modelo</strong>: Basado en los resultados, se recomienda utilizar el modelo con mejor F1 Score para casos donde se requiere equilibrio, o el modelo con mejor AUC para casos donde se necesita una buena discriminación general.</li>
            <li><strong>Umbral de clasificación</strong>: Ajustar el umbral de clasificación según el análisis de costo-beneficio para optimizar el rendimiento en el contexto específico de negocio.</li>
            <li><strong>Monitoreo</strong>: Implementar un sistema de monitoreo para detectar cambios en el rendimiento del modelo a lo largo del tiempo.</li>
            <li><strong>Reentrenamiento</strong>: Establecer un cronograma para reentrenar el modelo periódicamente con datos nuevos.</li>
        </ol>
    </body>
    </html>
    """
    
    # Guardar archivo HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

# Ejemplo de uso
if __name__ == "__main__":
    from src.config import REPORTS_DIR
    import pickle
    
    # Cargar métricas y parámetros (asumiendo que se guardaron previamente)
    try:
        with open('models/best_metrics.pkl', 'rb') as f:
            best_metrics = pickle.load(f)
        
        with open('models/best_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
        
        # Generar informes
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Informe Markdown
        md_report_path = os.path.join(REPORTS_DIR, 'model_evaluation_report.md')
        generate_markdown_report(best_metrics, best_params, md_report_path)
        
        # Informe HTML
        html_report_path = os.path.join(REPORTS_DIR, 'model_evaluation_report.html')
        generate_html_report(best_metrics, best_params, html_report_path)
        
        print(f"Informes generados en {REPORTS_DIR}")
    except Exception as e:
        print(f"Error al generar informes: {str(e)}")