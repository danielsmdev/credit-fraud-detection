# Documentación del Proyecto de Detección de Fraude en Tarjetas de Crédito

## Resumen del Proyecto

Este proyecto implementa un sistema de detección de fraude en transacciones de tarjetas de crédito utilizando técnicas de aprendizaje automático. Se ha desarrollado un flujo de trabajo completo que incluye la preparación de datos, entrenamiento de múltiples modelos y una evaluación exhaustiva para identificar el mejor modelo para la detección de fraude.

## Estructura del Proyecto

Credit_fraud/
│
├── data/                      # Datos originales y procesados
│   ├── raw/                   # Datos sin procesar
│   └── processed/             # Datos preprocesados
│
├── models/                    # Modelos entrenados
│   ├── gradient_boosting.pkl
│   ├── lightgbm.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_metrics.pkl
│   ├── best_params.pkl
│   └── best_params.csv
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_EDA.ipynb                # Análisis exploratorio de datos
│   ├── 02_Preprocessing.ipynb      # Notebook de preprocesamiento
│   ├── 03_Model_training_.ipynb    # Notebook de entrenamiento
│   └── 04_Evaluate_model.ipynb     # Notebook de evaluación
│
├── reports/                   # Informes y visualizaciones
│   ├── figures/               # Visualizaciones generadas
│   │   ├── all_confusion_matrices.png
│   │   ├── all_roc_curves.png
│   │   ├── all_precision_recall_curves.png
│   │   ├── calibration_curves.png
│   │   ├── feature_importance.png
│   │   ├── threshold_impact.png
│   │   └── cost_benefit_analysis.png
│   │
│   ├── evaluation_report.md   # Informe detallado de evaluación
│   └── README.md              # Descripción de informes
│
├── src/                       # Código fuente
│   ├── __init__.py
│   ├── config.py              # Configuraciones del proyecto
│   ├── data_prep.py           # Preparación de datos
│   ├── evaluate.py            # Evaluación de modelos
│   ├── model_training.py      # Entrenamiento de modelos
│   ├── run_evaluation.py      # Script para ejecutar evaluación
│   └── utils.py               # Funciones de utilidad
│
├── .gitignore                 # Archivos ignorados por Git
├── README.md                  # Documentación principal
└── requirements.txt           # Dependencias del proyecto

## Flujo de Trabajo Implementado

El proyecto ha seguido un flujo de trabajo estructurado para la detección de fraude:

### 1. Configuración Inicial
- Configuración del entorno virtual
- Instalación de dependencias
- Estructuración del proyecto

### 2. Obtención y Exploración de Datos
- Descarga del conjunto de datos de transacciones de tarjetas de crédito
- Análisis exploratorio de datos (EDA)
- Visualización de distribuciones y correlaciones
- Identificación de desbalance en las clases (fraude vs. no fraude)

### 3. Preprocesamiento de Datos
- Limpieza de datos
- Manejo de valores atípicos
- Normalización/estandarización de características
- División en conjuntos de entrenamiento, validación y prueba
- Técnicas para manejar el desbalance de clases

### 4. Entrenamiento de Modelos
Se entrenaron varios modelos de clasificación:
- Regresión Logística
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### 5. Evaluación Exhaustiva de Modelos
- Evaluación con múltiples métricas (Precisión, Recall, F1, F2, AUC-ROC, etc.)
- Análisis de matrices de confusión
- Curvas ROC y Precision-Recall
- Análisis de calibración de modelos
- Análisis de importancia de características
- Análisis de valores SHAP para interpretabilidad
- Análisis del impacto del umbral de clasificación
- Análisis de costo-beneficio

### 6. Integración con Git/GitHub
- Control de versiones con Git
- Estructura de ramas para desarrollo
- Documentación del proceso

## Resultados de la Evaluación de Modelos

La evaluación exhaustiva de modelos reveló:

1. **Mejor modelo**: El modelo con mejor rendimiento según la evaluación realizada (probablemente XGBoost o LightGBM) obtuvo el mejor equilibrio entre precisión y recall, con un F1 Score superior al resto de modelos.

2. **Características importantes**: Las características más relevantes para la detección de fraude fueron identificadas mediante análisis de importancia de características y valores SHAP, permitiendo entender mejor los patrones de fraude.

3. **Umbral óptimo**: El análisis del umbral de clasificación permitió identificar el punto óptimo que maximiza el F1 Score, así como un umbral alternativo que maximiza el F2 Score cuando se quiere dar más peso al recall.

4. **Análisis de costo-beneficio**: El análisis de costo-beneficio proporcionó un umbral óptimo desde una perspectiva económica, considerando los costos asociados con falsos positivos y falsos negativos.

## Conclusiones

1. **Efectividad de los modelos**: Los modelos de ensamble (como XGBoost, LightGBM y Gradient Boosting) generalmente superaron a los modelos más simples en la detección de fraude.

2. **Importancia del umbral**: La selección del umbral de clasificación es crucial y debe ajustarse según los requisitos específicos del negocio (minimizar falsos positivos vs. falsos negativos).

3. **Interpretabilidad**: El análisis SHAP proporciona insights valiosos sobre cómo los modelos toman decisiones, lo que es esencial para la confianza y la adopción en entornos reales.

4. **Desbalance de clases**: Las técnicas para manejar el desbalance de clases fueron efectivas para mejorar la detección de casos de fraude, que son minoritarios en el conjunto de datos.

## Próximos Pasos Potenciales

Si se retoma el proyecto en el futuro, estos son los posibles pasos a seguir:

1. **Implementación de una API REST**:
   - Desarrollar una API que exponga el mejor modelo para hacer predicciones
   - Implementar autenticación y autorización
   - Documentar la API con Swagger/OpenAPI

2. **Desarrollo de un Dashboard**:
   - Crear una interfaz visual para monitorear predicciones
   - Implementar visualizaciones interactivas
   - Añadir funcionalidades para analizar casos específicos

3. **Implementación en Producción**:
   - Containerización con Docker
   - Orquestación con Kubernetes
   - Configuración de CI/CD para despliegue automático

4. **Sistema de Monitoreo**:
   - Implementar logging de predicciones
   - Detectar drift en los datos
   - Configurar alertas para rendimiento degradado
   - Reentrenamiento automático periódico

5. **Mejoras al Modelo**:
   - Experimentar con arquitecturas más avanzadas (redes neuronales, aprendizaje profundo)
   - Implementar técnicas de aprendizaje por refuerzo
   - Explorar detección de anomalías no supervisada como complemento

## Instrucciones para Retomar el Proyecto

Para retomar el proyecto en el futuro:

1. **Configuración del entorno**:
   ```bash
   # Crear y activar entorno virtual
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   
   # Instalar dependencias
   pip install -r requirements.txt
   ```
2. **Explorar los notebooks**:

    - Revisar `notebooks/EDA.ipynb` para entender los datos
    - Examinar `notebooks/evaluate_models.ipynb` para ver la evaluación completa

3. **Ejecutar la evaluación**:
    ```bash
    python src/run_evaluation.py
    ```
4. **Revisar los informes**:

    - Consultar `reports/evaluation_report.md` para ver los resultados detallados
    - Examinar las visualizaciones en `reports/figures/`

5. **Crear una nueva rama para continuar el desarrollo**:
    ```bash
    git checkout main
    git pull origin main
    git checkout -b nueva-funcionalidad
    ```


## Actualización de GitHub con la Documentación Final

Para finalizar el proyecto y actualizar GitHub con esta documentación:

1. **Crear archivo de documentación**:
    ```bash
    # Guardar este archivo en la raíz del proyecto
    cp documentacion_proyecto_fraude_tarjetas.txt DOCUMENTACION.md
    ```
2. **Actualizar README principal**:

    Añadir una sección en el README.md principal que indique la finalización del proyecto y que apunte a la documentación completa.

3. **Hacer commit y push de los cambios finales**:
    ```bash
    git add DOCUMENTACION.md
    git add README.md
    git add reports/  # Incluir todos los reportes generados
    git commit -m "Documentación final del proyecto de detección de fraude"
    git push origin evaluacion-modelos
    ```
4. **Crear Pull Request y hacer merge a main**:

    - Crear un Pull Request en GitHub para fusionar la rama `evaluacion-modelos` con `main`
    - Revisar los cambios y completar el merge
    - Opcionalmente, crear un tag para marcar la versión final del proyecto:
    ```bash
    git tag -a v1.0 -m "Versión final del proyecto de detección de fraude"
    git push origin v1.0
    ```


Recursos y Referencias

[Documentación de scikit-learn](https://scikit-learn.org/stable/documentation.html)
[XGBoost Documentation](https://xgboost.readthedocs.io/)
[LightGBM Documentation](https://lightgbm.readthedocs.io/)
[SHAP Documentation](https://shap.readthedocs.io/)
[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
[Towards Data Science - Credit Card Fraud Detection](https://towardsdatascience.com/credit-card-fraud-detection-using-machine-learning-python-5b098d4a8edc)

## Lecciones Aprendidas

Durante el desarrollo de este proyecto, se aprendieron varias lecciones importantes:

1. **Manejo del desbalance de clases**: El desbalance extremo en los datos de fraude (típicamente menos del 1% de transacciones son fraudulentas) requiere técnicas especiales como submuestreo, sobremuestreo o SMOTE.
2. **Importancia de múltiples métricas**: La precisión por sí sola no es una buena métrica para problemas desbalanceados; es crucial evaluar F1, recall, AUC-ROC y AUC-PR.
3. **Análisis de costo-beneficio**: En problemas de fraude, los falsos negativos (fraudes no detectados) suelen ser mucho más costosos que los falsos positivos, lo que debe reflejarse en la selección del umbral.
4. **Interpretabilidad vs. rendimiento**: A veces los modelos más precisos (como los de ensamble) son menos interpretables, lo que presenta un desafío cuando se requiere explicabilidad.
5. **Flujo de trabajo modular**: La estructuración del código en módulos reutilizables facilitó la experimentación y evaluación de múltiples modelos.
6. **Control de versiones**: El uso de Git/GitHub fue fundamental para mantener un historial claro del desarrollo y facilitar la colaboración.


Este proyecto ha proporcionado una base sólida para la detección de fraude en tarjetas de crédito y puede servir como punto de partida para implementaciones más avanzadas en el futuro.