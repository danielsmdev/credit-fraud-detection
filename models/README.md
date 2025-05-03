# Modelos Entrenados

Esta carpeta contiene los modelos entrenados para la detección de fraude en tarjetas de crédito.

## Archivos de modelos

Los archivos de modelos (.joblib, .pkl) no se incluyen en el repositorio de Git debido a su tamaño.
Para generar estos archivos, ejecute el notebook de entrenamiento:

\\\ash
jupyter notebook notebooks/model_training.ipynb
\\\

## Archivos incluidos en Git

- est_params.csv: Parámetros óptimos para cada modelo
- README.md: Este archivo

## Modelos disponibles

- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Logistic Regression
- SVM
