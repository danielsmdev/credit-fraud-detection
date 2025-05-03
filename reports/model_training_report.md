# Informe de Evaluación de Modelos de Detección de Fraude

## Resumen de Métricas

| Modelo | Accuracy | Precision | Recall | F1 | F2 | AUC | AP |
|--------|----------|-----------|--------|----|----|-----|----|
| logistic_regression | 0.9901 | 0.1295 | 0.8632 | 0.2253 | 0.4047 | 0.9582 | 0.6726 |
| random_forest | 0.9995 | 0.9342 | 0.7474 | 0.8304 | 0.7785 | 0.9649 | 0.8127 |
| gradient_boosting | 0.9994 | 0.8571 | 0.7579 | 0.8045 | 0.7759 | 0.9420 | 0.7742 |
| xgboost | 0.9995 | 0.8824 | 0.7895 | 0.8333 | 0.8065 | 0.9648 | 0.8207 |
| lightgbm | 0.9995 | 0.9012 | 0.7684 | 0.8295 | 0.7918 | 0.9724 | 0.7565 |

## Mejores Hiperparámetros

### logistic_regression

```
solver: liblinear
penalty: l1
C: 1.0
```

### random_forest

```
n_estimators: 200
min_samples_split: 2
min_samples_leaf: 1
max_depth: None
```

### gradient_boosting

```
subsample: 1.0
n_estimators: 200
max_depth: 5
learning_rate: 0.1
```

### xgboost

```
subsample: 0.8
scale_pos_weight: 1
n_estimators: 200
max_depth: 5
learning_rate: 0.1
colsample_bytree: 0.8
```

### lightgbm

```
subsample: 0.8
num_leaves: 50
n_estimators: 200
max_depth: -1
learning_rate: 0.1
```

## Matrices de Confusión

Ver imagen: `all_confusion_matrices.png`

## Curvas ROC y Precision-Recall

Ver imagen: `model_comparison_dashboard.png`

## Análisis de Costo-Beneficio

Ver imagen: `cost_benefit_analysis.png`

## Conclusiones

El mejor modelo según F1 Score es **xgboost** con:

- F1 Score: 0.8333
- Precision: 0.8824
- Recall: 0.7895
- AUC-ROC: 0.9648

El mejor modelo según AUC-ROC es **lightgbm** con:

- AUC-ROC: 0.9724
- F1 Score: 0.8295
- Precision: 0.9012
- Recall: 0.7684

## Recomendaciones

1. **Selección de modelo**: Basado en los resultados, se recomienda utilizar el modelo **xgboost** para implementación en producción, ya que ofrece el mejor equilibrio entre precisión y recall.

2. **Umbral de clasificación**: Ajustar el umbral de clasificación según el análisis de costo-beneficio para optimizar el rendimiento en el contexto específico de negocio.

3. **Monitoreo**: Implementar un sistema de monitoreo para detectar cambios en el rendimiento del modelo a lo largo del tiempo.

4. **Reentrenamiento**: Establecer un cronograma para reentrenar el modelo periódicamente con datos nuevos.

