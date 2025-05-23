# Detección de Fraude en Tarjetas de Crédito

Este proyecto implementa un sistema de detección de fraude en transacciones de tarjetas de crédito utilizando técnicas de aprendizaje automático.

## Estructura del Proyecto

```
data/               # Directorio para datos
    raw/            # Datos sin procesar
    interim/        # Datos intermedios
    processed/      # Datos procesados listos para modelado

models/             # Modelos entrenados

notebooks/          # Jupyter notebooks
    01_EDA.ipynb               # Análisis exploratorio de datos
    02_Preprocessing.ipynb    # Preprocesamiento de datos

reports/            # Informes generados
    figures/        # Gráficas generadas

src/                # Código fuente
    config.py
    data_prep.py
    evaluate.py
    generate_report.py
    model_training.py
    run_preprocessing.py
    utils.py
    visualization_helpers.py

.gitignore
LICENSE
README.md
requirements.txt
```

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/TU_USUARIO/credit-fraud-detection.git
cd credit-fraud-detection
```

2. Crear un entorno virtual:

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux/Mac
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Descargar el dataset:
   - Ve a [Kaggle: Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Descarga el archivo `creditcard.csv`
   - Colócalo en la carpeta `data/raw/`

## Análisis Exploratorio de Datos

El notebook `notebooks/EDA.ipynb` contiene un análisis detallado del dataset, incluyendo:

- Distribución de clases (fraude vs no fraude)
- Análisis de variables numéricas
- Detección y análisis de outliers
- Análisis de correlaciones
- Análisis temporal de transacciones
- Comparación entre transacciones fraudulentas y legítimas

Para ejecutar el notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Preprocesamiento de Datos

El módulo `src/data_prep.py` contiene funciones para:

- Carga y verificación de calidad de datos
- Manejo de valores faltantes y outliers
- Transformación y creación de características
- División en conjuntos de entrenamiento y prueba
- Balanceo de clases con SMOTE
- Escalado de características

El notebook `notebooks/preprocessing.ipynb` muestra el proceso completo de preprocesamiento con visualizaciones.

Para ejecutar el preprocesamiento desde la línea de comandos:

```bash
python src/run_preprocessing.py
```


## Entrenamiento de Modelos

El notebook 
otebooks/model_training.ipynb entrena y optimiza varios modelos de machine learning para la detección de fraude, incluyendo:

- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Logistic Regression
- SVM

El proceso incluye:
1. Entrenamiento de modelos base
2. Comparación de rendimiento
3. Optimización de hiperparámetros
4. Análisis de características importantes
5. Visualizaciones avanzadas
6. Generación de informes

Para ejecutar el entrenamiento:
```bash
jupyter notebook notebooks/model_training.ipynb
```

Los modelos entrenados se guardan en la carpeta models/ (no incluidos en Git debido a su tamaño).
Las visualizaciones y reportes se guardan en la carpeta 
eports/.

## Flujo de trabajo con Git

Este proyecto sigue un flujo de trabajo basado en ramas:

- `main`: Rama principal que contiene código estable y listo para producción
- `desarrollo`: Rama de desarrollo donde se integran nuevas características

Para contribuir:

1. Crea una nueva rama desde `desarrollo`
2. Implementa tus cambios
3. Crea un pull request a `desarrollo`

## Documentación completa

Para una documentación detallada del proyecto, consulta el archivo [DOCUMENTACION.md](./DOCUMENTACION.md) que incluye:
- Descripción completa del flujo de trabajo
- Resultados de la evaluación de modelos
- Conclusiones y lecciones aprendidas
- Instrucciones para retomar el proyecto
- Próximos pasos potenciales

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.