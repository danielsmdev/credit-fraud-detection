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
    EDA.ipynb               # Análisis exploratorio de datos
    preprocessing.ipynb    # Preprocesamiento de datos

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

## Flujo de trabajo con Git

Este proyecto sigue un flujo de trabajo basado en ramas:

- `main`: Rama principal que contiene código estable y listo para producción
- `desarrollo`: Rama de desarrollo donde se integran nuevas características

Para contribuir:

1. Crea una nueva rama desde `desarrollo`
2. Implementa tus cambios
3. Crea un pull request a `desarrollo`

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

