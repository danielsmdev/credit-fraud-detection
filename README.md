# Detección de Fraude en Tarjetas de Crédito

Este proyecto implementa un sistema de detección de fraude en transacciones de tarjetas de crédito utilizando técnicas de aprendizaje automático.

## Estructura del Proyecto

\\\
 data/               # Directorio para datos
    raw/            # Datos sin procesar
    interim/        # Datos intermedios
    processed/      # Datos procesados listos para modelado
 models/             # Modelos entrenados
 notebooks/          # Jupyter notebooks
    EDA.ipynb       # Análisis exploratorio de datos
 reports/            # Informes generados
    figures/        # Gráficas generadas
 src/                # Código fuente
    config.py       # Configuración del proyecto
    data_prep.py    # Preprocesamiento de datos
    evaluate.py     # Evaluación de modelos
    generate_report.py # Generación de informes
    model_training.py # Entrenamiento de modelos
    utils.py        # Utilidades
    visualization_helpers.py # Ayudantes para visualización
 .gitignore          # Archivos a ignorar por Git
 LICENSE             # Licencia del proyecto
 README.md           # Este archivo
 requirements.txt    # Dependencias del proyecto
\\\

## Instalación

1. Clonar el repositorio:
\\\ash
git clone https://github.com/TU_USUARIO/credit-fraud-detection.git
cd credit-fraud-detection
\\\

2. Crear un entorno virtual:
\\\ash
python -m venv venv
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Linux/Mac
\\\

3. Instalar dependencias:
\\\ash
pip install -r requirements.txt
\\\

4. Descargar el dataset:
   - Ve a https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Descarga el archivo creditcard.csv
   - Colócalo en la carpeta \data/raw/\

## Análisis Exploratorio de Datos

El notebook \
otebooks/EDA.ipynb\ contiene un análisis detallado del dataset, incluyendo:

- Distribución de clases (fraude vs no fraude)
- Análisis de variables numéricas
- Detección y análisis de outliers
- Análisis de correlaciones
- Análisis temporal de transacciones
- Comparación entre transacciones fraudulentas y legítimas

Para ejecutar el notebook:
\\\ash
jupyter notebook notebooks/EDA.ipynb
\\\

## Flujo de trabajo con Git

Este proyecto sigue un flujo de trabajo basado en ramas:

- \main\: Rama principal que contiene código estable y listo para producción
- \desarrollo\: Rama de desarrollo donde se integran nuevas características

Para contribuir:
1. Crea una nueva rama desde \desarrollo\
2. Implementa tus cambios
3. Crea un pull request a \desarrollo\

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
