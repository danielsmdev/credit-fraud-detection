"""
Funciones para la preparación y procesamiento de datos en el proyecto de detección de fraude.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union

from src.config import (
    RAW_DATA_FILE, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE,
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, SCALER_FILE,
    OUTLIER_THRESHOLD
)
from src.utils import timer_decorator

@timer_decorator
def load_data(file_path: str = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados
    """
    print(f"Cargando datos desde {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    df = pd.read_csv(file_path)
    print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas.")
    
    return df

@timer_decorator
def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifica la calidad de los datos.
    
    Args:
        df: DataFrame a verificar
        
    Returns:
        Diccionario con información sobre la calidad de los datos
    """
    print("Verificando calidad de los datos...")
    
    # Verificar valores nulos
    null_values = df.isnull().sum().sum()
    null_columns = df.columns[df.isnull().any()].tolist()
    
    # Verificar valores duplicados
    duplicate_rows = df.duplicated().sum()
    
    # Verificar valores infinitos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_values = np.isinf(df[numeric_cols]).sum().sum()
    
    # Verificar valores extremos (outliers)
    outliers_info = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - OUTLIER_THRESHOLD * iqr
        upper_bound = q3 + OUTLIER_THRESHOLD * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outliers_pct = outliers / df.shape[0] * 100
        outliers_info[col] = {
            'count': outliers,
            'percentage': outliers_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    # Verificar distribución de clases
    if 'Class' in df.columns:
        class_distribution = df['Class'].value_counts(normalize=True) * 100
        class_imbalance = abs(class_distribution[0] - class_distribution[1])
    else:
        class_distribution = None
        class_imbalance = None
    
    # Crear informe de calidad
    quality_report = {
        'null_values': null_values,
        'null_columns': null_columns,
        'duplicate_rows': duplicate_rows,
        'infinite_values': infinite_values,
        'outliers_info': outliers_info,
        'class_distribution': class_distribution,
        'class_imbalance': class_imbalance
    }
    
    # Imprimir resumen
    print(f"Valores nulos: {null_values}")
    print(f"Columnas con valores nulos: {null_columns}")
    print(f"Filas duplicadas: {duplicate_rows}")
    print(f"Valores infinitos: {infinite_values}")
    
    if class_distribution is not None:
        print("Distribución de clases:")
        for cls, pct in class_distribution.items():
            print(f"  Clase {cls}: {pct:.2f}%")
    
    return quality_report

@timer_decorator
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja los valores faltantes en el DataFrame.
    
    Args:
        df: DataFrame con valores faltantes
        
    Returns:
        DataFrame con valores faltantes tratados
    """
    print("Manejando valores faltantes...")
    
    # Crear copia para no modificar el original
    df_clean = df.copy()
    
    # Identificar columnas con valores nulos
    null_columns = df.columns[df.isnull().any()].tolist()
    
    if not null_columns:
        print("No hay valores nulos que manejar.")
        return df_clean
    
    # Manejar valores nulos por tipo de columna
    for col in null_columns:
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        
        print(f"Columna {col}: {null_count} valores nulos ({null_pct:.2f}%)")
        
        # Si es numérica, imputar con mediana
        if pd.api.types.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            df_clean[col].fillna(median_value, inplace=True)
            print(f"  Imputado con mediana: {median_value}")
        
        # Si es categórica, imputar con moda
        else:
            mode_value = df[col].mode()[0]
            df_clean[col].fillna(mode_value, inplace=True)
            print(f"  Imputado con moda: {mode_value}")
    
    # Verificar valores infinitos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_mask = np.isinf(df_clean[col])
        inf_count = inf_mask.sum()
        
        if inf_count > 0:
            # Reemplazar infinitos con NaN y luego con mediana
            df_clean[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
            print(f"Columna {col}: {inf_count} valores infinitos reemplazados con mediana: {median_value}")
    
    return df_clean

@timer_decorator
def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma características para mejorar el modelado.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame con características transformadas
    """
    print("Transformando características...")
    
    # Crear copia para no modificar el original
    df_transformed = df.copy()
    
    # Transformar Amount con logaritmo (para normalizar distribución)
    if 'Amount' in df.columns:
        df_transformed['Amount_Log'] = np.log1p(df['Amount'])
        print("Transformada Amount con logaritmo (Amount_Log)")
    
    # Transformar Time a características más informativas
    if 'Time' in df.columns:
        # Convertir Time a horas del día (asumiendo que Time está en segundos desde el inicio del día)
        df_transformed['Hour'] = (df['Time'] / 3600) % 24
        print("Transformada Time a hora del día (Hour)")
        
        # Crear variable categórica para período del día
        df_transformed['Day_Period'] = pd.cut(
            df_transformed['Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'],
            include_lowest=True
        )
        print("Creada variable categórica para período del día (Day_Period)")
    
    # Crear características de interacción entre variables principales
    if all(col in df.columns for col in ['V1', 'V2', 'V3']):
        # Ejemplo: crear algunas interacciones entre las variables principales
        df_transformed['V1_V2'] = df['V1'] * df['V2']
        df_transformed['V1_V3'] = df['V1'] * df['V3']
        df_transformed['V2_V3'] = df['V2'] * df['V3']
        print("Creadas características de interacción (V1_V2, V1_V3, V2_V3)")
    
    # Crear características polinómicas para variables importantes
    if 'V14' in df.columns and 'V17' in df.columns:
        # V14 y V17 suelen ser importantes para detección de fraude
        df_transformed['V14_Sq'] = df['V14'] ** 2
        df_transformed['V17_Sq'] = df['V17'] ** 2
        print("Creadas características polinómicas (V14_Sq, V17_Sq)")
    
    # Crear indicadores de anomalías basados en distancias
    if all(col in df.columns for col in ['V1', 'V2', 'V3', 'V4', 'V5']):
        # Calcular distancia euclidiana desde el origen en el espacio de las primeras 5 componentes
        df_transformed['PCA_Dist'] = np.sqrt(
            df['V1']**2 + df['V2']**2 + df['V3']**2 + df['V4']**2 + df['V5']**2
        )
        print("Creada característica de distancia en espacio PCA (PCA_Dist)")
    
    # Crear variables dummy para variables categóricas
    if 'Day_Period' in df_transformed.columns:
        day_period_dummies = pd.get_dummies(df_transformed['Day_Period'], prefix='Period')
        df_transformed = pd.concat([df_transformed, day_period_dummies], axis=1)
        print("Creadas variables dummy para Day_Period")
    
    # Mostrar nuevas características
    new_features = set(df_transformed.columns) - set(df.columns)
    print(f"Total de nuevas características creadas: {len(new_features)}")
    print(f"Nuevas características: {new_features}")
    
    return df_transformed

@timer_decorator
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja los outliers en el DataFrame.
    
    Args:
        df: DataFrame con outliers
        
    Returns:
        DataFrame con outliers tratados
    """
    print("Manejando outliers...")
    
    # Crear copia para no modificar el original
    df_no_outliers = df.copy()
    
    # Identificar columnas numéricas (excluyendo la variable objetivo)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Class' in numeric_cols:
        numeric_cols.remove('Class')
    
    # Manejar outliers por columna
    for col in numeric_cols:
        # Calcular límites IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - OUTLIER_THRESHOLD * iqr
        upper_bound = q3 + OUTLIER_THRESHOLD * iqr
        
        # Identificar outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_count = len(outliers)
        outliers_pct = outliers_count / len(df) * 100
        
        # Solo tratar outliers si son menos del 5% de los datos
        if outliers_pct < 5 and outliers_count > 0:
            print(f"Columna {col}: {outliers_count} outliers ({outliers_pct:.2f}%)")
            
            # Aplicar capping (recorte)
            df_no_outliers.loc[df_no_outliers[col] < lower_bound, col] = lower_bound
            df_no_outliers.loc[df_no_outliers[col] > upper_bound, col] = upper_bound
            print(f"  Aplicado recorte con límites: [{lower_bound:.2f}, {upper_bound:.2f}]")
        elif outliers_count > 0:
            print(f"Columna {col}: {outliers_count} outliers ({outliers_pct:.2f}%) - No tratados por ser ≥ 5%")
    
    return df_no_outliers

@timer_decorator
def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, 
               random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        df: DataFrame a dividir
        test_size: Proporción del conjunto de prueba
        random_state: Semilla aleatoria
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Dividiendo datos en conjuntos de entrenamiento ({1-test_size:.0%}) y prueba ({test_size:.0%})...")
    
    # Separar características y variable objetivo
    if 'Class' in df.columns:
        X = df.drop('Class', axis=1)
        y = df['Class']
    else:
        raise ValueError("La columna 'Class' no está presente en el DataFrame.")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"X_train: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
    print(f"X_test: {X_test.shape[0]} muestras, {X_test.shape[1]} características")
    
    # Verificar distribución de clases
    train_class_dist = pd.Series(y_train).value_counts(normalize=True) * 100
    test_class_dist = pd.Series(y_test).value_counts(normalize=True) * 100
    
    print("Distribución de clases en conjunto de entrenamiento:")
    for cls, pct in train_class_dist.items():
        print(f"  Clase {cls}: {pct:.2f}%")
    
    print("Distribución de clases en conjunto de prueba:")
    for cls, pct in test_class_dist.items():
        print(f"  Clase {cls}: {pct:.2f}%")
    
    return X_train, X_test, y_train, y_test

@timer_decorator
def balance_classes(X_train: pd.DataFrame, y_train: pd.Series, 
                   random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balancea las clases en el conjunto de entrenamiento usando SMOTE.
    Maneja automáticamente columnas categóricas.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        random_state: Semilla aleatoria
        
    Returns:
        X_train_balanced, y_train_balanced
    """
    print("Balanceando clases con SMOTE...")
    
    # Verificar distribución original
    class_counts = pd.Series(y_train).value_counts()
    class_pcts = pd.Series(y_train).value_counts(normalize=True) * 100
    
    print("Distribución original:")
    for cls, count in class_counts.items():
        print(f"  Clase {cls}: {count} ({class_pcts[cls]:.2f}%)")
    
    # Asegurarse de que X_train sea un DataFrame
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    
    # Identificar columnas categóricas
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        print(f"Detectadas {len(categorical_cols)} columnas categóricas: {categorical_cols}")
        print("Aplicando one-hot encoding a columnas categóricas antes de SMOTE...")
        
        # Guardar columnas categóricas para restaurarlas después
        categorical_data = X_train[categorical_cols].copy()
        
        # Aplicar one-hot encoding
        X_train_numeric = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
        
        # Aplicar SMOTE a los datos numéricos
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train_numeric, y_train)
        
        print("SMOTE aplicado exitosamente a datos numéricos.")
    else:
        # Si no hay columnas categóricas, aplicar SMOTE directamente
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Verificar distribución después de SMOTE
    balanced_counts = pd.Series(y_resampled).value_counts()
    balanced_pcts = pd.Series(y_resampled).value_counts(normalize=True) * 100
    
    print("Distribución después de SMOTE:")
    for cls, count in balanced_counts.items():
        print(f"  Clase {cls}: {count} ({balanced_pcts[cls]:.2f}%)")
    
    return X_resampled, y_resampled

@timer_decorator
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Escala las características para normalizar su distribución.
    Maneja automáticamente la discrepancia de columnas entre conjuntos de datos.
    
    Args:
        X_train: Características de entrenamiento
        X_test: Características de prueba
        scaler_type: Tipo de escalador ('standard' o 'robust')
        
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    print(f"Escalando características con {scaler_type} scaler...")
    
    # Convertir a DataFrame si son arrays
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # Identificar columnas en cada conjunto
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    # Verificar si hay discrepancia de columnas
    if train_cols != test_cols:
        print(f"Detectada discrepancia de columnas entre conjuntos de entrenamiento y prueba.")
        print(f"Columnas solo en entrenamiento: {train_cols - test_cols}")
        print(f"Columnas solo en prueba: {test_cols - train_cols}")
        
        # Encontrar columnas comunes
        common_cols = list(train_cols.intersection(test_cols))
        print(f"Usando solo las {len(common_cols)} columnas comunes para escalar.")
        
        # Usar solo columnas comunes
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
    
    # Seleccionar tipo de escalador
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    elif scaler_type.lower() == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Tipo de escalador no válido. Use 'standard' o 'robust'.")
    
    # Ajustar escalador a datos de entrenamiento y transformar
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transformar datos de prueba
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Características escaladas. Rango típico: [{-3:.1f}, {3:.1f}]")
    
    return X_train_scaled, X_test_scaled, scaler

@timer_decorator
def save_processed_data(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray, 
                       scaler: Any, feature_names: List[str]) -> None:
    """
    Guarda los datos procesados para su uso posterior.
    
    Args:
        X_train_scaled: Características de entrenamiento escaladas
        X_test_scaled: Características de prueba escaladas
        y_train: Variable objetivo de entrenamiento
        y_test: Variable objetivo de prueba
        scaler: Escalador ajustado
        feature_names: Nombres de las características
    """
    print("Guardando datos procesados...")
    
    # Crear directorio si no existe
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Convertir arrays a DataFrames para guardar con nombres de columnas
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    y_train_df = pd.DataFrame(y_train, columns=['Class'])
    y_test_df = pd.DataFrame(y_test, columns=['Class'])
    
    # Guardar DataFrames
    X_train_df.to_csv(X_TRAIN_FILE, index=False)
    X_test_df.to_csv(X_TEST_FILE, index=False)
    y_train_df.to_csv(Y_TRAIN_FILE, index=False)
    y_test_df.to_csv(Y_TEST_FILE, index=False)
    
    # Guardar escalador
    joblib.dump(scaler, SCALER_FILE)
    
    # Guardar metadatos
    metadata = {
        'X_train_shape': X_train_scaled.shape,
        'X_test_shape': X_test_scaled.shape,
        'y_train_shape': y_train.shape,
        'y_test_shape': y_test.shape,
        'feature_names': feature_names,
        'scaler_type': type(scaler).__name__
    }
    
    metadata_file = os.path.join(PROCESSED_DATA_DIR, 'metadata.txt')
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Datos guardados en {PROCESSED_DATA_DIR}")
    print(f"X_train guardado en {X_TRAIN_FILE}")
    print(f"X_test guardado en {X_TEST_FILE}")
    print(f"y_train guardado en {Y_TRAIN_FILE}")
    print(f"y_test guardado en {Y_TEST_FILE}")
    print(f"Scaler guardado en {SCALER_FILE}")
    print(f"Metadatos guardados en {metadata_file}")

@timer_decorator
def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Carga los datos procesados previamente guardados.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("Cargando datos procesados...")
    
    # Verificar que los archivos existen
    for file_path in [X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe. Ejecute el preprocesamiento primero.")
    
    # Cargar datos
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE)['Class'].values
    y_test = pd.read_csv(Y_TEST_FILE)['Class'].values
    
    # Obtener nombres de características
    feature_names = X_train.columns.tolist()
    
    # Convertir a arrays numpy si es necesario
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    print(f"Datos cargados:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names

def preprocessing_pipeline(raw_data_file: str = RAW_DATA_FILE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Pipeline completo de preprocesamiento de datos.
    
    Args:
        raw_data_file: Ruta al archivo de datos crudos
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("Iniciando pipeline de preprocesamiento...")
    
    # 1. Cargar datos
    df = load_data(raw_data_file)
    
    # 2. Verificar calidad de datos
    quality_report = check_data_quality(df)
    
    # 3. Manejar valores faltantes
    df_clean = handle_missing_values(df)
    
    # 4. Transformar características
    df_transformed = transform_features(df_clean)
    
    # 5. Manejar outliers
    df_no_outliers = handle_outliers(df_transformed)
    
    # 6. Dividir datos
    X_train, X_test, y_train, y_test = split_data(df_no_outliers)
    
    # 7. Balancear clases
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
    
    # 8. Escalar características
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_balanced, X_test)
    
    # 9. Guardar datos procesados
    save_processed_data(X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler, X_train.columns)
    
    print("Pipeline de preprocesamiento completado.")
    
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, X_train.columns.tolist()

if __name__ == "__main__":
    # Ejecutar pipeline completo
    X_train, X_test, y_train, y_test, feature_names = preprocessing_pipeline()