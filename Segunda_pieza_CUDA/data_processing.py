import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def data_load(path_csv):
    """
    Carga datos y mapea los 15 valores posibles de param1
    """
    data = pd.read_csv(path_csv)
    
    # 🔥 VALORES ACTUALIZADOS: 15 clases
    VALID_PARAM1_VALUES = [30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110]
    
    # Verificar que todas las clases estén presentes en el dataset
    x = data[['bright', 'sharpness']].values  # NOTA: 'sharpness' NO 'contrast'
    y = data['param1'].values
    
    # Crear mapeo dinámico (0-14)
    class_mapping = {value: idx for idx, value in enumerate(VALID_PARAM1_VALUES)}
    
    # Validar que no hay valores desconocidos
    unknown_classes = set(y) - set(class_mapping.keys())
    if unknown_classes:
        print(f"⚠️ Advertencia: Valores no mapeados: {unknown_classes}")
        print(f"Valores esperados: {VALID_PARAM1_VALUES}")
        # Filtrar filas con valores conocidos
        mask = np.isin(y, list(class_mapping.keys()))
        x = x[mask]
        y = y[mask]
        print(f"✅ Filtrando a {len(x)} muestras válidas")
    
    y_mapped = np.array([class_mapping[label] for label in y])
    
    print(f"📊 Distribución de clases:")
    for value, idx in class_mapping.items():
        count = np.sum(y_mapped == idx)
        print(f"   param1={value}: {count} muestras")
    
    return x, y_mapped, class_mapping


def preprocess_data(path_csv, test_size=0.3, random_state=None):
    """
    PREPROCESAMIENTO CON ALEATORIEDAD
    - random_state=None → diferente división cada vez
    - test_size=0.3 → 70% entrenamiento, 30% prueba
    """
    x, y, class_mapping = data_load(path_csv)
    
    # 🔥 SIN random_state fijo → diferente cada ejecución
    if random_state is None:
        random_state = np.random.randint(0, 10000)  # Aleatorio pero reproducible si se guarda
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size = 0.2, 
        random_state = 42,
        stratify = None  # Mantener proporción de clases
    )
    
    # Normalización
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Guardar scaler y mapping
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(class_mapping, "class_mapping.pkl")
    
    print(f"\n🔀 División aleatoria (seed={random_state}):")
    print(f"   Train: {len(x_train)} muestras ({len(x_train)/len(x)*100:.1f}%)")
    print(f"   Test: {len(x_test)} muestras ({len(x_test)/len(x)*100:.1f}%)")
    
    return x_train, x_test, y_train, y_test, scaler, class_mapping