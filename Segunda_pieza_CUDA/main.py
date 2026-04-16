import numpy as np
import torch
from data_processing import preprocess_data
from train_model import train_model
from inference import load_trained_model

def main():
    print("="*60)
    print("🎯 SISTEMA DE ENTRENAMIENTO - Predicción de param1")
    print("="*60)
    
    # Verificación CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA no está disponible. Este sistema requiere GPU.")
    
    print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
    
    path_csv = "Drops_v1.csv"
    
    # ================== ENTRENAMIENTO ==================
    print("\n📂 Cargando y preparando datos...")
    x_train, x_test, y_train, y_test, scaler, class_mapping = preprocess_data(
        path_csv, 
        test_size=0.3,  # 70% train, 30% test
        random_state=None  # 🔥 ALEATORIO cada vez
    )
    
    input_size = x_train.shape[1]  # = 2
    num_classes = len(class_mapping)  # = 15
    
    print(f"\n🎯 Configuración del modelo:")
    print(f"   Features: {input_size}")
    print(f"   Clases: {num_classes} ({sorted(class_mapping.keys())})")
    
    # Entrenar modelo
    model = train_model(
        x_train, y_train,
        x_test, y_test,
        input_size, num_classes,
        num_epochs=200,
        batch_size=32,
        learning_rate=0.001
    )
    
    print("\n✨ Entrenamiento completado exitosamente!")
    
    # ================== VERIFICACIÓN FINAL ==================
    print("\n🔍 Verificando modelo guardado...")
    test_model, _ = load_trained_model(input_size, num_classes)
    print("✅ Modelo listo para inferencia")

if __name__ == "__main__":
    main()