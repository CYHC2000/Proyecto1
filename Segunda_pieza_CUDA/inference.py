import torch
import joblib
import numpy as np
from model import MulticlassModel

def load_trained_model(input_size, num_classes):
    """
    Carga modelo entrenado y sus dependencias
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MulticlassModel(input_size, num_classes)
    
    # Cargar checkpoint completo
    checkpoint = torch.load("modelo_entrenado.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # Cargar mapping de clases
    class_mapping = joblib.load("class_mapping.pkl")
    # Invertir mapping: índice -> valor param1
    idx_to_param1 = {v: k for k, v in class_mapping.items()}
    
    print(f"✅ Modelo cargado (Accuracy entrenamiento: {checkpoint['accuracy']*100:.2f}%)")
    print(f"📋 Clases soportadas: {sorted(class_mapping.keys())}")
    
    return model, idx_to_param1

def predict_parameters(model, features, idx_to_param1):
    """
    Predice param1 a partir de features [bright, sharpness]
    """
    device = next(model.parameters()).device
    
    scaler = joblib.load("scaler.pkl")
    
    # Asegurar formato correcto
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    
    predicted_param1 = idx_to_param1[predicted_class.item()]
    
    return predicted_param1, confidence.item()