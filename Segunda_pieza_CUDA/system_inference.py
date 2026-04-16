import cv2
import numpy as np
from pathlib import Path
from inference import load_trained_model, predict_parameters
from drops import imgAdjustment, dropsDetection

# ==================== CONFIGURACIÓN ====================
INPUT_FOLDER = "IMG/Entrenamiento"
IMAGE_SIZE = (900, 600)

# Parámetros fijos de Hough
minDist = 20
param2 = 15
minRadius = 10
maxRadius = 20
alpha = -3
beta = 80  # Ajuste de brillo

def extract_features(img_resize):
    """
    Extrae brillo y nitidez (sharpness) del canal verde
    """
    # Ajustar imagen
    adjusted = cv2.convertScaleAbs(img_resize, alpha=alpha, beta=beta)
    _, g, _ = cv2.split(adjusted)
    
    # Features CORRECTAS (coinciden con entrenamiento)
    bright_g = np.mean(g)  # Brillo promedio
    sharpness_g = np.var(cv2.Laplacian(g, cv2.CV_64F))  # Nitidez (varianza)
    
    return bright_g, sharpness_g

def run_system():
    """
    Sistema principal de inferencia
    """
    input_path = Path(INPUT_FOLDER)
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    if not image_files:
        print("❌ No se encontraron imágenes.")
        return
    
    # Cargar modelo (2 features, 15 clases)
    input_size = 2
    num_classes = 15
    model, idx_to_param1 = load_trained_model(input_size, num_classes)
    
    print(f"\n🔍 Procesando {len(image_files)} imágenes...")
    print("="*60)
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"⚠️ No se pudo cargar {img_path.name}")
            continue
        
        img_resize = cv2.resize(img, IMAGE_SIZE)
        
        # 1️⃣ Extraer características
        bright_g, sharpness_g = extract_features(img_resize)
        
        # 2️⃣ Predicción IA
        param1_pred, confidence = predict_parameters(model, [bright_g, sharpness_g], idx_to_param1)
        
        print(f"\n📷 {img_path.name}")
        print(f"   Características: Brillo={bright_g:.1f}, Nitidez={sharpness_g:.1f}")
        print(f"   🎯 param1 predicho: {param1_pred} (confianza: {confidence*100:.1f}%)")
        
        # 3️⃣ Aplicar Hough Circles con param1 predicho
        img_clean = imgAdjustment(alpha, beta, img_resize)
        
        total_objects, objects_data = dropsDetection(
            cleanImg=img_clean,
            minDist=minDist,
            param1=param1_pred,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
            imgResize=img_resize
        )
        
        print(f"   💧 Objetos detectados: {total_objects}")
        
        # Opcional: mostrar detección
        if total_objects > 0:
            from drops_back import draw_circles
            img_result = draw_circles(img_resize, objects_data)
            cv2.imshow(f"Detección - {img_path.name}", img_result)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    print("\n✅ Proceso finalizado.")

if __name__ == "__main__":
    run_system()