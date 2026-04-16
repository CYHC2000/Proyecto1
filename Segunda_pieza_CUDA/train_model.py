import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from model import MulticlassModel
import time

def train_model(x_train, y_train, x_test, y_test,
                input_size, num_classes,
                num_epochs=200,  # Reducido porque hay más clases
                batch_size=32,
                learning_rate=0.001):  # Learning rate más estándar

    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA no está disponible. Se requiere GPU.")
    
    device = torch.device("cuda")
    print(f"\n🚀 Entrenando en: {torch.cuda.get_device_name(0)}")
    
    # Convertir a tensores CUDA
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    model = MulticlassModel(input_size, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    softmax = nn.Softmax(dim=1)
    
    best_accuracy = 0
    best_epoch = 0
    
    print("\n" + "="*70)
    print(f"{'Época':<6} {'Loss Train':<12} {'Accuracy Test':<15} {'Confianza Prom':<15} {'Mejoría':<10}")
    print("="*70)
    
    for epoch in range(num_epochs):
        
        # ================= ENTRENAMIENTO =================
        model.train()
        running_loss = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        
        # ================= EVALUACIÓN =================
        model.eval()
        y_pred = []
        confidences = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                probs = softmax(outputs)
                max_probs, predicted = torch.max(probs, 1)
                y_pred.extend(predicted.cpu().numpy())
                confidences.extend(max_probs.cpu().numpy())
        
        accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)
        avg_confidence = np.mean(confidences)
        
        # Actualizar learning rate
        scheduler.step(1 - accuracy)
        
        # Mostrar progreso CADA época
        mejoria = "⭐ NUEVO BEST" if accuracy > best_accuracy else ""
        print(f"{epoch+1:<6} {avg_loss:<12.4f} {accuracy*100:<15.2f}% {avg_confidence*100:<15.2f}% {mejoria}")
        
        # Guardar mejor modelo
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': avg_loss
            }, "modelo_entrenado.pth")
    
    print("="*70)
    print(f"\n✅ MEJOR MODELO: Época {best_epoch} | Accuracy: {best_accuracy*100:.2f}%")
    
    # Evaluación final detallada
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        print("\n📊 Reporte de clasificación:")
        print(classification_report(y_test.cpu(), predicted.cpu(), digits=3))
    
    return model