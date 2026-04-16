import torch.nn as nn

class MulticlassModel(nn.Module):
    """
    Red neuronal simple para clasificación multiclase (15 clases)
    Input: 2 features (brightness, sharpness)
    Output: 15 clases (valores de param1)
    """
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        # Arquitectura más simple pero efectiva
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)