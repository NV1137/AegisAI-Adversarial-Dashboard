import torch
import torchvision.models as models

def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model