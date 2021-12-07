import numpy as np
import torch
from torch import nn, optim
from torch.nn.modules.linear import Linear
import torchvision.transforms as T
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torchvision import models

def get_resnet18_encoder(dim=512):
    return nn.Sequential(T.Resize([256, ]), models.resnet18(pretrained=False,num_classes=dim))

def get_shufflenetv2_encoder(dim=512):
    return nn.Sequential(T.Resize([256, ]), models.shufflenet_v2_x1_0(pretrained=False, num_classes=dim))

def get_alexnet_encoder(dim=512):
    return nn.Sequential(T.Resize([256, ]), models.alexnet(pretrained=False, num_classes=dim))

class SpectrumCNN(nn.Module):
    """
    Based on:
    ===
    Ahmed Selim, Francisco Paisana, Jerome A. Arokkiam, Yi Zhang, Linda Doyle, Luiz A. DaSilva,
    "Spectrum Monitoring for Radar Bands using Deep Convolutional Neural Networks",
    GLOBECOM, 2017 arXiv: arXiv:1705.00462v1
    ===
    """
    def __init__(self, dropout=0.5, fin_dim=512):
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3),
            nn.ReLU(),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3),
            nn.ReLU(),
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*5*5, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout), 
            nn.Linear(1024, fin_dim)
        )
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.fc(x)
        return x

