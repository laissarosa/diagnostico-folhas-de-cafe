import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def build_model(pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def set_trainable(m, phase):
    """phase: 'head' (só fc) ou 'finetune' (layer3 + layer4 + fc)."""
    for p in m.parameters():
        p.requires_grad = False
    for p in m.fc.parameters():
        p.requires_grad = True
    if phase == "finetune":
        for blk in (m.layer3, m.layer4):
            for p in blk.parameters():
                p.requires_grad = True


def set_train_mode(m, phase):
    """BatchNorm das camadas congeladas fica em modo de avaliação."""
    m.eval()
    m.fc.train()
    if phase == "finetune":
        m.layer3.train()
        m.layer4.train()