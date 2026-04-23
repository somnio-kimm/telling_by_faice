"""Model architectures for emotion classification."""

from torch import nn
from torchvision import models


class EmotionClassifier(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.features = base_model.features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        return self.classifier(x)
