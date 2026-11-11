"""Три подхода к классификации катаракты.

1) baseline  — небольшая CNN, обучаемая "с нуля";
2) efficientnet — трансферное обучение: EfficientNet-B0, предобученная на ImageNet;
3) dinov2 — замороженная самообученная ViT-S/14 (DINOv2) + обучаемая
   attention-голова (механизм внимания над патч-токенами).

Все модели возвращают один логит: P(катаракта) = sigmoid(выход).
"""
import torch
import torch.nn as nn
import timm
from torchvision import models


# --------------------------------------------------------------------------
# Подход 1: базовая CNN "с нуля"
# --------------------------------------------------------------------------
def build_baseline_cnn() -> nn.Module:
    """Простая 5-слойная CNN: блоки Conv-BN-ReLU-MaxPool + GAP + FC."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
    )


# --------------------------------------------------------------------------
# Подход 2: трансферное обучение (EfficientNet-B0, ImageNet)
# --------------------------------------------------------------------------
def build_efficientnet() -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # Заменяем классификатор: 1280 признаков -> 1 логит
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(model.classifier[1].in_features, 1),
    )
    return model


# --------------------------------------------------------------------------
# Подход 3: DINOv2 (самообучение) + attention-голова
# --------------------------------------------------------------------------
class DinoV2AttentionClassifier(nn.Module):
    """Замороженный DINOv2 ViT-S/14 как экстрактор признаков + обучаемая
    single-head self-attention над патч-токенами и MLP-классификатор.

    Идея: на маленьком датасете (~490 train-изображений) дообучение всей ViT
    склонно к переобучению; замороженный самообученный бэкбон даёт устойчивые
    признаки, а attention-голова учится взвешивать информативные патчи сетчатки.
    """

    def __init__(self, dim: int = 384, num_heads: int = 6, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,          # без собственной головы timm
            dynamic_img_size=True,  # допускаем вход 224x224 вместо "родных" 518
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # бэкбон всегда в режиме инференса
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)   # (B, 1+16*16, 384): CLS + патчи
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        pooled = self.norm(attn_out).mean(dim=1)     # среднее по токенам после attention
        return self.head(pooled)


def build_dinov2() -> nn.Module:
    return DinoV2AttentionClassifier()


# --------------------------------------------------------------------------
# Реестр моделей + вспомогательное
# --------------------------------------------------------------------------
MODEL_BUILDERS = {
    "baseline": build_baseline_cnn,
    "efficientnet": build_efficientnet,
    "dinov2": build_dinov2,
}


def count_params(model: nn.Module) -> tuple[int, int]:
    """(всего параметров, обучаемых параметров)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
