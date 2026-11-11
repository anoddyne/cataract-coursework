"""Общие настройки проекта: пути, гиперпараметры, воспроизводимость."""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"  # train/val/test/{normal,cataract}

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
EDA_DIR = REPORTS_DIR / "eda"

CLASSES = ["normal", "cataract"]  # индекс = метка класса
NUM_CLASSES = len(CLASSES)

# Гиперпараметры обучения
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 7  # эпох без улучшения val ROC-AUC
NUM_WORKERS = 0  # Windows: надёжнее без воркеров (датасет маленький)
SEED = 42
VAL_FRACTION = 0.2  # доля валидации, выделяемой из train

# Нормализация ImageNet (совместима с предобученными весами)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Устройство: AMD ROCm и NVIDIA CUDA обе доступны через torch.cuda-интерфейс
# (в ROCm-сборках PyTorch "cuda" транслируется в HIP). Если GPU нет — CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
