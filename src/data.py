"""Датасет и преобразования изображений."""
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_WORKERS,
    PROCESSED_DIR,
)

# Аугментации только для train: имитируют вариативность съёмки глаза
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CataractDataset(Dataset):
    """Читает пары (путь, метка) из data/processed/<split>/<class>."""

    def __init__(self, split: str, transform=None):
        self.samples = []
        self.transform = transform
        for label, cls in enumerate(CLASSES):
            cls_dir = PROCESSED_DIR / split / cls
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_loaders():
    """Возвращает DataLoader'ы для train/val/test."""
    train_ds = CataractDataset("train", train_transform)
    val_ds = CataractDataset("val", eval_transform)
    test_ds = CataractDataset("test", eval_transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS)
    return train_dl, val_dl, test_dl
