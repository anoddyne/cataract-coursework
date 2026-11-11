"""Оценка устойчивости моделей к искажениям входных данных.

Искажения применяются к изображениям тестовой выборки на уровне PIL:
  - гауссов шум (сигма 5/10/20/30 из 255);
  - сдвиг яркости (+/-20, +/-40, множитель 1 +/- d/255);
  - изменение контраста (x0.6/x0.8/x1.2/x1.4);
  - JPEG-сжатие (качество 90/70/50/30; точка 100 = исходное изображение).

Для каждой модели и уровня искажения считается Accuracy и ROC-AUC.
Результаты: reports/robustness.csv и reports/robustness_curves.png.
"""
import argparse
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import roc_auc_score

from config import CLASSES, DEVICE, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from data import eval_transform
from evaluate import load_model

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def apply_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.default_rng(0).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_perturbations():
    """{семейство: [(уровень, функция PIL->PIL), ...]} — уровень-опора
    (0 для шума/яркости, 1.0 для контраста, 100 для JPEG) = без искажения."""
    identity = lambda im: im
    return {
        "гауссов шум": [(s, (lambda im, s=s: apply_gaussian_noise(im, s)) if s else identity)
                        for s in (0, 5, 10, 20, 30)],
        "яркость": [(d, (lambda im, d=d: ImageEnhance.Brightness(im).enhance(1 + d / 255)) if d else identity)
                    for d in (-40, -20, 0, 20, 40)],
        "контраст": [(f, (lambda im, f=f: ImageEnhance.Contrast(im).enhance(f)) if f != 1 else identity)
                     for f in (0.6, 0.8, 1.0, 1.2, 1.4)],
        "JPEG": [(q, (lambda im, q=q: apply_jpeg(im, q)) if q < 100 else identity)
                 for q in (100, 90, 70, 50, 30)],
    }


@torch.inference_mode()
def evaluate_perturbed(model, perturb_fn):
    """Accuracy и ROC-AUC на тестовой выборке с данным искажением."""
    probs, labels = [], []
    for cls_idx, cls in enumerate(CLASSES):
        for path in sorted((PROCESSED_DIR / "test" / cls).iterdir()):
            if path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            img = perturb_fn(Image.open(path).convert("RGB"))
            x = eval_transform(img).unsqueeze(0).to(DEVICE)
            probs.append(torch.sigmoid(model(x)).item())
            labels.append(cls_idx)
    probs, labels = np.asarray(probs), np.asarray(labels)
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    auc = float(roc_auc_score(labels, probs))
    return acc, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["baseline", "efficientnet", "dinov2"])
    args = parser.parse_args()

    ensure_dirs()
    families = build_perturbations()
    rows = []

    for name in args.models:
        model = load_model(name)
        print(f"=== {name} ===")
        for family, levels in families.items():
            for level, fn in levels:
                acc, auc = evaluate_perturbed(model, fn)
                rows.append({"Модель": name, "Искажение": family,
                             "Уровень": level, "Accuracy": acc, "ROC-AUC": auc})
                print(f"  {family} {level}: acc={acc:.4f} auc={auc:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "robustness.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, (family, levels) in zip(axes, families.items()):
        xs = [lvl for lvl, _ in levels]
        for name in args.models:
            ys = [df[(df["Модель"] == name) & (df["Искажение"] == family)
                     & (df["Уровень"] == lvl)]["Accuracy"].iloc[0]
                  for lvl, _ in levels]
            ax.plot(xs, ys, marker="o", label=name)
        ax.set(title=family, xlabel="жёсткость искажения", ylabel="Accuracy")
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Устойчивость моделей к искажениям входных данных (тестовая выборка)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "robustness_curves.png", dpi=150)
    print(f"\nСохранено: {REPORTS_DIR / 'robustness.csv'} и "
          f"{REPORTS_DIR / 'robustness_curves.png'}")


if __name__ == "__main__":
    main()
