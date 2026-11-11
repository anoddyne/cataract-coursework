"""Оценка обученных моделей на тестовой выборке и сравнительный анализ.

Считает для каждой модели: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
матрицу ошибок, число параметров, размер файла модели, среднюю латентность
инференса (мс/изображение) и суммарное время обучения.

Результаты: reports/comparison.csv, reports/comparison.md,
reports/roc_curves.png, reports/confusion_matrices.png.
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from config import DEVICE, MODELS_DIR, REPORTS_DIR, ensure_dirs
from data import make_loaders
from models import MODEL_BUILDERS, count_params


def load_model(name: str) -> nn.Module:
    ckpt = torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE,
                      weights_only=False)
    model = MODEL_BUILDERS[name]()
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model


@torch.inference_mode()
def collect_predictions(model, loader):
    probs, labels = [], []
    for images, y in loader:
        logits = model(images.to(DEVICE))
        probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


@torch.inference_mode()
def measure_latency(model, img_size=224, batch_size=1, warmup=20, iters=100):
    """Средняя латентность инференса на изображение, мс."""
    x = torch.randn(batch_size, 3, img_size, img_size, device=DEVICE)
    for _ in range(warmup):
        model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters / batch_size * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["baseline", "efficientnet", "dinov2"])
    args = parser.parse_args()

    ensure_dirs()
    _, _, test_dl = make_loaders()
    rows, roc_data, cm_data = [], {}, {}

    for name in args.models:
        model = load_model(name)
        total_p, trainable_p = count_params(model)
        probs, labels = collect_predictions(model, test_dl)
        preds = (probs >= 0.5).astype(int)

        ckpt_mb = (MODELS_DIR / f"{name}.pt").stat().st_size / 1e6
        hist_path = REPORTS_DIR / f"history_{name}.json"
        train_time = None
        if hist_path.exists():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            if hist and "epoch_time_s" in hist[0]:
                train_time = sum(h.get("epoch_time_s", 0) for h in hist)

        latency = measure_latency(model)

        rows.append({
            "Модель": name,
            "Accuracy": accuracy_score(labels, preds),
            "Precision": precision_score(labels, preds, zero_division=0),
            "Recall": recall_score(labels, preds, zero_division=0),
            "F1": f1_score(labels, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(labels, probs),
            "PR-AUC": average_precision_score(labels, probs),
            "Параметров (млн)": round(total_p / 1e6, 2),
            "Обучаемых (млн)": round(trainable_p / 1e6, 2),
            "Размер модели, МБ": round(ckpt_mb, 1),
            "Латентность, мс/изобр.": round(latency, 2),
            "Время обучения, с": round(train_time, 1) if train_time else None,
        })
        roc_data[name] = roc_curve(labels, probs)
        cm_data[name] = confusion_matrix(labels, preds)
        print(f"[{name}] готово")

    df = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / "comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Markdown-таблица для отчёта
    md_lines = ["| " + " | ".join(df.columns) + " |",
                "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, r in df.iterrows():
        md_lines.append("| " + " | ".join(
            f"{v:.4f}" if isinstance(v, float) and 0 < v < 1 else str(v)
            for v in r.values) + " |")
    (REPORTS_DIR / "comparison.md").write_text(
        "\n".join(md_lines), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nТаблица сравнения: {csv_path}")

    # ROC-кривые всех моделей
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (fpr, tpr, _) in roc_data.items():
        auc = df.loc[df["Модель"] == name, "ROC-AUC"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="случайный классификатор")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC-кривые на тестовой выборке")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curves.png", dpi=150)

    # Матрицы ошибок
    fig, axes = plt.subplots(1, len(cm_data), figsize=(4.2 * len(cm_data), 4))
    if len(cm_data) == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cm_data.items()):
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14)
        ax.set_xticks([0, 1], ["normal", "cataract"])
        ax.set_yticks([0, 1], ["normal", "cataract"])
        ax.set(xlabel="Предсказание", ylabel="Истина", title=name)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=150)
    print("Графики ROC и матриц ошибок сохранены в", REPORTS_DIR)


if __name__ == "__main__":
    main()
