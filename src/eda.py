"""Разведочный анализ данных (EDA).

Строит и сохраняет в reports/eda/:
  - распределение классов по сплитам (дисбаланс);
  - распределение размеров изображений;
  - статистику цветовых каналов и яркости по классам;
  - сетку примеров изображений обоих классов;
  - сводную статистику в reports/eda_stats.json.
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import CLASSES, EDA_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS = ["train", "val", "test"]


def list_split(split):
    out = {}
    for cls in CLASSES:
        d = PROCESSED_DIR / split / cls
        out[cls] = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    return out


def plot_class_distribution(counts):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SPLITS))
    w = 0.35
    for i, cls in enumerate(CLASSES):
        vals = [counts[s][cls] for s in SPLITS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=cls)
        ax.bar_label(bars, fontsize=9)
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("Число изображений")
    ax.set_title("Распределение классов по сплитам")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_sizes(sizes, classes_per_img):
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(w, bins=30, alpha=0.7, label="ширина")
    axes[0].hist(h, bins=30, alpha=0.7, label="высота")
    axes[0].set(title="Распределение размеров", xlabel="пиксели", ylabel="число изображений")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    for cls, marker in zip(CLASSES, "o^"):
        idx = [i for i, c in enumerate(classes_per_img) if c == cls]
        axes[1].scatter(w[idx], h[idx], s=12, alpha=0.5, marker=marker, label=cls)
    axes[1].set(title="Ширина vs высота по классам", xlabel="ширина", ylabel="высота")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "image_sizes.png", dpi=150)
    plt.close(fig)
    return {"width_min": int(w.min()), "width_max": int(w.max()),
            "height_min": int(h.min()), "height_max": int(h.max()),
            "n_unique_sizes": len(set(sizes))}


def plot_channel_stats(images_by_class):
    """Средние яркости каналов R, G, B и общая гистограмма яркости."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    means = {}
    for cls in CLASSES:
        # изображения разного размера — усредняем по-канально каждое, потом по списку
        per_img = [np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(axis=0)
                   for im in images_by_class[cls][:150]]
        means[cls] = np.stack(per_img).mean(axis=0)
    x = np.arange(3)
    w = 0.35
    for i, cls in enumerate(CLASSES):
        bars = axes[0].bar(x + (i - 0.5) * w, means[cls], w, label=cls)
        axes[0].bar_label(bars, fmt="%.0f", fontsize=9)
    axes[0].set_xticks(x, ["R", "G", "B"])
    axes[0].set_ylabel("Средняя интенсивность (0-255)")
    axes[0].set_title("Средняя интенсивность каналов по классам")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    for cls in CLASSES:
        vals = np.concatenate([
            np.asarray(im.convert("L"), dtype=np.float32).ravel()
            for im in images_by_class[cls][:100]
        ])
        axes[1].hist(vals, bins=50, alpha=0.5, density=True, label=cls)
    axes[1].set(title="Гистограмма яркости (grayscale)", xlabel="яркость", ylabel="плотность")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "channel_stats.png", dpi=150)
    plt.close(fig)
    return {cls: [float(v) for v in means[cls]] for cls in CLASSES}


def plot_sample_grid(files_by_class, n=8):
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.6))
    for row, cls in enumerate(CLASSES):
        for col in range(n):
            ax = axes[row, col]
            img = Image.open(files_by_class[cls][col]).convert("RGB")
            ax.imshow(img)
            ax.set_title(cls, fontsize=9)
            ax.axis("off")
    fig.suptitle("Примеры изображений (train)", fontsize=12)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "samples.png", dpi=150)
    plt.close(fig)


def main():
    ensure_dirs()
    EDA_DIR.mkdir(exist_ok=True)

    counts = {s: {c: len(files) for c, files in list_split(s).items()} for s in SPLITS}
    total = {c: sum(counts[s][c] for s in SPLITS) for c in CLASSES}
    print("Распределение классов:", json.dumps(counts, ensure_ascii=False))
    print("Всего по классам:", total)
    imbalance = total[CLASSES[0]] / total[CLASSES[1]]
    print(f"Соотношение классов normal/cataract: {imbalance:.2f} (1.0 = идеальный баланс)")

    plot_class_distribution(counts)

    # Считываем все изображения один раз
    all_sizes, all_classes = [], []
    images_by_class = {c: [] for c in CLASSES}
    files_by_class = {c: [] for c in CLASSES}
    for s in SPLITS:
        for cls, files in list_split(s).items():
            for p in files:
                im = Image.open(p).convert("RGB")
                all_sizes.append(im.size)
                all_classes.append(cls)
                if len(images_by_class[cls]) < 150:
                    images_by_class[cls].append(im)
                if len(files_by_class[cls]) < 8 and s == "train":
                    files_by_class[cls].append(p)

    size_stats = plot_sizes(all_sizes, all_classes)
    print("Статистика размеров:", size_stats)

    channel_means = plot_channel_stats(images_by_class)
    print("Средние значения каналов (R,G,B):", channel_means)

    plot_sample_grid(files_by_class)

    stats = {"counts": counts, "total": total, "imbalance_ratio": imbalance,
             "size_stats": size_stats, "channel_means_rgb": channel_means}
    out = REPORTS_DIR / "eda_stats.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Сводка сохранена: {out}")


if __name__ == "__main__":
    main()
