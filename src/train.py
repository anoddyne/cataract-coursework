"""Обучение одной из трёх моделей проекта с ранней остановкой.

Использование:
    python src/train.py --model baseline
    python src/train.py --model efficientnet
    python src/train.py --model dinov2

Аргументы: --model, --epochs, --batch-size, --lr.
Сохраняет: models/<name>.pt, reports/history_<name>.json,
reports/curves_<name>.png. Лучшая модель выбирается по val ROC-AUC.
"""
import argparse
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from config import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOP_PATIENCE,
    LR,
    MODELS_DIR,
    NUM_EPOCHS,
    REPORTS_DIR,
    SEED,
    WEIGHT_DECAY,
    ensure_dirs,
)
from data import make_loaders
from models import MODEL_BUILDERS, count_params

# Значения по умолчанию под каждую модель: CNN "с нуля" и лёгкая голова
# DINOv2 обучаются с большим LR; предобученный бэкбон EfficientNet — с меньшим.
MODEL_DEFAULTS = {
    "baseline": {"lr": 1e-3, "epochs": 40},
    "efficientnet": {"lr": 3e-4, "epochs": 25},
    "dinov2": {"lr": 1e-3, "epochs": 20},
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_epoch(model, loader, criterion, optimizer=None):
    """Одна эпоха: train при заданном optimizer, иначе оценка."""
    training = optimizer is not None
    model.train(training)
    total_loss, all_labels, all_probs = 0.0, [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.vstack(all_probs).ravel()
    labels = np.vstack(all_labels).ravel()
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    defaults = MODEL_DEFAULTS[args.model]
    epochs = args.epochs or defaults["epochs"]
    lr = args.lr or defaults["lr"]

    ensure_dirs()
    set_seed(SEED)

    if DEVICE == "cuda":
        # ROCm (AMD) и CUDA (NVIDIA) обе видны как torch.cuda
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"(ROCm {torch.version.hip})" if torch.version.hip else
              f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU недоступна, обучение на CPU")

    train_dl, val_dl, _ = make_loaders()
    print(f"Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")
    print(f"Модель: {args.model} | эпохи: {epochs} | lr: {lr}")

    model = MODEL_BUILDERS[args.model]().to(DEVICE)
    total_p, trainable_p = count_params(model)
    print(f"Параметров: {total_p:,} всего, {trainable_p:,} обучаемых")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_auc, best_state, patience = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, criterion, optimizer)
        va = run_epoch(model, val_dl, criterion)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": tr["loss"],
                        "train_acc": tr["accuracy"], "train_auc": tr["auc"],
                        "val_loss": va["loss"], "val_acc": va["accuracy"],
                        "val_auc": va["auc"],
                        "epoch_time_s": round(time.time() - t0, 2)})
        print(f"Эпоха {epoch:3d}/{epochs} | "
              f"train loss {tr['loss']:.4f} auc {tr['auc']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} "
              f"auc {va['auc']:.4f} | {time.time() - t0:.1f}s")

        if va["auc"] > best_auc:
            best_auc, best_state, patience = va["auc"], copy.deepcopy(
                model.state_dict()), 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Ранняя остановка: {EARLY_STOP_PATIENCE} эпох "
                      f"без улучшения val AUC")
                break

    model_path = MODELS_DIR / f"{args.model}.pt"
    torch.save({"state_dict": best_state or model.state_dict(),
                "best_val_auc": best_auc,
                "classes": ["normal", "cataract"],
                "config": {"img_size": 224, "arch": args.model}},
               model_path)
    print(f"Лучшая модель сохранена: {model_path} (val AUC = {best_auc:.4f})")

    (REPORTS_DIR / f"history_{args.model}.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    epochs_list = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs_list, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs_list, [h["val_loss"] for h in history], label="val")
    axes[0].set(xlabel="Эпоха", ylabel="Loss", title=f"{args.model}: функция потерь")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs_list, [h["train_auc"] for h in history], label="train")
    axes[1].plot(epochs_list, [h["val_auc"] for h in history], label="val")
    axes[1].set(xlabel="Эпоха", ylabel="ROC-AUC", title=f"{args.model}: ROC-AUC")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"curves_{args.model}.png", dpi=150)
    print(f"История и кривые обучения сохранены в {REPORTS_DIR}")


if __name__ == "__main__":
    main()
