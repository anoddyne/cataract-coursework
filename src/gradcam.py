"""Grad-CAM: интерпретируемость предсказаний моделей.

Реализация Grad-CAM вручную (без внешних библиотек): регистрируем хуки на
выбранный свёрточный слой, считаем градиенты логита класса по активациям,
усредняем градиенты по пространству (веса кампании), взвешиваем активации,
применяем ReLU и нормируем.

Для ViT/DINOv2 слой выдает патч-токены (B, 1+S, C) — перед наложением
токены (без CLS) reshape'ятся в карту 16x16.

Результат: reports/gradcam/<model>__<image>.png — оригинал, тепловая карта,
наложение; и reports/gradcam_overview.png — сводная сетка.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import CLASSES, DEVICE, IMG_SIZE, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from data import eval_transform
from evaluate import load_model


def get_target_module(model: nn.Module, name: str):
    """Целевой слой Grad-CAM: последние карты признаков каждой архитектуры."""
    if name == "baseline":
        return model[16]                  # nn.Conv2d(256, 256) — последний свёрточный
    if name == "efficientnet":
        return model.features[-1]
    if name == "dinov2":
        # выход последнего блока ViT: (B, 1+256, 384) патч-токенов
        return model.backbone.blocks[-1]
    raise ValueError(name)


def reshape_dinov2_tokens(act: torch.Tensor) -> torch.Tensor:
    """(B, 1+S, C) -> (B, C, 16, 16): выбрасываем CLS-токен."""
    cls, patches = act[:, :1], act[:, 1:]
    b, s, c = patches.shape
    side = int(s ** 0.5)
    return patches.transpose(1, 2).reshape(b, c, side, side)


class GradCAM:
    def __init__(self, model: nn.Module, target_module: nn.Module,
                 token_based: bool = False):
        self.model = model
        self.token_based = token_based
        self.acts, self.grads = None, None
        self.h1 = target_module.register_forward_hook(self._save_act)

    def _save_act(self, module, inp, out):
        if not out.requires_grad:
            # Замороженный бэкбон: выход слоя — «лист» без grad_fn.
            # Включаем у него requires_grad, чтобы градиент считался от головы
            # вплоть до этой активации (стандартный приём Grad-CAM).
            out.requires_grad_(True)
        else:
            out.retain_grad()
        out.register_hook(self._save_grad)
        self.acts = out

    def _save_grad(self, grad):
        self.grads = grad.detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, 0].backward()  # логит "катаракты"

        acts, grads = self.acts, self.grads
        if self.token_based:
            acts, grads = reshape_dinov2_tokens(acts), reshape_dinov2_tokens(grads)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # GAP по градиентам
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                            align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        mx = cam.max()
        return cam / mx if mx > 0 else np.zeros_like(cam)

    def remove(self):
        self.h1.remove()


def overlay_heatmap(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    """Наложение тепловой карты (jet) на исходное изображение."""
    base = img_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    colormap = plt.get_cmap("jet")
    heat = (colormap(cam)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat)
    blended = Image.blend(base, heat_img, alpha=alpha)
    return base, Image.fromarray((cam * 255).astype(np.uint8)).convert("L"), blended


def pick_examples(n_per_class: int = 3):
    """Несколько тестовых изображений каждого класса."""
    out = {}
    for cls in CLASSES:
        files = sorted((PROCESSED_DIR / "test" / cls).iterdir())
        out[cls] = files[:n_per_class]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet",
                        choices=["baseline", "efficientnet", "dinov2"])
    parser.add_argument("--n", type=int, default=3,
                        help="число примеров на класс")
    args = parser.parse_args()

    ensure_dirs()
    out_dir = REPORTS_DIR / "gradcam"
    out_dir.mkdir(exist_ok=True)

    model = load_model(args.model)
    target = get_target_module(model, args.model)
    cam_engine = GradCAM(model, target, token_based=(args.model == "dinov2"))

    to_tensor = eval_transform
    examples = pick_examples(args.n)
    overview_rows = []

    for cls, files in examples.items():
        for path in files:
            img_pil = Image.open(path).convert("RGB")
            x = to_tensor(img_pil).unsqueeze(0).to(DEVICE)
            with torch.enable_grad():
                cam = cam_engine(x)
            with torch.inference_mode():
                prob = torch.sigmoid(model(x))[0, 0].item()

            base, heat, blended = overlay_heatmap(img_pil, cam)
            stem = f"{args.model}__{path.stem}_{cls}_p{prob:.2f}"
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
            axes[0].imshow(base); axes[0].set_title(f"оригинал ({cls})")
            axes[1].imshow(heat, cmap="jet"); axes[1].set_title("Grad-CAM")
            axes[2].imshow(blended)
            axes[2].set_title(f"P(катаракта)={prob:.2f}")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{stem}.png", dpi=130)
            plt.close(fig)
            overview_rows.append((base, heat, blended, cls, prob))
            print(f"{stem}.png")

    cam_engine.remove()

    # Сводная сетка: строки — классы, столбцы — примеры
    n_cols = max(len(v) for v in examples.values())
    fig, axes = plt.subplots(len(CLASSES), n_cols,
                             figsize=(2.6 * n_cols, 2.9 * len(CLASSES)))
    axes = np.atleast_2d(axes)
    idx = 0
    for row, cls in enumerate(CLASSES):
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(examples[cls]):
                base, heat, blended, _, prob = overview_rows[idx]
                ax.imshow(blended)
                ax.set_title(f"{cls}, p={prob:.2f}", fontsize=9)
                idx += 1
            ax.axis("off")
    fig.suptitle(f"Grad-CAM ({args.model}): области, повлиявшие на решение",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"gradcam_overview_{args.model}.png", dpi=140)
    print(f"Сводка: {REPORTS_DIR / f'gradcam_overview_{args.model}.png'}")


if __name__ == "__main__":
    main()
