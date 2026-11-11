"""Подготовка датасета: проверка структуры, разбивка train -> train/val.

Исходный датасет Kaggle (nandanp6/cataract-image-dataset) содержит папку
``processed_images/`` с готовым делением train/test и классами normal/cataract.
Мы оставляем официальный test нетронутым, а из train выделяем стратифицированную
валидацию. Результат складывается в data/processed/{train,val,test}/<class>.
"""
import shutil
import sys

from sklearn.model_selection import train_test_split

from config import CLASSES, PROCESSED_DIR, RAW_DIR, SEED, VAL_FRACTION, ensure_dirs

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_source_dir() -> "dict[str, dict[str, list]]":
    """Ищет в data/raw корневую папку вида <root>/train|test/<class>."""
    roots = list(RAW_DIR.glob("**/train/normal"))
    if not roots:
        print("Не найдена структура */train/normal в data/raw")
        sys.exit(1)
    root = roots[0].parent.parent  # .../processed_images
    print(f"Источник: {root}")
    out = {}
    for split in ("train", "test"):
        out[split] = {}
        for cls in CLASSES:
            cls_dir = root / split / cls
            files = sorted(
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in IMG_EXTENSIONS
            )
            out[split][cls] = files
    return out


def copy_files(files, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)


def main():
    ensure_dirs()
    src = find_source_dir()

    counts = {s: {c: len(src[s][c]) for c in CLASSES} for s in src}
    print("Исходное распределение:", counts)

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    # test остаётся как есть
    for cls in CLASSES:
        copy_files(src["test"][cls], PROCESSED_DIR / "test" / cls)

    # train -> train + val (стратифицированно)
    for cls in CLASSES:
        tr, va = train_test_split(
            src["train"][cls],
            test_size=VAL_FRACTION,
            random_state=SEED,
            shuffle=True,
        )
        copy_files(tr, PROCESSED_DIR / "train" / cls)
        copy_files(va, PROCESSED_DIR / "val" / cls)

    final_counts = {}
    for split in ("train", "val", "test"):
        final_counts[split] = {
            cls: len(list((PROCESSED_DIR / split / cls).glob("*")))
            for cls in CLASSES
        }
    print("Итоговое распределение:", final_counts)
    total = sum(sum(v.values()) for v in final_counts.values())
    print(f"Всего изображений: {total}")


if __name__ == "__main__":
    main()
