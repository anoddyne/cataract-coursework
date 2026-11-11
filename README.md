# Выявление катаракты по изображениям глаза

Курсовая работа: применение глубокого обучения к бинарной классификации
изображений глаз («normal» / «cataract»). Датасет:
[Kaggle — Cataract Image Dataset](https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset)
(612 изображений: 306 normal / 306 cataract).

Реализованы три подхода и полный цикл анализа:

1. **Предобработка и EDA** — распределение классов, размеры, цветовые каналы,
   pipeline аугментаций.
2. **Подход 1 (базовый)** — свёрточная сеть, обученная «с нуля».
3. **Подход 2 (transfer learning)** — EfficientNet-B0, предобученная на ImageNet.
4. **Подход 3 (продвинутый)** — замороженная самообученная ViT-S/14 (DINOv2)
   с обучаемой attention-головой (механизм внимания над патч-токенами).
5. **Сравнительный анализ** — Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
   параметры, латентность инференса, время обучения.
6. **Интерпретируемость** — Grad-CAM (собственная реализация, без внешних
   библиотек): для ViT градиенты считаются по патч-токенам (16×16).
7. **Устойчивость** — гауссов шум, яркость, контраст, JPEG-сжатие.

## Структура проекта

```
├── data/
│   ├── raw/                  # исходный архив датасета Kaggle
│   └── processed/            # train/val/test/{normal,cataract} (создаётся скриптом)
├── src/
│   ├── config.py             # пути, гиперпараметры, выбор устройства (ROCm/CUDA/CPU)
│   ├── prepare_data.py       # разбиение train -> train/val (стратифицированно)
│   ├── eda.py                # разведочный анализ (графики -> reports/eda/)
│   ├── data.py               # Dataset и transforms, аугментации
│   ├── models.py             # три модели: baseline CNN / EfficientNet-B0 / DINOv2+attention
│   ├── train.py              # обучение (--model baseline|efficientnet|dinov2)
│   ├── evaluate.py           # метрики на тесте, ROC, матрицы ошибок, сравнение
│   ├── gradcam.py            # интерпретируемость Grad-CAM
│   └── robustness.py         # устойчивость к искажениям
├── models/                   # чекпоинты .pt
├── reports/                  # графики, таблицы, метрики (CSV/JSON/PNG/MD)
└── python_vs.bat             # запуск Python в окружении VS (нужно для MIOpen JIT на Windows-ROCm)
```

## Установка

Требуется Python 3.12.

```cmd
python -m venv .venv
.venv\Scripts\activate
```

PyTorch с поддержкой ROCm для AMD Radeon RX 9060 XT (gfx1200, Windows) —
по официальной инструкции AMD
([Install PyTorch — AMD ROCm docs](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)):

```cmd
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ ^
    "torch[device-gfx1200]==2.14.0+rocm10.1.0" ^
    "torchvision[device-gfx1200]==0.29.0a0+rocm10.1.0"
```

Для NVIDIA GPU или CPU: `pip install torch torchvision`.

Остальные зависимости: `pip install -r requirements.txt`.

**Важно (Windows-ROCm):** JIT-компиляция ядер MIOpen требует заголовков C++
из Visual Studio Build Tools. Скрипты запускаются через обёртку
`python_vs.bat` (активирует vcvars64 + вызывает python). Без неё BatchNorm
падает с `HIPRTC_ERROR_COMPILATION`.

## Запуск

```cmd
python_vs.bat src/prepare_data.py          # разбиение train/val/test
python_vs.bat src/eda.py                   # EDA-графики -> reports/eda/
python_vs.bat src/train.py --model baseline
python_vs.bat src/train.py --model efficientnet
python_vs.bat src/train.py --model dinov2
python_vs.bat src/evaluate.py              # сравнение всех моделей
python_vs.bat src/gradcam.py --model efficientnet   # интерпретируемость
python_vs.bat src/robustness.py            # устойчивость к искажениям
```

Настройки (размер изображения 224, batch 32, seed 42, ранняя остановка по
val ROC-AUC) — в `src/config.py`. Все модели обучаются с BCEWithLogitsLoss,
AdamW и косинусным расписанием LR.

## Результаты (тестовая выборка, 121 изображение)

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Параметры | Латентность, мс |
|---|---|---|---|---|---|---|---|
| Baseline CNN (с нуля) | 0.959 | 0.967 | 0.951 | 0.959 | 0.983 | 0.98 млн | 1.1 |
| **EfficientNet-B0 (transfer)** | **0.975** | **1.000** | 0.951 | **0.975** | **0.999** | 4.0 млн | 11.4 |
| DINOv2 ViT-S/14 + attention | 0.967 | **1.000** | 0.934 | 0.966 | 0.998 | 22.7 млн (0.64 обучаемых) | 13.2 |

- Лучшее качество — transfer learning на EfficientNet-B0 (ROC-AUC 0.9986,
  ноль ложных срабатываний: Precision = 1.0).
- DINOv2 сопоставим по качеству при этом обучается только attention-голова
  (0.64 млн параметров из 22.7 млн) — признак силы самообученных признаков
  на малом датасете.
- Baseline CNN заметно проще (1 млн параметров, латентность 1.1 мс —
  в 10 раз быстрее) и всё же достигает ROC-AUC 0.983.

### Устойчивость к искажениям (Accuracy)

| Искажение | Baseline | EfficientNet | DINOv2 |
|---|---|---|---|
| Чистый тест | 0.959 | 0.975 | 0.967 |
| Гауссов шум σ=10 | 0.959 | 0.975 | 0.967 |
| Гауссов шум σ=30 | 0.868 | **0.678** | **0.975** |
| Яркость +40 | 0.934 | 0.975 | 0.967 |
| Контраст ×0.6 | **0.826** | 0.975 | 0.959 |
| JPEG q=30 | 0.950 | 0.975 | 0.967 |

Выводы: EfficientNet-B0 даёт лучшую точность на чистых данных, но резко
чувствителен к сильному гауссову шуму; DINOv2 (самообученные признаки)
самый шумоустойчивый; CNN «с нуля» чувствительна к падению контраста.
Полные кривые — `reports/robustness_curves.png`.

### Интерпретируемость (Grad-CAM)

Для изображений с катарактой модели фокусируются на мутном хрусталике
(зрачковая область с рефлексом); для здоровых глаз активации рассеяны
вне зрачка. Файлы: `reports/gradcam_overview_{model}.png` и поштучные
карты в `reports/gradcam/`.

## Артефакты для отчёта

- `reports/eda/` — распределение классов, размеры, каналы, примеры.
- `reports/comparison.csv` / `comparison.md` — таблица метрик.
- `reports/roc_curves.png`, `reports/confusion_matrices.png`.
- `reports/curves_{model}.png` — кривые обучения.
- `reports/history_{model}.json` — история эпох.
- `reports/gradcam/` — тепловые карты.
- `reports/robustness.csv` — полный лог устойчивости.
- `models/{baseline,efficientnet,dinov2}.pt` — чекпоинты.
