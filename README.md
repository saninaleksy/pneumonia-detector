# Детекция пневмонии по рентгеновским снимкам

---

## Описание задачи

Цель проекта - построить систему бинарной
классификации рентгеновских снимков грудной клетки для выявления
пневмонии.

Модель принимает изображение (X-ray) и возвращает вероятность наличия
пневмонии.

Выход модели - число от 0 до 1 (вероятность пневмонии).

---

## Архитектура решения

Структура проекта:

    pneumonia_detection/
    │
    ├── commands.py
    ├── datamodule.py
    ├── export_to_onnx.py
    ├── import_data.py
    ├── inference.py
    ├── model.py
    ├── onnx_to_tensorrt.py
    ├── run_triton.py
    ├── train.py
    ├── triton_setup.py
    │
    configs/
    ├── config.yaml
    ├── data/default.yaml
    ├── export/default.yaml
    ├── inference/default.yaml
    ├── logger/mlflow.yaml
    ├── model/default.yaml
    ├── train/default.yaml
    ├── triton/default.yaml
    │
    dvc.yaml
    pyproject.toml

---

# Setup

## Установка Poetry

```bash
pip install poetry
```

## Установка зависимостей

```bash
poetry env use python3.11
poetry install
poetry run pre-commit install
```

После этого будут установлены:

- PyTorch Lightning
- Hydra
- MLflow
- DVC
- ONNX Runtime
- tritonclient
- инструменты проверки кода

Для использования инференса с gpu нужно дополнительно установить pycuda и tensorrt

---

## Настройка MLflow

Запуск сервера:

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
```

---

## Настройка DVC

Инициализация:

```bash
dvc init
```

Подключение удалённого хранилища:

```bash
poetry run pip install "dvc[gdrive]"
dvc remote add -d storage gdrive://<FOLDER_ID>
```

Загрузка данных:

```bash
poetry run dvc pull
```

---

# Train

## Загрузка данных

Датасет: Kaggle Chest X-Ray Pneumonia.

Данные подгружаются автоматически при обучении, при желании можно загрузить вручную:

```bash
poetry run python -m pneumonia_detection.commands import_data
```

Либо через DVC (если он предварительно настроен):

```bash
poetry run dvc pull
```

Для загрузки через Kaggle необходимо предварительно добавить конфигурационный файл ~/.kaggle/kaggle.json

```
{
  "username": "kaggle_username",
  "key": "KGAT_1234567890"
}

```

---

## Запуск обучения

Базовый запуск:

```bash
poetry run python -m pneumonia_detection.commands train
```

Переопределение параметров Hydra:

```bash
poetry run python -m pneumonia_detection.commands train train.max_epochs=5 train.accelerator=gpu
```

В ходе обучения:

- сохраняются чекпойнты в artifacts/checkpoints
- строятся графики в artifacts/plots
- метрики логируются в MLflow

---

# Production preparation

После завершения обучения модель необходимо подготовить к production.

## Экспорт в ONNX

```bash
poetry run python -m pneumonia_detection.commands export     export.ckpt_path=artifacts/checkpoints/best.ckpt     export.onnx_path=artifacts/model.onnx
```

## Конвертация в TensorRT

```bash
poetry run python -m pneumonia_detection.commands onnx_to_tensorrt     export.onnx_path=artifacts/model.onnx     export.engine_path=artifacts/model.engine
```

## Подготовка Triton

```bash
poetry run python -m pneumonia_detection.commands setup_triton
```

Будет создана структура:

    triton_models/
    └── pneumonia_detection/
        ├── config.pbtxt
        ├── labels.txt
        └── 1/
            └── model.onnx или model.plan

---

# Infer

После обучения модель можно использовать для предсказания на новых
данных.

---

## Инференс через ONNX

```bash
poetry run python -m pneumonia_detection.commands infer     inference.backend=onnx     inference.onnx_path=artifacts/model.onnx     inference.img_path=/path/image.jpeg
```

---

## Инференс через TensorRT

```bash
poetry run python -m pneumonia_detection.commands infer     inference.backend=tensorrt     inference.engine_path=artifacts/model.engine     inference.img_path=/path/image.jpeg
```

---

## Инференс через Triton

1.  Запуск Triton:

```bash
poetry run python -m pneumonia_detection.commands run_triton
```

2.  Запуск инференса:

```bash
poetry run python -m pneumonia_detection.commands infer     inference.backend=triton     inference.triton_url=localhost:8000     inference.img_path=/path/image.jpeg
```
