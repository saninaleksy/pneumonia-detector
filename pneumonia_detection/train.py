from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
from sklearn.metrics import auc, confusion_matrix, roc_curve

from pneumonia_detection.datamodule import ChestXRayDataModule
from pneumonia_detection.model import PneumoniaModel


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_mlflow_logger(cfg: DictConfig) -> Optional[MLFlowLogger]:
    tracking_uri = OmegaConf.select(cfg, "logger.mlflow.tracking_uri")
    experiment = OmegaConf.select(cfg, "logger.mlflow.experiment_name")
    if tracking_uri is None or experiment is None:
        return None

    run_name = OmegaConf.select(cfg, "logger.mlflow.run_name")
    log_model = bool(OmegaConf.select(cfg, "logger.mlflow.log_model"))

    return MLFlowLogger(
        tracking_uri=str(tracking_uri),
        experiment_name=str(experiment),
        run_name=(None if run_name is None else str(run_name)),
        log_model=log_model,
    )


def _build_trainer(cfg: DictConfig, logger: Any, callbacks: list[Any]) -> Trainer:
    return Trainer(
        max_epochs=int(OmegaConf.select(cfg, "train.max_epochs")),
        accelerator=str(OmegaConf.select(cfg, "train.accelerator")),
        devices=OmegaConf.select(cfg, "train.devices"),
        num_nodes=int(OmegaConf.select(cfg, "train.num_nodes")),
        precision=str(OmegaConf.select(cfg, "train.precision")),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(OmegaConf.select(cfg, "train.log_every_n_steps")),
        deterministic=bool(OmegaConf.select(cfg, "train.deterministic")),
    )


def _save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for (row, col), val in np.ndenumerate(cm):
        plt.text(col, row, str(int(val)), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _save_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, out_path: Path
) -> None:
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _read_metrics_csv(csv_logger: CSVLogger) -> pd.DataFrame:
    metrics_file = Path(csv_logger.log_dir) / "metrics.csv"
    if not metrics_file.is_file():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_file}")
    return pd.read_csv(metrics_file)


def _epoch_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    series = df[["epoch", col]].dropna()
    if series.empty:
        return None
    return series.groupby("epoch")[col].last()


def _save_training_curves(csv_logger: CSVLogger, out_dir: Path) -> None:
    df = _read_metrics_csv(csv_logger)

    train_loss = _epoch_series(df, "train_loss")
    val_loss = _epoch_series(df, "val_loss")
    train_acc = _epoch_series(df, "train_acc")
    val_acc = _epoch_series(df, "val_acc")
    lr = _epoch_series(df, "lr-Adam") or _epoch_series(df, "lr")

    if train_loss is not None or val_loss is not None:
        plt.figure()
        if train_loss is not None:
            plt.plot(train_loss.index, train_loss.values, label="train_loss")
        if val_loss is not None:
            plt.plot(val_loss.index, val_loss.values, label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss.png")
        plt.close()

    if train_acc is not None or val_acc is not None:
        plt.figure()
        if train_acc is not None:
            plt.plot(train_acc.index, train_acc.values, label="train_acc")
        if val_acc is not None:
            plt.plot(val_acc.index, val_acc.values, label="val_acc")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy.png")
        plt.close()

    if lr is not None:
        plt.figure()
        plt.plot(lr.index, lr.values, label="lr")
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "lr.png")
        plt.close()


def _collect_probs_and_targets(
    model: torch.nn.Module, datamodule: ChestXRayDataModule
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device

    probs_all: list[float] = []
    y_all: list[int] = []

    loader = datamodule.test_dataloader()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            probs = model(images)
            probs_all.extend(probs.detach().cpu().numpy().ravel().tolist())
            y_all.extend(targets.detach().cpu().numpy().astype(int).ravel().tolist())

    return np.asarray(probs_all, dtype=np.float32), np.asarray(y_all, dtype=int)


def train_model(cfg: DictConfig) -> None:
    seed = int(OmegaConf.select(cfg, "train.seed"))
    seed_everything(seed, workers=True)

    dm = ChestXRayDataModule(cfg.data)
    model_cfg = OmegaConf.merge(cfg.model, {"data": cfg.data})
    model = PneumoniaModel(model_cfg)

    csv_logger = CSVLogger(save_dir="artifacts", name="csv_logs")
    mlflow_logger = _build_mlflow_logger(cfg)

    loggers: list[Any] = [csv_logger]
    if mlflow_logger is not None:
        loggers.append(mlflow_logger)

    ckpt_dir = Path(str(OmegaConf.select(cfg, "train.checkpoint.dirpath")))
    _ensure_dir(ckpt_dir)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor=str(OmegaConf.select(cfg, "train.checkpoint.monitor")),
        mode=str(OmegaConf.select(cfg, "train.checkpoint.mode")),
        save_top_k=int(OmegaConf.select(cfg, "train.checkpoint.save_top_k")),
        filename=str(OmegaConf.select(cfg, "train.checkpoint.filename")),
    )

    lr_cb = LearningRateMonitor(logging_interval="epoch")
    trainer = _build_trainer(cfg, logger=loggers, callbacks=[ckpt_cb, lr_cb])

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    plots_dir = Path("artifacts") / "plots"
    _ensure_dir(plots_dir)

    _save_training_curves(csv_logger, plots_dir)

    model.freeze()
    probs, targets = _collect_probs_and_targets(model, dm)

    pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(targets, pred)
    _save_confusion_matrix(cm, plots_dir / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = float(auc(fpr, tpr))
    _save_roc_curve(fpr, tpr, roc_auc, plots_dir / "roc_curve.png")
