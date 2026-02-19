from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy


class PneumoniaModel(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if OmegaConf.is_config(cfg):
            hparams = OmegaConf.to_container(cfg, resolve=True)
        else:
            hparams = cfg
        self.save_hyperparameters(hparams)

        in_ch = int(OmegaConf.select(cfg, "data.input_channels"))
        img_size = int(OmegaConf.select(cfg, "data.img_size"))

        channels = list(OmegaConf.select(cfg, "backbone.channels"))
        kernel_size = int(OmegaConf.select(cfg, "backbone.kernel_size"))
        stride = int(OmegaConf.select(cfg, "backbone.stride"))
        padding = int(OmegaConf.select(cfg, "backbone.padding"))

        pool_kernel = int(OmegaConf.select(cfg, "backbone.pool_kernel"))
        pool_stride = int(OmegaConf.select(cfg, "backbone.pool_stride"))
        ceil_mode = bool(OmegaConf.select(cfg, "backbone.ceil_mode"))

        dropouts = list(OmegaConf.select(cfg, "backbone.dropouts"))
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], kernel_size, stride, padding),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_kernel, pool_stride, ceil_mode=ceil_mode),
            nn.Conv2d(channels[0], channels[1], kernel_size, stride, padding),
            nn.Dropout(p=float(dropouts[1])),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_kernel, pool_stride, ceil_mode=ceil_mode),
            nn.Conv2d(channels[1], channels[2], kernel_size, stride, padding),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_kernel, pool_stride, ceil_mode=ceil_mode),
            nn.Conv2d(channels[2], channels[3], kernel_size, stride, padding),
            nn.Dropout(p=float(dropouts[3])),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_kernel, pool_stride, ceil_mode=ceil_mode),
            nn.Conv2d(channels[3], channels[4], kernel_size, stride, padding),
            nn.Dropout(p=float(dropouts[4])),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_kernel, pool_stride, ceil_mode=ceil_mode),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, img_size, img_size)
            flat_dim = int(self.features(dummy).view(1, -1).shape[1])

        hidden_dim = int(OmegaConf.select(cfg, "head.hidden_dim"))
        head_dropout = float(OmegaConf.select(cfg, "head.dropout"))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.criterion = torch.nn.BCEWithLogitsLoss()

        threshold = float(OmegaConf.select(cfg, "metrics.threshold"))
        self.train_acc = BinaryAccuracy(threshold=threshold)
        self.val_acc = BinaryAccuracy(threshold=threshold)
        self.test_acc = BinaryAccuracy(threshold=threshold)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(inp))
        return logits.squeeze(dim=1)

    def _step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        images, targets = batch
        targets = targets.float()

        logits = self(images)
        loss = self.criterion(logits, targets)

        probs = torch.sigmoid(logits)

        if stage == "train":
            acc = self.train_acc(probs, targets.int())
        elif stage == "val":
            acc = self.val_acc(probs, targets.int())
        else:
            acc = self.test_acc(probs, targets.int())

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._step(batch, "val")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._step(batch, "test")

    def configure_optimizers(self):
        optimizer_name = str(OmegaConf.select(self.cfg, "optimizer.name")).lower()
        lr = float(OmegaConf.select(self.cfg, "optimizer.lr"))

        if optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        scheduler_name = str(OmegaConf.select(self.cfg, "scheduler.name")).lower()

        if scheduler_name == "reduce_on_plateau":
            mode = str(OmegaConf.select(self.cfg, "scheduler.mode"))
            factor = float(OmegaConf.select(self.cfg, "scheduler.factor"))
            patience = int(OmegaConf.select(self.cfg, "scheduler.patience"))
            min_lr = float(OmegaConf.select(self.cfg, "scheduler.min_lr"))

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

            monitor = str(OmegaConf.select(self.cfg, "scheduler.monitor"))
            interval = str(OmegaConf.select(self.cfg, "scheduler.interval"))
            frequency = int(OmegaConf.select(self.cfg, "scheduler.frequency"))

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": interval,
                    "frequency": frequency,
                },
            }
