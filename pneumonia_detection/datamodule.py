from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from pneumonia_detection.import_data import download_data

log = logging.getLogger(__name__)


class ChestXRayDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(str(self.cfg.root_dir))

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        train_dir = self.data_dir / "train"
        if not train_dir.exists():
            log.info(
                "Dataset not found at %s. Attempting Kaggle download...", train_dir
            )
            kaggle_dataset = str(self.cfg.kaggle_dataset)
            download_data(self.data_dir, kaggle_dataset)

    def _assert_structure(self) -> None:
        required = ["train", "test"]
        missing = [name for name in required if not (self.data_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required dataset folders {missing} in: {self.data_dir}"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        self._assert_structure()

        img_size = int(self.cfg.img_size)

        base = [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
        ]
        post = [transforms.ToTensor()]

        aug = []
        if bool(self.cfg.augmentation):
            aug = [
                transforms.RandomAffine(
                    degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]

        train_tf = transforms.Compose(base + aug + post)
        eval_tf = transforms.Compose(base + post)

        if stage in (None, "fit"):
            base_train = datasets.ImageFolder(self.data_dir / "train")
            targets = np.asarray(
                [sample for _, sample in base_train.samples], dtype=int
            )

            val_fraction = float(self.cfg.val_fraction)
            seed = int(self.cfg.seed)

            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_fraction, random_state=seed
            )
            train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

            train_full = datasets.ImageFolder(
                self.data_dir / "train", transform=train_tf
            )
            val_full = datasets.ImageFolder(self.data_dir / "train", transform=eval_tf)

            self.train_dataset = Subset(train_full, train_idx)
            self.val_dataset = Subset(val_full, val_idx)

        if stage in (None, "validate"):
            if self.val_dataset is None:
                base_train = datasets.ImageFolder(self.data_dir / "train")
                targets = np.asarray([y for _, y in base_train.samples], dtype=int)

                val_fraction = float(self.cfg.val_fraction)
                seed = int(self.cfg.seed)

                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=val_fraction, random_state=seed
                )
                _, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

                val_full = datasets.ImageFolder(
                    self.data_dir / "train", transform=eval_tf
                )
                self.val_dataset = Subset(val_full, val_idx)

        if stage in (None, "test"):
            self.test_dataset = datasets.ImageFolder(
                self.data_dir / "test", transform=eval_tf
            )

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.num_workers),
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self._loader(self.test_dataset, shuffle=False)
