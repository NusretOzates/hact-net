from torchmetrics import F1Score, Accuracy
import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
import torch
from typing import Any


class HistoCartographyModel(pl.LightningModule):
    """
    Base class for HistoCartography models.
    """
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

        self.train_f1 = F1Score(task="multiclass", average="weighted", num_classes=3)
        self.train_detailed_f1 = F1Score(task="multiclass", average=None, num_classes=3)

        self.val_f1 = F1Score(task="multiclass", average="weighted", num_classes=3)
        self.val_detailed_f1 = F1Score(task="multiclass", average=None, num_classes=3)

        self.test_f1 = F1Score(task="multiclass", average="weighted", num_classes=3)
        self.test_detailed_f1 = F1Score(task="multiclass", average=None, num_classes=3)

        self.save_hyperparameters(kwargs)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.train_detailed_f1.reset()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        f1_0, f1_1, f1_2 = self.val_detailed_f1.compute()
        self.log_dict(
            {
                "val_f1_0": f1_0,
                "val_f1_1": f1_1,
                "val_f1_2": f1_2,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

        self.val_detailed_f1.reset()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        f1_0, f1_1, f1_2 = self.test_detailed_f1.compute()
        self.log_dict(
            {
                "test_f1_0": f1_0,
                "test_f1_1": f1_1,
                "test_f1_2": f1_2,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

        self.test_detailed_f1.reset()



    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        data = batch[:-1]
        labels = batch[-1]
        logits = self.forward(*data)
        loss = cross_entropy(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        self.train_f1(logits, labels)
        f1_0, f1_1, f1_2 = self.train_detailed_f1(logits, labels)
        self.log_dict(
            {
                "train_weighted_f1_score": self.train_f1,
                "train_f1_0": f1_0,
                "train_f1_1": f1_1,
                "train_f1_2": f1_2,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        """
        data = batch[:-1]
        labels = batch[-1]
        logits = self.forward(*data)
        loss = cross_entropy(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        self.val_f1(logits, labels)
        self.val_detailed_f1.update(logits, labels)
        self.log("val_weighted_f1_score", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        return {"val_loss": loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        """
        data = batch[:-1]
        labels = batch[-1]
        logits = self.forward(*data)
        loss = cross_entropy(logits, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        self.test_f1(logits, labels)
        self.test_detailed_f1.update(logits, labels)
        self.log("test_weighted_f1_score", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        return {"test_loss": loss, "logits": logits, "labels": labels}


    def configure_optimizers(self) -> Any:
        """
        Configure the optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=5e-4)
        return optimizer