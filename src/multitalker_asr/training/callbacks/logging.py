from lightning.pytorch.callbacks import Callback
from loguru import logger


class PrintLossCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", 0.0)
        val_loss = metrics.get("val_loss", 0.0)
        val_wer = metrics.get("val_wer", 1.0)
        logger.success(
            f"Epoch {trainer.current_epoch} End | "
            f"Train Loss: {float(train_loss):.4f} | "
            f"Val Loss: {float(val_loss):.4f} | "
            f"Val WER: {float(val_wer):.4f}"
        )
