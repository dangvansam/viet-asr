import argparse
import os
from typing import Tuple, Union

from tqdm import tqdm
import torch
from torch.optim import Adam, AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from loguru import logger

from utils import load_yaml
from vietasr.dataset.dataset import ASRDataset, ASRCollator
from vietasr.model.asr_model import ASRModel
from vietasr.utils.lr_scheduler import WarmupLR

class ASRTrainer():
    def __init__(self, config: dict, output_dir: str=None) -> None:

        self.train_dataset = ASRDataset(meta_filepath=config["dataset"]["train_filepath"])
        self.valid_dataset = ASRDataset(meta_filepath=config["dataset"]["valid_filepath"])
        self.collate_fn = ASRCollator(bpe_model_path=config["dataset"]["bpe_model_path"])

        model = ASRModel(vocab_size=2000, **config["model"])

        optimizer = AdamW(
            model.parameters(),
            lr=config["train"].get("lr", 1e-4),
            weight_decay=config["train"].get("weight_decay")
        )
        lr_scheduler = WarmupLR(
            optimizer,
            warmup_steps=config["train"].get("warmup_steps", 25000)
        )

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = config["train"]["output_dir"]

        if config["train"]["device"] == "gpu":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        model.to(self.device)

        self.num_epoch = config["train"]["num_epoch"]
        self.acc_steps = config["train"]["acc_steps"]
        self.batch_size = config["dataset"]["batch_size"]
        self.num_worker = config["dataset"]["num_worker"]
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.epoch = 0
        self.valid_loss_best = 1000000

        if config["train"].get("pretrained_path"):
            self.load_pretrained(config["train"].get("pretrained_path"))

    def train_one_epoch(self, epoch: int) -> float:

        self.model.train()

        train_loss_epoch = 0
        ctc_loss_epoch = 0
        decoder_loss_epoch = 0
        decoder_acc = 0

        dataloader = DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_worker,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=self.collate_fn
        )

        num_batch = len(dataloader)
        self.optimizer.zero_grad()

        # for batch in tqdm(dataloader, desc=f"[TRAIN] EPOCH {epoch}", unit="batch"):
        for i, batch in enumerate(dataloader):
            
            batch = (b.to(self.device) for b in batch)

            retval = self.model(*batch)
            loss = retval["loss"]
            loss = loss / self.acc_steps
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0) 

            if (i + 1) % self.acc_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

            train_loss = loss.detach().item()
            train_loss_epoch += train_loss
            ctc_loss = retval["ctc_loss"].detach().item()
            ctc_loss_epoch += ctc_loss
            decoder_loss = retval["decoder_loss"].detach().item()
            decoder_loss_epoch += decoder_loss

            # num_words_decoder += batch[3].shape[1] * batch[3].shape[0]
            # decoder_acc += (retval["decoder_out"].argmax(1) == batch[3]).sum().item()
            
            if (i + 1) % 100 == 0:
                logger.info(f"[TRAIN] EPOCH {epoch} | BATCH {i+1}/{num_batch} | loss={train_loss} | ctc_loss={ctc_loss} | decoder_loss={decoder_loss}")
                predicts = self.model.get_predicts(retval["encoder_out"], retval["encoder_out_lens"])
                labels = self.model.get_labels(batch[2], batch[3])
                logger.warning(f"+ Label  : {self.collate_fn.tokenizer.ids2text(labels[0])}")
                logger.warning(f"+ Predict: {self.collate_fn.tokenizer.ids2text(predicts[0])}")

        train_stats = {
            "train_loss": train_loss_epoch / num_batch,
            "train_ctc_loss": ctc_loss_epoch / num_batch,
            "train_decoder_loss": decoder_loss_epoch / num_batch,
        }
        return train_stats

    def valid_one_epoch(self, epoch: int) -> float:
        self.model.eval()

        valid_loss_epoch = 0
        ctc_loss_epoch = 0
        decoder_loss_epoch = 0
        decoder_acc = 0

        dataloader = DataLoader(
            dataset=self.valid_dataset,
            num_workers=self.num_worker,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        num_batch = len(dataloader)

        # for batch in tqdm(dataloader, desc=f"[TRAIN] EPOCH {epoch}", unit="batch"):
        for i, batch in  enumerate(dataloader):
            batch = (b.to(self.device) for b in batch)
            retval = self.model(*batch)
            loss = retval["loss"]

            valid_loss = loss.detach().item()
            valid_loss_epoch += valid_loss
            ctc_loss = retval["ctc_loss"].detach().item()
            ctc_loss_epoch += ctc_loss
            decoder_loss = retval["decoder_loss"].detach().item()
            decoder_loss_epoch += decoder_loss

            if (i + 1) % 100 == 0:
                logger.info(f"[VALID] EPOCH {epoch} | BATCH {i+1}/{num_batch} | loss={valid_loss} | ctc_loss={ctc_loss} | decoder_loss={decoder_loss}")
                predicts = self.model.get_predicts(retval["encoder_out"], retval["encoder_out_lens"])
                labels = self.model.get_labels(batch[2], batch[3])
                logger.warning(f"+ Label  : {self.collate_fn.tokenizer.ids2text(labels[0])}")
                logger.warning(f"+ Predict: {self.collate_fn.tokenizer.ids2text(predicts[0])}")

        valid_stats = {
            "valid_loss": valid_loss_epoch / num_batch,
            "vaid_ctc_loss": ctc_loss_epoch / num_batch,
            "vaid_decoder_loss": decoder_loss_epoch / num_batch,
        }
        return valid_stats

    def load_pretrained(self, pretrained_path: str):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epoch.load_state_dict(checkpoint["epoch"])
        logger.success(f"Loaded checkpoint from: {pretrained_path}")

    def run(self, ):
        
        logger.info("="*40)
        logger.info(f"START TRAINING ASR MODEL")
        logger.info("="*40)
        logger.info(f"Config: {self.config}")

        os.makedirs(self.output_dir, exist_ok=True)

        valid_loss_best = self.valid_loss_best
        valid_acc_best = 0

        for epoch in range(self.epoch, self.num_epoch + 1):
            logger.info(f"[TRAIN] EPOCH {epoch + 1}/{self.num_epoch} START")
            stats = self.train_one_epoch(epoch + 1)
            logger.success(f"[TRAIN] STATS: {stats}")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "valid_loss_best": valid_loss_best
                },
                f"{self.output_dir}/checkpoint_epoch_{epoch}.pt")
                
            logger.info(f"[TRAIN] EPOCH {epoch + 1}/{self.num_epoch} DONE, Save checkpoint to: {self.output_dir}/checkpoint_epoch_{epoch}.pt")

            logger.info(f"[VALID] EPOCH {epoch + 1}/{self.num_epoch} START")
            stats = self.valid_one_epoch(epoch + 1)
            logger.success(f"[VALID] STATS: {stats}")
            valid_loss = stats["valid_loss"]

            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                torch.save(self.model.state_dict(), f"{self.output_dir}/valid_loss_best.pt")
                logger.success(f"saved best model to {self.output_dir}/valid_loss_best.pt")

            logger.info(f"[VALID] EPOCH {epoch + 1}/{self.num_epoch} DONE")

        logger.success(f"TRAINING ASR MODEL DONE!")