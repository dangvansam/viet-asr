import os
from typing import Union

import torch
from torch.optim import Adam, AdamW
from torch.utils.data.dataloader import DataLoader
from loguru import logger
import numpy as np

from utils import load_yaml
from vietasr.dataset.dataset import ASRDataset, ASRCollator
from vietasr.model.asr_model import ASRModel
from vietasr.utils.lr_scheduler import WarmupLR

class ASRTask():
    def __init__(self, config: Union[dict,str], output_dir: str=None, device: str="cpu") -> None:

        if isinstance(config, str):
            config = load_yaml(config)

        self.collate_fn = ASRCollator(bpe_model_path=self.config["dataset"]["bpe_model_path"])
        self.vocab = self.collate_fn.tokenizer.get_vocab()

        model = ASRModel(vocab_size=len(self.vocab), **config["model"])
        print(model)

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = config["train"]["output_dir"]

        self.config = config
        self.model = model
        self.device = torch.device("device")

    def train_one_epoch(self) -> float:

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
                logger.info(f"[TRAIN] EPOCH {self.epoch} | BATCH {i+1}/{num_batch} | loss={train_loss} | ctc_loss={ctc_loss} | decoder_loss={decoder_loss}")
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

    def valid_one_epoch(self) -> float:
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
                logger.info(f"[VALID] EPOCH {self.epoch} | BATCH {i+1}/{num_batch} | loss={valid_loss} | ctc_loss={ctc_loss} | decoder_loss={decoder_loss}")
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

    def load_checkpoint(self, pretrained_path: str):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epoch = int(checkpoint["epoch"])
        logger.success(f"Loaded checkpoint from: {pretrained_path}")

    def load_weight(self, weight_path: str):
        weight = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(weight["model"])
        self.model.to(self.device)
        logger.success(f"Loaded weight from: {weight_path}")

    def run_train(self):

        logger.info("="*40)
        logger.info(f"START TRAINING ASR MODEL")
        logger.info("="*40)
        logger.info(f"Config: {self.config}")

        self.num_epoch = self.config["train"]["num_epoch"]
        self.acc_steps = self.config["train"]["acc_steps"]
        self.batch_size = self.config["dataset"]["batch_size"]
        self.num_worker = self.config["dataset"]["num_worker"]
        self.epoch = 0
        self.valid_loss_best = 1000000

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["train"].get("lr", 1e-4),
            weight_decay=self.config["train"].get("weight_decay")
        )
        self.lr_scheduler = WarmupLR(
            self.optimizer,
            warmup_steps=self.config["train"].get("warmup_steps", 25000)
        )

        self.model.to(self.device)

        if self.config["train"].get("pretrained_path"):
            self.load_checkpoint(self.config["train"].get("pretrained_path"))

        self.train_dataset = ASRDataset(meta_filepath=self.config["dataset"]["train_filepath"])
        self.valid_dataset = ASRDataset(meta_filepath=self.config["dataset"]["valid_filepath"])

        os.makedirs(self.output_dir, exist_ok=True)

        valid_loss_best = self.valid_loss_best
        valid_acc_best = 0

        for epoch in range(self.epoch, self.num_epoch):
            self.epoch = epoch + 1
            logger.info(f"[TRAIN] EPOCH {epoch + 1}/{self.num_epoch} START")
            stats = self.train_one_epoch()
            logger.success(f"[TRAIN] STATS: {stats}")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "epoch": self.epoch,
                    "valid_loss_best": valid_loss_best
                },
                f"{self.output_dir}/checkpoint.pt")
            torch.save(self.model.state_dict(), f"{self.output_dir}/epoch_{self.epoch}.pt")
                
            logger.info(f"[TRAIN] EPOCH {epoch + 1}/{self.num_epoch} DONE, Save checkpoint to: {self.output_dir}/checkpoint_epoch_{epoch}.pt")

            logger.info(f"[VALID] EPOCH {epoch + 1}/{self.num_epoch} START")
            stats = self.valid_one_epoch()
            logger.success(f"[VALID] STATS: {stats}")
            valid_loss = stats["valid_loss"]

            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                torch.save(self.model.state_dict(), f"{self.output_dir}/valid_loss_best.pt")
                logger.success(f"saved best model to {self.output_dir}/valid_loss_best.pt")

            logger.info(f"[VALID] EPOCH {epoch + 1}/{self.num_epoch} DONE")

        logger.success(f"TRAINING ASR MODEL DONE!")

    def run_test(self, test_meta_filepath: str, model_path: str, device: str=None):
        
        logger.info("="*40)
        logger.info(f"START TESTING ASR MODEL")
        logger.info("="*40)
        logger.info(f"+ test_meta_filepath: {test_meta_filepath}")
        logger.info(f"+ model_path: {model_path}")
        logger.info(f"+ device: {device}")
        logger.info(f"+ Config: {self.config}")

        test_dataset = ASRDataset(test_meta_filepath)
        dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=self.num_worker,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )

        if device is not None:
            if device == "cpu":
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)

        self.load_weight(model_path)

        self.model.eval()

        test_loss_total = 0
        test_ctc_loss_total = 0
        test_decoder_loss_total = 0
        test_decoder_acc_total = 0

        num_batch = len(dataloader)

        # for batch in tqdm(dataloader, desc=f"[TRAIN] EPOCH {epoch}", unit="batch"):
        for i, batch in  enumerate(dataloader):
            batch = (b.to(self.device) for b in batch)
            retval = self.model(*batch)
            loss = retval["loss"]

            test_loss = loss.detach().item()
            test_loss_total += test_loss
            test_ctc_loss = retval["ctc_loss"].detach().item()
            test_ctc_loss_total += test_ctc_loss
            test_decoder_loss = retval["decoder_loss"].detach().item()
            test_decoder_loss_total += test_decoder_loss

            if (i + 1) % 100 == 0:
                logger.info(f"[VALID] EPOCH {self.epoch} | BATCH {i+1}/{num_batch} | loss={test_loss} | ctc_loss={test_ctc_loss} | decoder_loss={test_decoder_loss}")
                predicts = self.model.get_predicts(retval["encoder_out"], retval["encoder_out_lens"])
                labels = self.model.get_labels(batch[2], batch[3])
                logger.warning(f"+ Label  : {self.collate_fn.tokenizer.ids2text(labels[0])}")
                logger.warning(f"+ Predict: {self.collate_fn.tokenizer.ids2text(predicts[0])}")

        valid_stats = {
            "test_loss": test_loss_total / num_batch,
            "test_ctc_loss": test_ctc_loss_total / num_batch,
            "test_decoder_loss": test_decoder_loss_total / num_batch,
        }

    def setup_beamsearch(self):
        kenlm_path = self.config["decode"]["kenlm_path"]
        kenlm_alpha = self.config["decode"]["kenlm_alpha"]
        kenlm_beta = self.config["decode"]["kenlm_beta"]
        self.beam_size = self.config["decode"]["beam_size"]

        from pyctcdecode import build_ctcdecoder

        self.ctc_decoder = build_ctcdecoder(
            self.vocab,
            kenlm_model_path=kenlm_path,
            alpha=kenlm_alpha,
            beta=kenlm_beta
        )

    def ctc_beamsearch(self, input: torch.Tensor, length: torch.Tensor=None):
        encoder_out, _ = self.forward_encoder(input, length)
        logit = self.ctc.log_softmax(encoder_out)
        text = self.ctc_decoder.decode(
            logits=logit,
            beam_width=self.beam_size
        )
        return text

    def transcribe(self, input: Union[str, np.array, torch.Tensor]) -> str:
        if isinstance(input, str):
            import torchaudio
            input = torchaudio.load(input)[0]
        elif isinstance(input, np.array):
            input = torch.from_numpy(input)
        elif isinstance(input, torch.Tensor):
            input = input
        else:
            raise NotImplementedError
        text = self.ctc_beamsearch(input)
        return text

