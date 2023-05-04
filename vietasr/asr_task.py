import os
from typing import Union

import torch
from torch.optim import Adam, AdamW
from torch.utils.data.dataloader import DataLoader
from loguru import logger
import numpy as np

from utils import load_config, save_config
from vietasr.dataset.dataset import ASRDataset, ASRCollator
from vietasr.model.asr_model import ASRModel
from vietasr.utils.lr_scheduler import WarmupLR
from vietasr.utils.utils import calculate_wer

class ASRTask():
    def __init__(self, config: str, output_dir: str=None, device: str="cpu") -> None:

        config = load_config(config)

        self.collate_fn = ASRCollator(bpe_model_path=config["dataset"]["bpe_model_path"])
        self.vocab = self.collate_fn.get_vocab()
        model = ASRModel(vocab_size=len(self.vocab), pad_id=self.collate_fn.pad_id, **config["model"])
        # print(model)

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = config["train"]["output_dir"]

        self.device = torch.device(device)

        self.config = config
        self.model = model
        self.ctc_decoder = None
        self.optimizer = None
        self.lr_scheduler = None
        self.epoch = None

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
            
            batch = [b.to(self.device) for b in batch]

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
                logger.warning(f"+ Label  : {self.collate_fn.ids2text(labels[0])}")
                logger.warning(f"+ Predict: {self.collate_fn.ids2text(predicts[0])}")

        train_stats = {
            "train_loss": train_loss_epoch / num_batch,
            "train_ctc_loss": ctc_loss_epoch / num_batch,
            "train_decoder_loss": decoder_loss_epoch / num_batch,
        }
        return train_stats

    def valid_one_epoch(self) -> float:
        valid_loss_epoch = 0
        ctc_loss_epoch = 0
        decoder_loss_epoch = 0
        decoder_acc = 0
        predicts = []
        labels = []

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
            batch = [b.to(self.device) for b in batch]
            with torch.no_grad():
                retval = self.model(*batch)
            loss = retval["loss"]

            valid_loss = loss.detach().item()
            valid_loss_epoch += valid_loss
            ctc_loss = retval["ctc_loss"].detach().item()
            ctc_loss_epoch += ctc_loss
            decoder_loss = retval["decoder_loss"].detach().item()
            decoder_loss_epoch += decoder_loss

            predict = self.model.get_predicts(retval["encoder_out"], retval["encoder_out_lens"])
            label = self.model.get_labels(batch[2], batch[3])
            predict_str = [self.collate_fn.ids2text(x) for x in predict]
            label_str = [self.collate_fn.ids2text(x) for x in label]
            predicts += predict_str
            labels += label_str
            
            if (i + 1) % 100 == 0:
                logger.info(f"[VALID] EPOCH {self.epoch} | BATCH {i+1}/{num_batch} | loss={valid_loss} | ctc_loss={ctc_loss} | decoder_loss={decoder_loss}")
                logger.warning(f"+ Label  : {label_str[0]}")
                logger.warning(f"+ Predict: {predict_str[0]}")

        valid_stats = {
            "valid_loss": valid_loss_epoch / num_batch,
            "vaid_ctc_loss": ctc_loss_epoch / num_batch,
            "vaid_decoder_loss": decoder_loss_epoch / num_batch,
            "valid_wer": calculate_wer(predicts, labels),
            "valid_cer": calculate_wer(predicts, labels, use_cer=True)
        }
        return valid_stats

    def load_checkpoint(self, pretrained_path: str):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        if checkpoint.get("optimizer") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("lr_scheduler") and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if checkpoint.get("epoch") and self.epoch:
            self.epoch = int(checkpoint["epoch"])
        logger.success(f"Loaded checkpoint from: {pretrained_path}")

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

        save_config(self.config, os.path.join(self.output_dir, "config.yaml"))

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
            torch.save({"model": self.model.state_dict()}, f"{self.output_dir}/epoch_{self.epoch}.pt")
                
            logger.info(f"[TRAIN] EPOCH {epoch + 1}/{self.num_epoch} DONE, Save checkpoint to: {self.output_dir}/checkpoint_epoch_{epoch}.pt")

            logger.info(f"[VALID] EPOCH {epoch + 1}/{self.num_epoch} START")
            stats = self.valid_one_epoch()
            logger.success(f"[VALID] STATS: {stats}")
            valid_loss = stats["valid_loss"]

            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                torch.save({"model": self.model.state_dict()}, f"{self.output_dir}/valid_loss_best.pt")
                logger.success(f"saved best model to {self.output_dir}/valid_loss_best.pt")

            logger.info(f"[VALID] EPOCH {epoch + 1}/{self.num_epoch} DONE")

        logger.success(f"TRAINING ASR MODEL DONE!")

    def run_test(
            self,
            test_meta_filepath: str,
        ):
        
        logger.info("="*40)
        logger.info(f"START TESTING ASR MODEL")
        logger.info("="*40)
        logger.info(f"+ test_meta_filepath: {test_meta_filepath}")
        logger.info(f"+ device: {self.device}")
        logger.info(f"+ Config: {self.config}")
        
        batch_size = self.config["dataset"]["batch_size"]
        num_worker = self.config["dataset"]["num_worker"]
        
        test_dataset = ASRDataset(test_meta_filepath)
        dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_worker,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )

        test_loss_total = 0
        test_ctc_loss_total = 0
        test_decoder_loss_total = 0
        test_decoder_acc_total = 0
        predicts = []
        labels = []
        
        num_batch = len(dataloader)

        # for batch in tqdm(dataloader, desc=f"[TRAIN] EPOCH {epoch}", unit="batch"):
        for i, batch in  enumerate(dataloader):
            batch = [b.to(self.device) for b in batch]
            with torch.no_grad():
                retval = self.model(*batch)
            loss = retval["loss"]

            test_loss = loss.detach().item()
            test_loss_total += test_loss
            test_ctc_loss = retval["ctc_loss"].detach().item()
            test_ctc_loss_total += test_ctc_loss
            test_decoder_loss = retval["decoder_loss"].detach().item()
            test_decoder_loss_total += test_decoder_loss

            if self.ctc_decoder is not None:
                encoder_out = retval["encoder_out"]
                encoder_out_lens = retval["encoder_out_lens"]
                predict_str = []
                for j in range(encoder_out.shape[0]):
                    predict_str.append(self.ctc_beamsearch(encoder_out[j].unsqueeze(0), encoder_out_lens[j].unsqueeze(0)))
            else:
                predict = self.model.get_predicts(retval["encoder_out"], retval["encoder_out_lens"])
                predict_str = [self.collate_fn.ids2text(x) for x in predict]
            
            predicts += predict_str

            label = self.model.get_labels(batch[2], batch[3])
            label_str = [self.collate_fn.ids2text(x) for x in label]
            labels += label_str
            
            if (i + 1) % 10 == 0:
                logger.info(f"[TEST] BATCH {i+1}/{num_batch} | loss={test_loss} | ctc_loss={test_ctc_loss} | decoder_loss={test_decoder_loss}")
                logger.warning(f"+ Label  : {label_str[0]}")
                logger.warning(f"+ Predict: {predict_str[0]}")
        
        wer = calculate_wer(predicts, labels)
        cer = calculate_wer(predicts, labels, use_cer=True)
        
        logger.success(f"Test set: {test_meta_filepath} done.")
        logger.success(f" + CER={cer}%")
        logger.success(f" + WER={wer}%")

    def setup_beamsearch(
        self,
        kenlm_path: str=None,
        word_vocab_path: str=None,
        kenlm_alpha: float=None,
        kenlm_beta: float=None,
        beam_size: int=2,
    ):
        if not kenlm_path:
            kenlm_path = self.config["decode"].get("kenlm_path")
        if not word_vocab_path:
            word_vocab_path = self.config["decode"].get("word_vocab_path")
        if not kenlm_alpha:
            kenlm_alpha = self.config["decode"].get("kenlm_alpha")
        if not kenlm_beta:
            kenlm_beta = self.config["decode"].get("kenlm_beta")
        if not beam_size:
            beam_size = self.config["decode"].get("beam_size")

        if beam_size > 1 and not kenlm_path:
            logger.error(f"must pass --kenlm_path (or set in config file) for language model, if beamsize > 1")
            exit()

        self.beam_size = beam_size

        from pyctcdecode import build_ctcdecoder

        self.ctc_decoder = build_ctcdecoder(
            self.vocab,
            kenlm_model_path=kenlm_path,
            unigrams=word_vocab_path,
            alpha=kenlm_alpha,
            beta=kenlm_beta
        )
        logger.success("Setup ctc decoder done")

    def ctc_beamsearch(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
        )->str:
        
        encoder_out = encoder_out[:, : encoder_out_lens[0], :]
        assert len(encoder_out.shape) == 3, encoder_out.shape
        assert encoder_out.shape[0] == 1, encoder_out.shape
        assert encoder_out.shape[1] == encoder_out_lens[0]

        with torch.no_grad():
            logit = self.model.ctc.log_softmax(encoder_out).detach().cpu().squeeze(0).numpy()
        text = self.ctc_decoder.decode(
            logits=logit,
            beam_width=self.beam_size
        )
        # remove <blank> from text
        text = text.replace("<blank>", "").strip()
        return text

    def transcribe(self, _input: Union[str, np.array, torch.Tensor]) -> str:
        if isinstance(_input, str):
            import librosa
            _input = librosa.load(_input, sr=16000, mono=True)[0]
            _input = torch.from_numpy(_input)
        elif isinstance(_input, np.array):
            _input = torch.from_numpy(_input)
        elif isinstance(_input, torch.Tensor):
            _input = _input
        else:
            raise NotImplementedError
        if len(_input.shape) == 1:
            _input = _input.unsqueeze(0)

        length = torch.Tensor([_input.shape[1]]).long()

        _input = _input.to(self.device)
        length = length.to(self.device)

        # get encoder out
        with torch.no_grad():
            self.model.feature_extractor.eval()
            if self.model.pos_encoder is not None:
                self.model.pos_encoder.eval()
            if self.model.subsampling is not None:
                self.model.subsampling.eval()
            self.model.ctc.eval()
            encoder_out, encoder_out_lens = self.model.forward_encoder(_input, length)

        # beamsearch decode
        if self.ctc_decoder is not None:
            text = self.ctc_beamsearch(encoder_out, encoder_out_lens)
        else:
            # greedy decode
            ids = self.model.get_predicts(encoder_out, encoder_out_lens)[0]
            text = self.collate_fn.ids2text(ids)
        return text

