from typing import Union
import torch

from torch import nn
from torch import Tensor

from torch.nn import CrossEntropyLoss, CTCLoss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from torchaudio.transforms import RNNTLoss


class SmoothCTCLoss(nn.Module):

    def __init__(
            self, 
            num_classes: int, 
            blank: int = 0, 
            weight: float = 0.00
        ):

        super(SmoothCTCLoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes

        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(
            self, 
            log_probs: Tensor, 
            targets: Tensor, 
            input_lengths: Tensor, 
            target_lengths: Tensor
        ) -> Tensor:

        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        if self.weight != 0:
            kl_inp = log_probs.transpose(0, 1)
            kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
            kl_tar = nn.functional.softmax(kl_tar, dim= 2)
            kldiv_loss = self.kldiv(kl_inp, kl_tar)

            #print(ctc_loss, kldiv_loss)
            loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        else:
            loss = ctc_loss

        return loss


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(
            self, 
            classes: int, 
            smoothing: float = 0.05, 
            dim: int = 1):

        super(SmoothCrossEntropyLoss, self).__init__()

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler
    This scheduler is almost same as NoamLR Scheduler except for following difference:
    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    Note that the maximum lr equals to optimizer.lr in this scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
            for lr in self.base_lrs
        ]


def test_WarumupLR(step):
    linear = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(linear.parameters(), 0.001)
    sch = WarmupLR(opt, step)
    lr = opt.param_groups[0]["lr"]

    opt.step()
    sch.step()
    lr2 = opt.param_groups[0]["lr"]
    print(lr, lr2)
    assert lr != lr2
