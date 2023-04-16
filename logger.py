import os
from torch.utils.tensorboard import SummaryWriter


def prepare_logdir(folder_log: str):
    if os.path.isdir(folder_log):
        os.system('rm -r {}'.format(folder_log))

    os.system("mkdir -p {}".format(folder_log))
    return 



class S2T_Logger(SummaryWriter):
    def __init__(self, logdir: str):
        super(S2T_Logger, self).__init__(logdir)

    def log_done_epoch(
            self, 
            train_loss: float, 
            valid_loss: float, 
            att_acc: float, 
            epoch: int
        ) -> None:

        self.add_scalar("Train_loss/epochs", train_loss, epoch)
        self.add_scalar("Valid_loss/epochs", valid_loss, epoch)
        self.add_scalar("Attention_Acc/epochs", att_acc, epoch)


    def log_weights(self, model, epoch):

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), epoch)