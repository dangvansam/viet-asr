import json
import os
import torch
import yaml
from typing import List


def load_config(path):
    with open(path) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args

def save_config(config: dict, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

def pad_list(xs: List[torch.Tensor], pad_value: float, max_len: int = 0) -> torch.Tensor:
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(max_len, max(x.size(0) for x in xs))
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

def load_json(path: str) -> dict:
    with open(path, encoding="utf8") as fp:
        data = json.load(fp)
    return data

def save_json(data: dict, path: str) -> None:
    data = json.dumps(data, indent=4, ensure_ascii=False, sort_keys=False)
    with open(path, mode="w", encoding="utf8") as fp:
        fp.write(data)
    return

def update_checkpoint(
        training_model: torch.nn.Module,
        checkpoint_dict: dict,
        save_folder_path: str,
        epoch: int,
        num_snapshots: int
    ) -> None:

    epoch += 1

    save_path = os.path.join(save_folder_path, "epoch_{}_state_dict.pt".format(epoch))
    checkpoint_dict[epoch] = save_path
    torch.save(training_model.state_dict(), save_path)
    print('\n\nSaved model at epoch {}\n'.format(epoch))

    delta = epoch - num_snapshots

    if delta in checkpoint_dict:
        old_checkpoint_path = checkpoint_dict[delta]
        os.system("rm {}".format(old_checkpoint_path))
        print('Removed old model checkpoint at epoch {}\n'.format(delta))