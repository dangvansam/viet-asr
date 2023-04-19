from torch.utils.data.dataloader import DataLoader
from vietasr.dataset.dataset import ASRDataset, ASRCollator

if __name__=='__main__':
    dataset = ASRDataset(meta_filepath="data/train.txt")
    collate_fn = ASRCollator(bpe_model_path="data/bpe_2000/bpe.model")

    dataloader = DataLoader(
        dataset=dataset,
        num_workers=1,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    for i, x in enumerate(dataloader):
        print(x[0].shape)
        print(x[1].shape)
        print(x[2].shape)
        print(x[3].shape)
        # break