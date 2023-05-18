from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from .components.adult_dataset import Adult

class AdultDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/adult",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset_kwargs = {
                'dir_path': data_dir}

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train: Optional[Dataset] = None
        self.test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            train = Adult(**self.dataset_kwargs, split='train')
            self.sv_train, self.usv_train = random_split(train, [15081, 15081], torch.Generator().manual_seed(42)) 
        else:
            self.test = Adult(**self.dataset_kwargs, split='test')

    def train_dataloader(self):
        sv_loader = DataLoader(
                dataset=self.sv_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True
                )
        usv_loader = DataLoader(
                dataset=self.usv_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True
                )
        return {'supervised': sv_loader, 'unsupervised': usv_loader}

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    amazon_datamodule = AdultDataModule(data_dir="./data/adult/")
    amazon_datamodule.setup(stage='fit')
    train_dataloader = amazon_datamodule.train_dataloader()
    batch = next(iter(train_dataloader['supervised']))
    print(batch[1].shape)
