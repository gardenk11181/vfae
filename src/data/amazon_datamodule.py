from typing import Any, Dict, Optional, Tuple, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from .components.amazon_dataset import Amazon

class AmazonDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        source: str = "books",
        target: str = "dvd",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset_kwargs = {
                'dir_path': data_dir,
                'source': source,
                'target': target}

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.source_train: Optional[Dataset] = None
        self.target_train: Optional[Dataset] = None
        self.target_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.source_train = Amazon(**self.dataset_kwargs, idx=0)
            self.target_train = Amazon(**self.dataset_kwargs, idx=1)
        else:
            self.target_test = Amazon(**self.dataset_kwargs, idx=2)

    def train_dataloader(self):
        source_train_loader = DataLoader(
                dataset=self.source_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True
                )
        target_train_loader = DataLoader(
                dataset=self.target_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                )
        return {'source_train': source_train_loader, 'target_train': target_train_loader}

    def test_dataloader(self):
        return DataLoader(
            dataset=self.target_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    amazon_datamodule = AmazonDataModule(data_dir="./data/amazon/")
    amazon_datamodule.setup(stage='fit')
    train_dataloader = amazon_datamodule.train_dataloader()
    source_train_loader = train_dataloader['source_train']
    batch = next(iter(source_train_loader))
    print(batch[0].shape)
