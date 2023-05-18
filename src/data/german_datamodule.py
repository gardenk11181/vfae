from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from .components.german_dataset import German

class GermanDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/german",
        idx: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset_kwargs = {
                'dir_path': data_dir,
                'idx': idx}

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train: Optional[Dataset] = None
        self.test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train = German(**self.dataset_kwargs, split='train')
        else:
            self.test = German(**self.dataset_kwargs, split='test')

    def train_dataloader(self):
        return DataLoader(
                dataset=self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True
                )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    amazon_datamodule = GermanDataModule(data_dir="./data/german/")
    amazon_datamodule.setup(stage='fit')
    train_dataloader = amazon_datamodule.train_dataloader()
    batch = next(iter(train_dataloader))
    print(batch[0].shape)
