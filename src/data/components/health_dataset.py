import os
from typing import List
import pandas as pd

import torch
from torch.utils.data import Dataset

class Health(Dataset):
    def __init__(
            self,
            dir_path: str, # ~/health
            split: str, # train or test
            idx: int # 1~5
    ) -> None:
        path = os.path.join(dir_path, f'health_{split}_{idx}.pkl')
        self.df = pd.read_pickle(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> List:
        item = self.df.iloc[index]
        y = torch.as_tensor(item.pop('label'), dtype=torch.float32).reshape(-1)
        s = torch.as_tensor(item.pop('age'), dtype=torch.float32).reshape(-1)
        x = torch.as_tensor(item, dtype=torch.float32)

        return [x, s, y]

