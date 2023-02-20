import os
from typing import List
import pandas as pd

import torch
from torch.utils.data import Dataset

class Amazon(Dataset):
    def __init__(
            self,
            dir_path: str, # ~/amazon
            source: str, # source domain name
            target: str, # target domain name
            idx: int # 0: source train, 1: target train, 2: target test
    ) -> None:
        path = os.path.join(dir_path, f'{source}_to_{target}.pkl')
        dataframes = pd.read_pickle(path)[idx]
        if idx!= 1:
            self.x = pd.DataFrame(dataframes[0].todense())
            self.y = dataframes[1]
        else:
            self.x = pd.DataFrame(dataframes.todense())
            self.y = None

        if idx == 0:
            self.s = torch.zeros(0, dtype=torch.float32)
        else:
            self.s = torch.zeros(1, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index) -> List:
        x = torch.as_tensor(self.x.iloc[index], dtype=torch.float32)
        if self.y is not None:
            y = torch.as_tensor(self.y.iloc[index], dtype=torch.float32)
            return [x, self.s, y]

        return [x, self.s]

