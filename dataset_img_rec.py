import torch
import torch.nn as nn
import numpy as np

class dataset_img_rec(torch.utils.data.Dataset):
    def __init__(self, array_value:np.ndarray[int|float]):
        self.input_value = torch.tensor(array_value, dtype=torch.float32)
        self.target_value = torch.tensor(array_value, dtype=torch.float32)
    def __len__(self):
        return len(self.input_value)
    def __getitem__(self, index):
        return {
            "input": self.input_value[index],
            "target": self.target_value[index]
        }

    