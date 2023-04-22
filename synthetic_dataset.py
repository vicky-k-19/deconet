import torch
import torch.nn
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import ToTensor


class Synthetic(Dataset):
    def __init__(self, input_vec) -> None:

        self.input_vec = input_vec.t()

    def __len__(self):

        return self.input_vec.shape[0]

    def __getitem__(self, index):
        x = self.input_vec[index]

        return x


def synthetic_generator(input_dim, set_size):
    input_vectors = torch.normal(0, 0.01, size=(input_dim, set_size))

    return input_vectors
