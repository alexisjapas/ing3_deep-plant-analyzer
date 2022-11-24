import torch
from torch.utils.data import Dataset
from matplotlib import image as mpimg
from os import listdir
import numpy as np
from random import uniform
import pandas as pd

import transformers as trfs


class FeaturesPredictionDataset(Dataset):
    def __init__(self, annotations_file_path: str, crop_size: int):
        self.data = pd.read_csv(annotations_file_path)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        # INPUT
        # read input
        input_path = self.data["Chemin"].iloc[index]
        input_plant = mpimg.imread(input_path).astype(float)

        # crop input target
        if self.crop_size > 0:
            input_plant = trfs.random_crop(input_plant, self.crop_size, self.crop_size)

        # normalize input
        input_plant = trfs.normalize(input_plant)

        # transform to tensor
        tensor_input = torch.as_tensor(np.array([input_plant]), dtype=torch.float)

        # TARGET
        # read target
        target = np.array([
            self.data["Bord"].iloc[index],
            self.data["Phyllotaxie"].iloc[index],
            self.data["Type_feuille"].iloc[index],
            self.data["Ligneux"].iloc[index],
        ])
        tensor_target = torch.as_tensor(target)

        return tensor_input, tensor_target

if __name__ == '__main__':
    dataset = FeaturesPredictionDataset('../dataset/Test.csv', crop_size=80)
    print(dataset[1])
