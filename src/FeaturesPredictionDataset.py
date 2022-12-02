import torch
from torch.utils.data import Dataset
from matplotlib import image as mpimg
from os import listdir
import numpy as np
from random import uniform
import pandas as pd

import utils


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
        #input_plant = utils.rgb2gray(input_plant)

        # normalize input
        input_plant = utils.normalize(input_plant)

        # crop input target
        if self.crop_size > 0:
            input_plant = utils.random_crop(input_plant, self.crop_size, self.crop_size)

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
        tensor_target = torch.as_tensor(target, dtype=torch.float)

        return tensor_input, tensor_target

if __name__ == '__main__':
    from tqdm import tqdm
    print("start")
    dataset = FeaturesPredictionDataset('../dataset/Train.csv', crop_size=40)
    for i in range(10):
        for data in tqdm(dataset):
            pass
    print("end")
