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
        # Read input target
        target_path = self.targets_paths[index]
        target = mpimg.imread(target_path).astype(float)

        # Crop input target
        if self.crop_size > 0:
            target = trfs.random_crop(target, self.crop_size, self.crop_size)

        # Normalize input
        target = trfs.normalize(target)

        # Transform to tensor
        tensor_input = torch.as_tensor(np.array([noisy_input]), dtype=torch.float)
        tensor_target = torch.as_tensor(np.array([target]), dtype=torch.float)

        return tensor_noisy_input, tensor_target

if __name__ == '__main__':
    dataset = FeaturesPredictionDataset('dataset/Test.csv', crop_size=80)
    print(len(dataset))
