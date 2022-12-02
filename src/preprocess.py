import numpy as np
import pandas as pd
import os
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from utils import rgb2gray, random_crop
from PIL import Image


# parameters
widthBoundary = 4900
heightBoundary = 3200
datasetPaths = ['../dataset/Train', '../dataset/Test']

if not os.path.exists('../dataset/preprocessed'):
    os.mkdir('../dataset/preprocessed')
for datasetPath in datasetPaths:
    OutputsPath = '../dataset/preprocessed/' + os.path.basename(datasetPath)
    if not os.path.exists(OutputsPath):
        os.mkdir(OutputsPath)

    for directory in os.listdir(datasetPath):
        if not os.path.exists(OutputsPath + '/' + directory):
            os.mkdir(OutputsPath + '/' + directory)
        inputsPath = datasetPath + '/' + directory
        for img in os.listdir(inputsPath):
            imPath = inputsPath + '/' + img
            output = rgb2gray((mpimg.imread(imPath).astype(float)))
            dim = output.shape
            if (widthBoundary < dim[0]) and (heightBoundary < dim[1]):
                outputPath = OutputsPath + '/' + directory + '/' + img
                output = random_crop(output, widthBoundary, heightBoundary)
                output = (((output - output.min()) / (output.max() - output.min())) * 2**8).astype(np.uint8)
                output = Image.fromarray(output)
                output.save(outputPath)
