from random import randint
import numpy as np

import transformers as trfs


def random_crop(image, y_dim, x_dim):
    y_top = randint(0, image.shape[0] - y_dim)
    y_bot = y_top + y_dim
    x_left = randint(0, image.shape[1] - x_dim)
    x_right = x_left + x_dim
    return image[y_top:y_bot, x_left:x_right]


def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib import image as mpimg


    image = mpimg.imread("../dataset/Test/amborella/amborella071.jpg")
    plt.figure(0)
    plt.imshow(image)
    cropped_image = random_crop(image, 82, 82)
    print(image.shape, cropped_image.shape)
    plt.figure(1)
    plt.imshow(cropped_image)
    plt.show()
