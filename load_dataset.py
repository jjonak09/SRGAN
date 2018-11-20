import numpy as np
from PIL import Image
from scipy.misc import imresize
import matplotlib.pyplot as plt
from glob import glob
import scipy


def load_dataset(batch_size=1):
    path = glob('dataset/*.jpg')
    batch_images = np.random.choice(path, size=batch_size)
    hr_images = []
    lr_images = []

    for index, image_path in enumerate(path):
        img = np.array(Image.open(image_path)).astype(np.float)

        hr_image = imresize(img, (256, 256))
        lr_image = imresize(img, (64, 64))

        hr_images.append(hr_image)
        lr_images.append(lr_image)

    hr_images = np.array(hr_images) / 127.5 - 1
    lr_images = np.array(lr_images) / 127.5 - 1

    return hr_images, lr_images
