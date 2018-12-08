import numpy as np
from PIL import Image
from scipy.misc import imresize
import matplotlib.pyplot as plt
from glob import glob
import scipy


def Data_Loader(train =True,batchsize=1):

    if train:
        path = glob('train_dataset/*.jpg')
    else:
        path = glob('test_dataset/*.jpg')
    batch_images = np.random.choice(path, size=batchsize)
    hr_images = []
    lr_images = []

    for index, image_path in enumerate(batch_images):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = np.array(img).astype(np.float)

        hr_image = imresize(img, (256, 256))
        lr_image = imresize(img, (64, 64))

        hr_images.append(hr_image)
        lr_images.append(lr_image)

    hr_images = np.array(hr_images) / 127.5 - 1
    lr_images = np.array(lr_images) / 127.5 - 1

    return hr_images, lr_images


# train_hr, train_lr = Data_Loader(batchsize=8)
# print(train_hr.shape)
# print(train_lr.shape)

# test_hr, test_lr = Data_Loader(batchsize=1,train=False)
# print(test_hr.shape)
# print(test_lr.shape)
