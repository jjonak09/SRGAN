from __future__ import print_function, division
import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys,os
import numpy as np
from glob import glob
from PIL import Image
from scipy.misc import imresize
from srgan_model import Generator,Discriminator,VGG
from Dataset_Loader import Data_Loader

# -----------------
# parameters
# -----------------

lr_shape = (64,64,3)
hr_shape = (256,256,3)
batch_size = 5
save_interval = 100
model_interval = 1000
epochs = 100000


optimizer = Adam(0.0002,0.5)

VGG = VGG(hr_shape)
VGG.trainable = False
VGG.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

dis = Discriminator(hr_shape)
dis.compile(loss = 'mse',
            optimizer=optimizer,
            metrics=['accuracy'])

gen = Generator(lr_shape)
hr_image = Input(shape=hr_shape)
lr_image = Input(shape=lr_shape)
fake_hr_image = gen(lr_image)
fake_features = VGG(fake_hr_image)

dis.trainable  = False
validity = dis(fake_hr_image)
combined = Model([lr_image,hr_image],
                    [validity,fake_features])

combined.compile(loss=['binary_crossentropy','mse'],
                        loss_weights=[1e-3,1],
                        optimizer=optimizer)

for epoch in range(epochs):

    # -----------------
    # load dataset
    # -----------------

    hr_images, lr_images = Data_Loader(batchsize=batch_size)

    hr_fake = gen.predict(lr_images)
    valid = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))

    # ----------------------
    #  loss for Discriminator
    # ----------------------

    errD_real = dis.train_on_batch(hr_images,valid)
    errD_fake = dis.train_on_batch(hr_fake,fake)
    errD = 0.5 * np.add(errD_real, errD_fake)

    # ------------------
    #  loss for Generator
    # ------------------

    valid = np.ones((batch_size,1))
    image_features = VGG.predict(hr_images)

    errG = combined.train_on_batch([lr_images,hr_images],
                                    [valid,image_features])

    print("errD: {}, errG: {}".format(errD[0],errG[0]))



    if epoch%save_interval == 0:

        # -----------------
        # save image
        # -----------------

        r, c = 2, 2
        hr_images, lr_images = Data_Loader(batchsize=2,train=False)
        fake_hr = gen.predict(lr_images)

        lr_images = 0.5 * lr_images + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        hr_images = 0.5 * hr_images + 0.5

        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, hr_images]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("result/srgan-{}.png".format(epoch))
        plt.close()

    # if epoch%model_interval == 0:
    #     gen.save("srgan_model/model-{}epoch.h5".format(epoch))
