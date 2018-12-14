from __future__ import print_function, division
import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.activations import sigmoid,relu,softplus
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys,os
import numpy as np
from glob import glob
from PIL import Image
from scipy.misc import imresize
from srgan_model import Generator,Discriminator,VGG
from Dataset_Loader import Data_Loader


def content_loss(real,fake):
    w,h = real.shape[1],real.shape[2]
    loss = (1/int(w)*int(h)) * K.mean(K.square(real - fake), axis=-1)
    return loss
# -----------------
# parameters
# -----------------

lr_shape = (64,64,3)
hr_shape = (256,256,3)
batch_size = 6
save_interval = 100
model_interval = 5000
epochs = 100000
lr_D = 0.0002
lr_G = 0.0002
beta_1 = 0.5


optimizer = Adam(0.0002,0.5)

gen = Generator(lr_shape)
dis = Discriminator(hr_shape)

VGG = VGG(hr_shape)
VGG.trainable = False
VGG.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

hr_image = Input(shape=hr_shape)
lr_image = Input(shape=lr_shape)
fake_hr_image = gen(lr_image)
fake_features = VGG(fake_hr_image)
real_features = VGG(hr_image)
validity = dis(fake_hr_image)

loss_gen = content_loss(real_features,fake_features) + 0.001 * K.sum(softplus(-validity)) /batch_size

training_updates = Adam(lr=lr_G,beta_1=beta_1).get_updates(
    gen.trainable_weights,[],loss_gen)

gen_train = K.function([lr_image,hr_image],
                        [loss_gen],
                        training_updates)


loss_fake = K.sum(softplus(validity))/ batch_size
loss_real = K.sum(softplus(-dis(hr_image)))/ batch_size

loss_dis = loss_real + loss_fake

training_updates = Adam(lr=lr_D,beta_1=beta_1).get_updates(
    dis.trainable_weights,[],loss_dis)

dis_train = K.function([lr_image,hr_image],
                        [loss_real,loss_fake],
                        training_updates)


for epoch in range(epochs):

    # -----------------
    # load dataset
    # -----------------
    hr_images, lr_images = Data_Loader(batchsize=batch_size)
    # ----------------------
    #  loss for Discriminator
    # ---------------------
    errD_real,errD_fake = dis_train([lr_images,hr_images])
    errD = 0.5 * np.add(errD_real, errD_fake)
    # ------------------
    #  loss for Generator
    # ------------------
    errG, = gen_train([lr_images,hr_images])

    print("errD: {}, errG: {}".format(errD,errG[0][0]))



    if epoch%save_interval == 0:

        # -----------------
        # save image
        # -----------------
        test = []
        img = Image.open('dragan_image.jpg')
        img = img.convert('RGB')
        img = np.array(img).astype(np.float)
        test.append(img)
        test = np.array(test) / 127.5 - 1

        test_img = gen.predict(test)

        test_img = ((test_img - test_img.min()) * 255 / (test_img.max() - test_img.min())).astype(np.uint8)
        test_img = test_img.reshape(1, -1, 256, 256, 3).swapaxes(1,
                                                            2).reshape(1*256, -1, 3).astype(np.uint8)

        test_img = gen.predict(test)

        test_img = ((test_img - test_img.min()) * 255 / (test_img.max() - test_img.min())).astype(np.uint8)
        test_img = test_img.reshape(1, -1, 256, 256, 3).swapaxes(1,
                                                            2).reshape(1*256, -1, 3).astype(np.uint8)
        plt.imshow(Image.fromarray(test_img))
        filename = 'result' + str(epoch) + 'epoch.jpg'
        plt.imsave('result/' + filename ,test_img)
