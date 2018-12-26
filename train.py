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
import argparse


def content_loss(real,fake):
    loss =  K.mean(K.square(real - fake))
    return loss

# -----------------
# parameters
# -----------------

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--epoch", default=100000, type=int,
                    help="the number of epochs")
parser.add_argument("--save_interval", default=100, type=int,
                    help="the interval of snapshot")
parser.add_argument("--model_interval", default=5000, type=int,
                    help="the interval of savemodel")
parser.add_argument("--batch_size", default=4, type=int, help="batchsize")
parser.add_argument("--lam", default=10.0, type=float,
                    help="the weight of regularizer")

args = parser.parse_args()

lr_shape = (64,64,3)
hr_shape = (256,256,3)
batch_size = args.batch_size
save_interval = args.save_interval
model_interval = args.model_interval
_lambda = args.lam
epochs = args.epoch
lr_vgg = 2e-4
lr_D = 2e-4
lr_G = 2e-4
beta_1 = 0.5

optimizer = Adam(lr_vgg,beta_1)
gen = Generator(lr_shape)
dis = Discriminator(hr_shape)

VGG1,VGG2 = VGG(hr_shape)
VGG1.trainable = False
VGG2.trainable = False
VGG1.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
VGG2.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

# -----------------
# loss of generator
#
# perceptual loss = adversarial loss
#                    + contet loss
# -----------------

hr_image = Input(shape=hr_shape)
lr_image = Input(shape=lr_shape)
fake_hr_image = gen(lr_image)
fake_features1 = VGG1(fake_hr_image)
real_features1 = VGG1(hr_image)
fake_features2 = VGG2(fake_hr_image)
real_features2 = VGG2(hr_image)
validity = dis(fake_hr_image)

# -----------------
#  adversarial loss
# -----------------
loss_gen =  1e-3 * K.sum(softplus(-validity)) /batch_size
# -----------------
#  content loss
#
#  VGG19の層の出力の誤差
# -----------------
loss_gen += content_loss(real_features1,fake_features1)
loss_gen += content_loss(real_features2,fake_features2)

training_updates = Adam(lr=lr_G,beta_1=beta_1).get_updates(
    gen.trainable_weights,[],loss_gen)

gen_train = K.function([lr_image,hr_image],
                        [loss_gen],
                        training_updates)

# -------------------------
# compute grandient penalty
# -------------------------

delta_input = K.placeholder(shape=(None, 256,256,3))
alpha = K.random_uniform(
    shape=[batch_size, 1, 1, 1],
    minval=0.,
    maxval=1.
)

dis_mixed = Input(shape=hr_shape,
                  tensor=hr_image + delta_input)
loss_real = K.sum(softplus(-dis(hr_image)))/ batch_size
loss_fake = K.sum(softplus(validity))/ batch_size

dis_mixed_real = alpha * hr_image + ((1 - alpha) * dis_mixed)
grad_mixed = K.gradients(dis(dis_mixed_real), [dis_mixed_real])[0]
norm = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
grad_penalty = K.mean(K.square(norm - 1))

loss_dis = loss_fake + loss_real + _lambda * grad_penalty
# -----------------
# loss of discriminator
# -----------------

training_updates = Adam(lr=lr_D,beta_1=beta_1).get_updates(
    dis.trainable_weights,[],loss_dis)

dis_train = K.function([lr_image,hr_image, delta_input],
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
    delta = 0.5 * hr_images.std() * np.random.random(size=hr_images.shape)
    delta *= np.random.uniform(size=(batch_size, 1, 1, 1))
    errD_real,errD_fake = dis_train([lr_images,hr_images,delta])
    errD = errD_real - errD_fake
    # ------------------
    #  loss for Generator
    # ------------------
    errG, = gen_train([lr_images,hr_images])

    print("errD: {}, errG: {}".format(errD,errG))



    if epoch%save_interval == 0:

        # -----------------
        # save image
        # -----------------

        r, c = 2, 3
        hr_images, lr_images = Data_Loader(batchsize=2,train=False)
        fake_hr = gen.predict(lr_images)

        lr_images = 0.5 * lr_images + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        hr_images = 0.5 * hr_images + 0.5

        titles = ['Low resolution', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([lr_images,fake_hr, hr_images]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("result/srgan-{}.png".format(epoch))
        plt.close()
        if epoch % model_interval == 0:
            gen.save("srgan_model/model-{}-epoch.h5".format(epoch))
