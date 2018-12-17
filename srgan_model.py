from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add,Lambda
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
# from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import datetime
import matplotlib.pyplot as plt
import sys,os
import numpy as np
import keras.backend as K
import tensorflow as tf
conv_init = RandomNormal(0, 0.02)

# ---------------------------------------Generator--------------------------------------------------


def SubpixelConv2D(input_shape, scale=2):

    # Copyright (c) 2017 Jerry Liu
    # Released under the MIT license
    # https://github.com/twairball/keras-subpixel-conv/blob/master/LICENSE

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)

def Gen_ResBlock(layer_input,base):
        x = Conv2D(base,3,strides=1,padding="same",kernel_initializer=conv_init)(layer_input)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)
        x = Conv2D(base,3,strides=1,padding="same",kernel_initializer=conv_init)(x)
        x = BatchNormalization(momentum=0.8)(x)
        return Add()([x,layer_input])

# def CBR(layer_input,out_channel=256):
#     x = UpSampling2D(size=2)(layer_input)
#     x = Conv2D(out_channel,3,strides=1,padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation('relu')(x)
#     return x

def CBR(layer_input,out_channel=256):
    x = Conv2D(out_channel,3,strides=1,padding="same",kernel_initializer=conv_init)(layer_input)
    x = SubpixelConv2D(x)(x)
    x = Activation('relu')(x)
    return x


def Generator(input_shape,base=64,n_residual_blocks=3):

    input = Input(shape=input_shape)
    h1 = Conv2D(base,9,strides=1,padding="same",kernel_initializer=conv_init)(input)
    # h1 = BatchNormalization(momentum=0.8)(h1)
    h1 = Activation('relu')(h1)

    r = Gen_ResBlock(h1,base=base)
    for _ in range(n_residual_blocks - 1):
        r = Gen_ResBlock(r,base=base)

    h2 = Conv2D(base,3,strides=1,padding="same",kernel_initializer=conv_init)(r)
    h2 = BatchNormalization(momentum=0.8)(h2)
    h2 = Activation('relu')(h2)
    h2 = Add()([h2,h1])
    h3 = CBR(h2)
    h3 = CBR(h3)
    output = Conv2D(3,9,strides=1,padding="same",kernel_initializer=conv_init)(h3)

    return Model(inputs=input,outputs=output)


# ---------------------------------------Discriminator--------------------------------------------------


def Dis_ResBlock(layer_input, out_channel,kernel_size=3, strides=1, bn=True):
        d = Conv2D(out_channel, kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=conv_init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

def  Discriminator(input_shape,base=64):

    input = Input(shape=input_shape)
    h = input
    h = Dis_ResBlock(h,base,bn=False)
    h = Dis_ResBlock(h,base,strides=2)
    h = Dis_ResBlock(h,base*2)
    h = Dis_ResBlock(h,base*2,strides=2)
    h = Dis_ResBlock(h,base*4)
    h = Dis_ResBlock(h,base*4,strides=2)
    h = Dis_ResBlock(h,base*8)
    h = Dis_ResBlock(h,base*8,strides=2)
    h = Dense(base*16)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Flatten()(h)
    validity = Dense(1)(h)

    return Model(inputs=input,outputs=validity)


# ---------------------------------------VGG--------------------------------------------------


# def VGG(hr_input_shape):
#         vgg = VGG19(weights="imagenet")
#         vgg.outputs = [vgg.layers[9].output]
#         img = Input(shape=hr_input_shape)
#         img_features = vgg(img)
#         return Model(img, img_features)

def VGG(hr_input_shape):
        vgg = VGG19(weights="imagenet")
        img = Input(shape=hr_input_shape)

        vgg.outputs = [vgg.layers[9].output]
        img_features1 = vgg(img)
        model1 = Model(img, img_features1)

        vgg.outputs = [vgg.layers[14].output]
        img_features2 = vgg(img)
        model2 = Model(img, img_features2)

        return model1,model2



if __name__ == "__main__":

    hr_input_shape = (256,256,3)
    lr_input_shape = (64,64,3)
#     dis = Discriminator(hr_input_shape)
#     gen = Generator(lr_input_shape)
#     dis.summary()
#     gen.summary()
    vgg1,vgg2 = VGG(hr_input_shape)
    vgg1.summary()
    vgg2.summary()
