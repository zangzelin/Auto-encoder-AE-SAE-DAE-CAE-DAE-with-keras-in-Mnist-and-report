# Denoise Autoencoder with 3 fully connetion layer.
# time: 2019-4-22
# author: Zelin Zang
# email: zangzelin@gmail.com
# github: https://github.com/zangzelin/

import numpy as np

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import os

import ae


class DAE(ae.AE):

    def __init__(
        self,
        ENCODING_DIM_INPUT=784,
        ENCODING_DIM_LAYER1=128,
        ENCODING_DIM_LAYER2=64,
        ENCODING_DIM_LAYER3=10,
        ENCODING_DIM_OUTPUT=2,
        Name="dae"
    ):

        self.ENCODING_DIM_INPUT = ENCODING_DIM_INPUT
        self.ENCODING_DIM_OUTPUT = ENCODING_DIM_OUTPUT
        self.name = Name

        # input placeholder
        input_image = Input(shape=(ENCODING_DIM_INPUT, ))

        # encoding layer
        encode_layer1 = Dense(ENCODING_DIM_LAYER1,
                              activation='relu')(input_image)
        encode_layer2 = Dense(ENCODING_DIM_LAYER2,
                              activation='relu')(encode_layer1)
        encode_layer3 = Dense(ENCODING_DIM_LAYER3,
                              activation='relu')(encode_layer2)
        encode_output = Dense(ENCODING_DIM_OUTPUT)(encode_layer3)

        # decoding layer
        decode_layer1 = Dense(ENCODING_DIM_LAYER3,
                              activation='relu')(encode_output)
        decode_layer2 = Dense(ENCODING_DIM_LAYER2,
                              activation='relu')(decode_layer1)
        decode_layer3 = Dense(ENCODING_DIM_LAYER1,
                              activation='relu')(decode_layer2)
        decode_output = Dense(ENCODING_DIM_INPUT,
                              activation='tanh')(decode_layer3)

        # build surprised learning model
        SL_output = Dense(10, activation='softmax')(encode_output)

        # build autoencoder, encoder
        autoencoder = Model(inputs=input_image, outputs=decode_output)
        encoder = Model(inputs=input_image, outputs=encode_output)
        SL_model = Model(inputs=input_image, outputs=SL_output)

        # compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        SL_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.sl_model = SL_model

    def SlTraining(self, x, y, EPOCHS=10):

        self.history_record =  self.sl_model.fit(x, y, epochs=EPOCHS)

    def addNoise(self, x_train, x_test, NOISE_FACTOR=0.5):
        # add noise to the mnist


        x_train_noisy = x_train + NOISE_FACTOR * \
            np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + NOISE_FACTOR * \
            np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

        x_train_noisy = np.clip(x_train_noisy, 0., 1.)     # limit into [0, 1]
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)   # limit into [0, 1]
        self.showImages(x_test_noisy, x_test, txt='addNoise')

        return x_train_noisy, x_test_noisy
