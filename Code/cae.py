# conversational Autoencoder with 3 hidden layer.
# time: 2019-4-22
# author: Zelin Zang
# email: zangzelin@gmail.com
# github: https://github.com/zangzelin/

import os

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Input, MaxPool2D, UpSampling2D, Flatten
from keras.models import Model

import ae


class CAE(ae.AE):

    def __init__(
        self,
        CHANNEL_1=16,
        CHANNEL_2=8,
        CHANNEL_OUTPUT=1,
        Name="cae"
    ):

        self.CHANNEL_1 = CHANNEL_1
        self.CHANNEL_2 = CHANNEL_2
        self.name = Name

        # input placeholder
        input_image = Input(shape=(28, 28, 1))

        # encoding layer
        x = Conv2D(CHANNEL_1, (3, 3), activation='relu',
                   padding="same")(input_image)
        x = MaxPool2D((2, 2), padding='same')(x)
        x = Conv2D(CHANNEL_2, (3, 3), activation='relu', padding='same')(x)
        encode_output = MaxPool2D((2, 2), padding='same')(x)

        # decoding layer
        x = Conv2D(CHANNEL_2, (3, 3), activation='relu',
                   padding='same')(encode_output)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(CHANNEL_1, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decode_output = Conv2D(CHANNEL_OUTPUT, (3, 3),
                               activation='sigmoid', padding='same')(x)

        # build surprised learning model
        encode_output_flatten = Flatten()(decode_output)
        SL_output = Dense(10, activation='softmax')(encode_output_flatten)

        # build autoencoder, encoder
        autoencoder = Model(inputs=input_image, outputs=decode_output)
        encoder = Model(inputs=input_image, outputs=encode_output)
        SL_model = Model(inputs=input_image, outputs=SL_output)

        # compile autoencoder
        autoencoder.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        SL_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        autoencoder.summary()

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.sl_model = SL_model

    def SlTraining(self, x, y, EPOCHS=20):

        self.history_record = self.sl_model.fit(x, y, epochs=EPOCHS)
