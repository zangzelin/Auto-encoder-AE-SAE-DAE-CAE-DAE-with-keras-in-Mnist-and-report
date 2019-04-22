# Variational Auto-Encoders, VAE(Kingma, 2014)
# time: 2019-4-22
# author: Zelin Zang
# email: zangzelin@gmail.com
# github: https://github.com/zangzelin/

from __future__ import absolute_import, division, print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Conv2D, Dense, Flatten, Input, Lambda, MaxPool2D,
                          UpSampling2D)
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras.utils import plot_model

import ae


def sampling(args):
    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    # Reparameterization trick by sampling from an isotropic unit Gaussian.
    # # Arguments
    #     args (tensor): mean and log of variance of Q(z|X)
    # # Returns
    #     z (tensor): sampled latent vector

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE(ae.AE):

    def __init__(
        self,
        ENCODING_DIM_INPUT=784,
        intermediate_dim=512,
        batch_size=128,
        latent_dim=2,
        mse_loss=True,
        Name="vae"
    ):

        self.name = Name

        # input placeholder
        input_image = Input(shape=(ENCODING_DIM_INPUT,), name='encoder_input')

        # VAE model = encoder + decoder
        # encoding layer
        x = Dense(intermediate_dim, activation='relu')(input_image)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = Lambda(sampling, output_shape=(latent_dim,),
                   name='z')([z_mean, z_log_var])

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(ENCODING_DIM_INPUT, activation='sigmoid')(x)

        # # build surprised learning model
        # SL_output = Dense(10, activation='softmax')()

        # build autoencoder, encoder
        encoder = Model(input_image, [z_mean, z_log_var, z], name='encoder')
        decoder = Model(latent_inputs, outputs, name='decoder')
        # SL_model = Model(inputs=input_image, outputs=SL_output)

        outputs = decoder(encoder(input_image)[2])
        autoencoder = Model(input_image, outputs, name='vae_mlp')

        # compile autoencoder
        # VAE loss = mse_loss or xent_loss + kl_loss
        if mse_loss:
            reconstruction_loss = mse(input_image, outputs)
        else:
            reconstruction_loss = binary_crossentropy(input_image,
                                                      outputs)

        reconstruction_loss *= ENCODING_DIM_INPUT
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        autoencoder.add_loss(vae_loss)
        autoencoder.compile(optimizer='adam')
        autoencoder.summary()

        # SL_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        # self.sl_model = SL_model

    def TrainNetwork(self, x_train, y_train, x_test, EPOCHS=10, BATCH_SIZE=64):
        # training
        self.history_record = self.autoencoder.fit(
            x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, None))
        return self.history_record

    def PlotRepresentation(self, x_test, y_test, need_show=False, txt=''):
        # plot the hidden result.
        # :param x_test: the images after encoding
        # :param y_test: the label.
        # :return:


        # test and plot
        encode_images = self.encoder.predict(x_test)

        encode_images = encode_images[0]
        plt.scatter(encode_images[:, 0], encode_images[:, 1], c=y_test, s=3)
        plt.colorbar()

        if need_show:
            plt.show()
        else:
            path = './output/'+self.name
            if not os.path.exists(path):
                os.makedirs('./output/'+self.name)
            plt.savefig('./output/'+self.name+'/PlotRepresentation'+txt+'.png')
        plt.close()

    def plotAccuray(self,
                    need_show=False,
                    txt=''
                    ):
        # plot the accuracy and loss line.
        # :param history_record:
        # :return:


        history_record = self.history_record
        # accuracy = history_record.history["acc"]
        loss = history_record.history["loss"]
        epochs = range(len(loss))

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training loss')
        plt.legend()

        if need_show:
            plt.show()
        else:
            path = './output/'+self.name

            if not os.path.exists(path):
                os.makedirs('./output/'+self.name)
            plt.savefig('./output/'+self.name+'/plotloss'+txt+'.png')
        plt.close()

    def PlotLatent(self, txt=''):

        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        path = './output/'+self.name

        if not os.path.exists(path):
            os.makedirs('./output/'+self.name)
        plt.savefig('./output/'+self.name+'/PlotLatent'+txt+'.png')

    def PlotModel(self, need_show=False, txt=''):
        if need_show:
            pass
        else:
            path = './output/'+self.name

            if not os.path.exists(path):
                os.makedirs('./output/'+self.name)
            plot_model(self.encoder, to_file='./output/'+self.name +
                       '/Model_encoder'+txt+'.png', show_shapes=True)

            plot_model(self.autoencoder, to_file='./output/'+self.name +
                       '/Model_autoencoder'+txt+'.png', show_shapes=True)
            plot_model(self.decoder, to_file='./output/'+self.name +
                       '/Model_decoder'+txt+'.png', show_shapes=True)
