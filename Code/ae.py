
# Autoencoder with single hidden layer.
# time: 2019-4-22
# author: Zelin Zang
# email: zangzelin@gmail.com
# github: https://github.com/zangzelin/

import os

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import plot_model
from sklearn.decomposition import PCA


def PCAdecomposition(data):
    # lower the demention with PCA method: n->2

    estimator = PCA(n_components=2)
    pca_X_train = estimator.fit_transform(data)
    return pca_X_train


class AE:

    def __init__(self,
                 ENCODING_DIM_INPUT=784,
                 ENCODING_DIM_OUTPUT=2,
                 Name="ae"):

        self.ENCODING_DIM_INPUT = ENCODING_DIM_INPUT
        self.ENCODING_DIM_OUTPUT = ENCODING_DIM_OUTPUT
        self.name = Name

        input_image = Input(shape=(ENCODING_DIM_INPUT, ))

        # encoding layer
        hidden_layer = Dense(ENCODING_DIM_OUTPUT,
                             activation='relu')(input_image)
        # decoding layer
        decode_output = Dense(ENCODING_DIM_INPUT,
                              activation='relu')(hidden_layer)

        # build autoencoder, encoder, decoder
        autoencoder = Model(inputs=input_image, outputs=decode_output)
        encoder = Model(inputs=input_image, outputs=hidden_layer)
        # decoder = Model(inputs=hidden_layer, outputs=decode_output)

        # compile autoencoder
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        self.autoencoder = autoencoder
        self.encoder = encoder
        # self.decoder = decoder

    def TrainNetwork(self, x_train, y_train, EPOCHS=10, BATCH_SIZE=64):
        # training
        self.history_record = self.autoencoder.fit(
            x_train, x_train,  epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        return self.history_record

    def Predict(self, x_test):
        return self.autoencoder.predict(x_test)

    def PlotRepresentation(self, x_test, y_test, need_show=False, txt=''):
        # plot the hidden result.
        # :param x_test: the images after encoding
        # :param y_test: the label.

        # test and plot
        encode_images = self.encoder.predict(x_test)

        if len(encode_images.shape) > 3:
            encode_images_pca = []
            encode_images_liner = encode_images.reshape(
                (-1, np.multiply.reduce(encode_images.shape[1:]))
            )
            encode_images_pca.append(PCAdecomposition(encode_images_liner))
            encode_images = np.array(encode_images_pca)[0]

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

    def showImages(self,
                   decode_images,
                   x_test,
                   need_show=False,
                   txt=''):
        # plot the images.
        # :param decode_images: the images after decoding
        # :param x_test: testing data

        # decode_images = self.autoencoder.predict(x_test)
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i+1)
            ax.imshow(x_test[i].reshape(28, 28))
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            ax.imshow(decode_images[i].reshape(28, 28))
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if need_show:
            plt.show()
        else:
            path = './output/'+self.name
            if not os.path.exists(path):
                os.makedirs('./output/'+self.name)
            plt.savefig('./output/'+self.name+'/showImage'+txt+'.png')
        plt.close()

    def plotAccuray(self,
                    need_show=False,
                    txt=''
                    ):
        # plot the accuracy and loss line.

        history_record = self.history_record
        accuracy = history_record.history["acc"]
        loss = history_record.history["loss"]
        epochs = range(len(accuracy))

        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.title('Training accuracy')
        plt.legend()

        if need_show:
            plt.show()
        else:
            path = './output/'+self.name

            if not os.path.exists(path):
                os.makedirs('./output/'+self.name)
            plt.savefig('./output/'+self.name+'/plotAcc'+txt+'.png')
        plt.close()

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

    def PlotModel(self, need_show=False, txt=''):
        # plot the structure of the model

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
