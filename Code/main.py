# time: 2019-4-22
# author: Zelin Zang
# email: zangzelin@gmail.com
# github: https://github.com/zangzelin/

# this is the main function of ae/cae/dae/vae
# run 'python3 main.py ae' to test ae
# run 'python3 main.py cae' to test cae
# run 'python3 main.py dae' to test dae
# run 'python3 main.py vae' to test vae


import ae
import dae
import cae
import vae
from keras.datasets import mnist
import sys
import numpy as np


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


if __name__ == "__main__":

    modelname = sys.argv[1]

    # Step1： load data  x_train: (60000, 28, 28), y_train: (60000,) x_test: (10000, 28, 28), y_test: (10000,)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Step2: normalize
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    if modelname == 'ae':
        # Step3: reshape data, x_train: (60000, 784), x_test: (10000, 784), one row denotes one sample.
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))

        encoder = ae.AE()
        encoder.TrainNetwork(x_train, y_train, EPOCHS=100)
        encoder.PlotRepresentation(x_test, y_test, txt='with_relu')

        decode_images = encoder.Predict(x_test)
        encoder.showImages(decode_images, x_test, txt='with_relu')

        encoder.plotAccuray()
        encoder.PlotModel()

    elif modelname == 'dae':

        # Step3: reshape data, x_train: (60000, 784), x_test: (10000, 784), one row denotes one sample.
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))

        # pre train the network with self-supervision
        encoder = dae.DAE()
        x_train, x_test = encoder.addNoise(
            x_train, x_test)                                            
        encoder.TrainNetwork(x_train, y_train, EPOCHS=20)               
        encoder.PlotRepresentation(x_test, y_test, txt='without_sl')    

        # show the predict figure 
        decode_images = encoder.Predict(x_test)
        encoder.showImages(decode_images, x_test, txt='without_sl')
        encoder.plotAccuray(txt='without_sl')

        # train the network with label-supervision
        y_train_onehot = convert_to_one_hot(y_train, 10)
        encoder.SlTraining(x_train, y_train_onehot, EPOCHS=20)
        encoder.PlotRepresentation(x_test, y_test, txt='with_sl')
        decode_images = encoder.Predict(x_test)
        encoder.showImages(decode_images, x_test, txt='with_sl')

        encoder.plotAccuray(txt='with_sl')
        encoder.PlotModel()

    elif modelname == 'cae':
        # Step3: reshape data, x_train: (60000, 28, 28, 1), x_test: (10000, 28, 28, 1), one row denotes one sample.
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

        # pre train the network with self-supervision
        encoder = cae.CAE()
        # x_train, x_test = encoder.addNoise(x_train, x_test)   
        history_record = encoder.TrainNetwork(x_train, y_train, EPOCHS=20)
        encoder.PlotRepresentation(x_test, y_test, txt='without_sl')

        # show the predict figure 
        decode_images = encoder.Predict(x_test)
        encoder.showImages(decode_images, x_test, txt='without_sl')
        encoder.plotAccuray(txt='without_sl')

        # train the network with label-supervision
        y_train_onehot = convert_to_one_hot(y_train, 10)
        encoder.SlTraining(x_train, y_train_onehot, EPOCHS=20)
        encoder.PlotRepresentation(x_test, y_test, txt='with_sl')
        decode_images = encoder.Predict(x_test)
        encoder.showImages(decode_images, x_test, txt='with_sl')

        encoder.plotAccuray(txt='with_sl')
        encoder.PlotModel()

    elif modelname == 'vae':

        # Step3: reshape data, x_train: (60000, 28, 28, 1), x_test: (10000, 28, 28, 1), one row denotes one sample.
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))

        # train the model with kl + mse
        encoder = vae.VAE(mse_loss=False)
        # x_train, x_test = encoder.addNoise(x_train, x_test)
        history_record = encoder.TrainNetwork(
            x_train, y_train, x_test, EPOCHS=50)
        encoder.PlotRepresentation(x_test, y_test, txt='KL')

        decode_images = encoder.Predict(x_test)
        encoder.showImages(decode_images, x_test, txt='KL')
        encoder.plotAccuray(txt='KL')
        encoder.PlotLatent(txt='KL')

        # train the model with kl + xent
        encoder = vae.VAE(mse_loss=True)
        # x_train, x_test = encoder.addNoise(x_train, x_test)
        history_record = encoder.TrainNetwork(
            x_train, y_train, x_test, EPOCHS=50)
        encoder.PlotRepresentation(x_test, y_test, txt='xent')

        decode_images = encoder.Predict(x_test)
        encoder.showImages(decode_images, x_test, txt='xent')
        encoder.plotAccuray(txt='xent')
        encoder.PlotLatent(txt='xent')

        encoder.PlotModel()
