import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

import theano

from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import RMSprop

from seya.layers.coding import SparseCoding

from agnez import grid2d


def main():
    S = loadmat('patches.mat')['data'].T.astype(theano.config.floatX)
    print S.shape

    image_patches = fetch_mldata("natural scenes data")
    X = image_patches.data

    mean = S.mean(axis=0)
    S -= mean[np.newaxis]

    model = Sequential()
    model.add(
        SparseCoding(
            input_dim=256,
            output_dim=1000,  # we are learning 49 filters,
            n_steps=25,  # remember the self.n_steps in the scan loop?
            truncate_gradient=1,  # no backpropagation through time today now,
            # just regular sparse coding
            W_regularizer=l2(.00005),
            return_reconstruction=True  # we will output Ax which approximates the input
        )
    )

    rmsp = RMSprop(lr=.1)
    model.compile(loss='mse', optimizer=rmsp)  # RMSprop for Maximization as well

    nb_epoch = 100
    batch_size = 100
    model.fit(S,  # input
              S,  # and output are the same thing, since we are doing generative modeling.
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              show_accuracy=False,
              verbose=2)

    A = model.params[0].get_value()
    I = grid2d(A)
    plt.imshow(I)


if __name__ == "__main__":
    main()
