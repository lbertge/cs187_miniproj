from keras.utils.visualize_util import plot
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.utils.visualize_util import plot
from keras.optimizers import SGD, RMSprop
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l1, l2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.visualize_util import model_to_dot



batch_size = 100
classes = 10
nb_epoch = 10
nb_filters = 32
input_shape = (28, 28, 1)

pool_size = (2, 2)
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(xtr, ytr), (xts, yts) = mnist.load_data()

N = 10000
xtr = xtr[:N]
ytr = ytr[:N]
xts = xts[:N]
yts = yts[:N]
xtr = xtr.reshape(xtr.shape[0], 28, 28, 1)
xts = xts.reshape(xts.shape[0], 28, 28, 1)

xtr = xtr.astype('float32')
xts = xts.astype('float32')

xtr /= 255
xts /= 255

Ytr = np_utils.to_categorical(ytr, classes)
Yts = np_utils.to_categorical(yts, classes)


model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('sigmoid'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(classes))
model.add(Activation('sigmoid'))

model.compile(optimizer = RMSprop(), loss='mse', metrics = ['accuracy'])

plot(model, to_file='modelCNN.png', show_shapes = True)