from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l1, l2
import numpy as np
import matplotlib.pyplot as plt

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
model.add(Activation('tanh'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(30))
model.add(Activation('sigmoid'))
model.add(Dense(classes))
model.add(Activation('sigmoid'))

model.compile(optimizer = RMSprop(), loss='mse', metrics = ['accuracy'])
score = []
history = model.fit(xtr, Ytr, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)
loss, acc = model.evaluate(xts, Yts, verbose = 0)
score.append([loss, acc])
# summarize history for accuracy
print(history.history.keys())
plt.figure()
# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('training accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('training loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


