from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.visualize_util import plot
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l1, l2
import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
classes = 10
nb_epoch = 50


(xtr, ytr), (xts, yts) = mnist.load_data()
N = 10000
xtr = xtr[:N]
ytr = ytr[:N]
xts = xts[:N]
yts = yts[:N]

xtr = xtr.reshape(xtr.shape[0], 784)
xts = xts.reshape(xts.shape[0], 784)

xtr = xtr.astype('float32')
xts = xts.astype('float32')

xtr /= 255
xts /= 255

Ytr = np_utils.to_categorical(ytr, classes)
Yts = np_utils.to_categorical(yts, classes)


k = 8
W1 = [np.random.random((784, 30)), np.random.random((30, ))]
W2 = [np.random.random((30, 30)), np.random.random((30, ))]
W3 = [np.random.random((30, 30)), np.random.random((30, ))]
W4 = [np.random.random((30, 10)), np.random.random((10, ))]


score = []

model = Sequential()
model.add(Dense(30, input_shape = (784, ), weights = W1))
model.add(Activation('sigmoid'))
model.add(Dense(30, weights = W2))
model.add(Activation('sigmoid'))
model.add(Dense(30, weights = W3))
model.add(Activation('sigmoid'))
model.add(Dense(10, weights = W4))
model.add(Activation('sigmoid'))

model.compile(optimizer = RMSprop(), loss='mse', metrics = ['accuracy'])

history = model.fit(xtr, Ytr, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)
loss, acc = model.evaluate(xts, Yts, verbose = 0)
score.append([loss, acc])

mat = []
mat.append(model.layers[2].get_weights()[0])

print score

plt.matshow(mat[0], cmap = plt.cm.jet)
plt.title('Weights without regularization')
plt.colorbar()

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

# score = model.evaluate(Xts.T, yts, batch_size =200)
# print score
