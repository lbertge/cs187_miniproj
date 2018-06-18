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
nb_epoch = 20


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


k = 10
W1 = [np.random.random((784, 30)), np.random.random((30, ))]
W2 = [np.random.random((30, 30)), np.random.random((30, ))]
W3 = [np.random.random((30, 30)), np.random.random((30, ))]
W4 = [np.random.random((30, 10)), np.random.random((10, ))]

score = []
accuracy = []
plt.figure()

mat = []
for i in range(k):
	reg = 10 ** (-i)
	model = Sequential()
	model.add(Dense(30, input_shape = (784, ), W_regularizer = l1(reg), weights = W1))
	model.add(Activation('sigmoid'))
	model.add(Dense(30, W_regularizer = l1(reg), weights = W2))
	model.add(Activation('sigmoid'))
	model.add(Dense(30, W_regularizer = l1(reg), weights = W3))
	model.add(Activation('sigmoid'))
	model.add(Dense(10, W_regularizer = l1(reg), weights = W4))
	model.add(Activation('sigmoid'))

	model.compile(optimizer = RMSprop(), loss='mse', metrics = ['accuracy'])

	model.fit(xtr, Ytr, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)
	loss, acc = model.evaluate(xts, Yts, verbose = 0)
	score.append([i, loss])
	accuracy.append([i, acc])

	mat.append(model.layers[2].get_weights()[0])

print score
print accuracy
ax1 = plt.subplot(211)
ax1.set_title('Mean-squared error loss')
ax1.set_xlabel('Lambda = 10^-i')
ax1.set_ylabel('Loss')
plt.plot(zip(*score)[0], zip(*score)[1])

ax2 = plt.subplot(212)
ax2.set_title('Accuracy of the test set')
ax2.set_xlabel('Lambda = 10^-i')
ax2.set_ylabel('Accuracy')
plt.plot(zip(*accuracy)[0], zip(*accuracy)[1])

for i in range(k):
	plt.matshow(mat[i], cmap = plt.cm.jet)
	plt.title('Weights for lambda 10^' + str(-i))
	plt.colorbar()

plt.show()


# score = model.evaluate(Xts.T, yts, batch_size =200)
# print score
