from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.visualize_util import plot
from keras.optimizers import SGD
from keras.regularizers import l1, l2
import numpy as np
import scipy.io
import pydot_ng as pydot
import time

x  = scipy.io.loadmat('dataset.mat')
Xts = x['Xts']
Xtr = x['Xtr']
ytr = x['ytr']
yts = x['yts']

reg = 0.01

weights1 = np.random.random((2, 5))
print weights1.shape

def model(reg = True):
	model = Sequential()
	if reg:
		model.add(Dense(5, input_dim=2))
		model.add(Activation('sigmoid'))
		model.add(Dense(5))
		model.add(Activation('sigmoid'))
		model.add(Dense(1, W_regularizer = l2(reg)))
		model.add(Activation('sigmoid'))
	else:
		model.add(Dense(5, input_dim=2))
		model.add(Activation('sigmoid'))
		model.add(Dense(5))
		model.add(Activation('sigmoid'))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
	return model


model = model(reg = True)
sgd = SGD(lr=0.1, momentum = 0, decay = 0, nesterov = False)
model.compile(optimizer = sgd, loss = 'mse', metrics = ['accuracy'])

for layer in model.layers:
	print layer.get_weights()

model.fit(Xtr.T, ytr, nb_epoch=50, batch_size=2, verbose = 2)

for layer in model.layers:
	print layer.get_weights()
score = model.evaluate(Xts.T, yts, batch_size =200)
print score
