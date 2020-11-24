#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Oct. 9, 2019 

@author: Brice Loose

This code implements a modified version of Jason Brownlee's tutorial on implementing
dropout in the LSTM model to avoid overfitting.

"""


# multivariate multi-step encoder-decoder lstm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout

import wild_edit_2
from SWIMS_time_lag import lag_shift
import imp
 
# split a univariate dataset into train/test sets
def split_dataset(data,n_input):
	# split into standard weeks
	train, test = data[0:-300], data[-300:]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/n_input))
	test = array(split(test, len(test)/n_input))
	return train, test
 

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=100):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		#out_end = in_end + n_out
		# ensure we have enough data for this instance
		if in_end <= len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_start:in_end, -1])
		# move along one time step
		in_start += 1
	return array(X), array(y)
 
# train the model
#def build_model(train, n_input,valX,valY):
#	# prepare data
#
#	train_x, train_y = to_supervised(train, n_input)
#	# define parameters
#	verbose, epochs, batch_size = 1, 10, 32
#	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
#	# reshape output into [samples, timesteps, features]
#	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
#    
#    valY = valY.reshape((valY.shape[0], valY.shape[1], 1))
#
#	# define model
#	model = Sequential()
#	model.add(LSTM(100, activation='tanh', input_shape=(n_timesteps, n_features)))
#	model.add(RepeatVector(n_outputs))
#	model.add(LSTM(100, activation='tanh', return_sequences=True))
#	model.add(TimeDistributed(Dense(50, activation='relu')))
#	model.add(TimeDistributed(Dense(1)))
#	model.compile(loss='mse', optimizer='adam')
#	# fit network
#	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
#           validation_data=(valX,valY),shuffle=False)
#	return model
 
 
# load the new file
pth = '/Users/suelto/GDrive/research/ZIPP/ZIPP2_EN602/EN602_Loose/science/UMS/MS Data/'


massspec_15_1 = pd.read_csv(pth+'MSData_7_15_2017 22_44.dfData',sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1')
hdrs = massspec_15_1.columns.values

#Shift to compensate for lage in UMS flow-thru system#
massspec_15_1 = lag_shift(massspec_15_1,hdrs[17:-1],41)


# Missing CTD data at lines 4287 4289
massspec_15_1.iloc[4202:4207, :] = np.nan;
massspec_15_1 = massspec_15_1.dropna(axis=0, how='all')
ms = massspec_15_1.iloc[4100:6500, :]
#ms = massspec_15_1
ms = ms.reset_index(drop=True)


# Oxygen

ms[hdrs[3]] = wild_edit_2.despike(ms[hdrs[3]],2,10,20)
ms[hdrs[5]] = wild_edit_2.despike(ms[hdrs[5]],2,10,20)

idx = [62,5,36,28,57,3,64]

n_input = 100


Xnp = ms[hdrs[idx]].interpolate(method='linear', order=1, limit_direction='both')

Xf = Xnp.values
Xf_mean = Xf.mean(axis=0)
Xf_std = Xf.std(axis=0)
dataset = (Xf-Xf_mean)/Xf_std

#dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset,n_input)
# evaluate model and get scores
n_input = 100

test_x, test_y = to_supervised(test, n_input)

#%%

#model = build_model(train, n_input,test_x,test_y)

	# prepare data

train_x, train_y = to_supervised(train, n_input)
#train_x = train; train_y = train[:,:,3]
	# define parameters
#verbose, epochs, batch_size = 1, 10, 32 
verbose, epochs, batch_size = 1, 30, 20 
 
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

	# define model
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(Dropout(0.1))
model.add(LSTM(50, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(20, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')
	# fit network
#result = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
#   validation_data=(test_x, test_y),shuffle=False)
result = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
   validation_data=(test_x, test_y))




#%%

#train_x, train_y = to_supervised(train, 100)

yhat = model.predict(train_x,verbose=1)

mz40 = yhat*Xf_std[-1]+Xf_mean[-1]
train_40 = train_y[:,6]*Xf_std[-1]+Xf_mean[-1]

mse = np.sqrt(np.sum((mz40[:,1,:]-train_40)**2))/np.mean(train_40)


#yhat2 = model.predict(test_x,verbose=1)
plt.figure()
plt.plot(mz40[:,1,:])
plt.plot(train_40)

#id = np.arange(test_y.shape[0]-100.,test_y.shape[0],1)
#plt.plot(id[:,:,0],yhat2[:,1,:])
#plt.plot(id[:,:,0],test_y[:,6])
plt.title((['rmse',mse]))

plt.show()


plt.figure()
plt.plot(result.history['loss'],label='loss')
plt.plot(result.history['val_loss'],label='valid_loss')
plt.xlabel('epoch');
plt.legend()
plt.show()




#model.save('LSTM_mz40.h5')
#
#from keras.models import load_model
#
#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
#
## returns a compiled model
## identical to the previous one
#model = load_model('my_model.h5')

