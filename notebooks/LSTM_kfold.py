#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:49:17 2020

@author: Brice

LSTM_kfold.py


"""

# multivariate multi-step encoder-decoder lstm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


from tensorflow.keras.losses import sparse_categorical_crossentropy
from sklearn.model_selection import KFold

import wild_edit_2
from SWIMS_time_lag import lag_shift
import imp
 
# split a univariate dataset into train/test sets
def split_dataset(data,n_input):
	# split into standard weeks
	train, test = data[0:-n_input*2], data[-n_input*2:]
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

 
#%%
pth = '../ZIPP2_EN602/EN602_Loose/science/UMS/MS Data/'
target = 64

#idx = [62,5,36,28,57,3,64]
idx = [36,28,57,64]
n_input = 100
    

###------------- CALIBRATE July 12 ----------------------- #####
massspec_12_1 = pd.read_csv(pth+'MSData_7_12_2017 21_08.dfData',sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1')
hdrs = massspec_12_1.columns.values

#Shift to compensate for lage in UMS flow-thru system#
massspec_12_1 = lag_shift(massspec_12_1,hdrs[17:-1],41)
massspec_12_1 = massspec_12_1.dropna(axis=0, how='all')
ms = massspec_12_1.iloc[5858:9895, :]
ms = ms.reset_index(drop=True)
y12 = ms[hdrs[target]]

X = ms[hdrs[idx]].interpolate(method='linear', order=1, limit_direction='both')

Xf = X.values
Xf_mean = Xf.mean(axis=0)
Xf_std = Xf.std(axis=0)
x = (Xf-Xf_mean)/Xf_std

#model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
# prepare data
#shorten dataset to split into pieces
x = x[0:len(x)-np.mod(len(x),n_input)]
#x = array(split(x, len(x)/n_input))

# split into train and test
train, test = split_dataset(x,n_input)
test_x, test_y = to_supervised(test, n_input)
train_x, train_y = to_supervised(train, n_input)

#%%
# evaluate model and get scores


# Model configuration
loss_function = sparse_categorical_crossentropy
validation_split = 0.2


num_folds = 5

verbose, epochs, batch_size = 1, 5, 80 
 
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))


# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((train_x, test_x), axis=0)
targets = np.concatenate((train_y, test_y), axis=0)


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture

	# define model
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(Dropout(0.4))
    model.add(LSTM(50, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(20, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
    	# fit network
    #result = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
    #   validation_data=(test_x, test_y),shuffle=False)
    #result = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
    #   validation_data=(test_x, test_y))
    result = model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_data=(inputs[test], targets[test]))


    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

#----
#%%

yhat = model.predict(train_x,verbose=1)

mz32 = yhat*Xf_std[-1]+Xf_mean[-1]
train_32 = train_y[:,6]*Xf_std[-1]+Xf_mean[-1]

mse = np.sqrt(np.sum((mz32[:,1,:]-train_32)**2))/np.mean(train_32)


#yhat2 = model.predict(test_x,verbose=1)
plt.figure()
plt.plot(mz32[:,1,:])
plt.plot(train_32)
plt.title((['rmse',mse]))
plt.show()


plt.figure()
plt.plot(result.history['loss'],label='Train RMSD')
plt.plot(result.history['val_loss'],label='Test RMSD')
plt.xlabel('Epoch');
plt.ylabel('% RMSD');
plt.legend()
plt.show()
