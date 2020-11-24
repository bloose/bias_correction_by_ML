#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:48:08 2020

@author: suelto

Modified from Jason Brownlee blog: 
    https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

This code seeks to implement a random search for hyperparameter estimation.
   

"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import patsy
from patsy import dmatrix
import Solubility as S
import wild_edit_2
from SWIMS_time_lag import lag_shift
import seaborn as sns


# multivariate multi-step encoder-decoder lstm
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
		if in_end <= len(data)-1:
			X.append(data[in_start:in_end, 0:])
			y.append(data[in_start:in_end, -1])
            
		# move along one time step
		in_start += 1
	return array(X), array(y)
 



"""
New code below
"""

verbose = 1
n_input = 100

# define the grid search parameters
batch_size = [20, 60, 80]
epochs = [5, 10, 20]
dropout = [0.1,0.4,0.6,0.8]
param_grid = dict(batch_size=batch_size, epochs=epochs,n_drop=dropout)

pth = '../ZIPP2_EN602/EN602_Loose/science/UMS/MS Data/'
target = 64

#idx = [62,5,36,28,57,3,64]
idx = [36,28,57,64]



# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

# Function to create model, required for KerasClassifier
###### Create model function is above #####
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset

#%%
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
x = array(split(x, len(x)/n_input))



#%%

train_x, train_y = to_supervised(x, n_input)

#train_x = x; train_y = x[:,:,3]

# define parameters
#verbose, epochs, batch_size = 1, 15, 50
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

# train the model
def build_model(n_timesteps=n_timesteps, n_features=n_features, n_outputs=n_outputs, n_drop=0.5):

	# define model
	model = Sequential()
	model.add(LSTM(50, activation='tanh', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs)); model.add(Dropout(n_drop))
	model.add(LSTM(50, activation='tanh', return_sequences=True))
	model.add(TimeDistributed(Dense(20, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	return model



# create model
#model = KerasClassifier(build_fn=build_model, verbose=1)
model = KerasRegressor(build_fn=build_model, verbose=0)

#model= build_model(n_timesteps, n_features, n_outputs)



grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(train_x, train_y)



# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))