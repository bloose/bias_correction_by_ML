#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Oct. 7, 2019 

@author: Brice Loose

In this version of Z2_appcal_O2_LSTM.py we apply the fitted LSTM model to compute
the bias correction.

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
 
# train the model
def build_model(train, n_input,verbose,epochs,batch_size):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	#verbose, epochs, batch_size = 1, 15, 50
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(50, activation='tanh', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs)); model.add(Dropout(0.4))
	model.add(LSTM(50, activation='tanh', return_sequences=True))
	model.add(TimeDistributed(Dense(20, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
	return model


verbose, epochs, batch_size = 1, 20, 80
n_input = 100

pth = '../ZIPP2_EN602/EN602_Loose/science/UMS/MS Data/'
target = 64

idx = [62,5,36,28,57,3,64]
#idx = [36,28,57,64]


#%%

###------------- CALIBRATE July 12 ----------------------- #####
massspec_12_1 = pd.read_csv(pth+'MSData_7_12_2017 21_08.dfData',sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1')
hdrs = massspec_12_1.columns.values

#Shift to compensate for lage in UMS flow-thru system#
massspec_12_1 = lag_shift(massspec_12_1,hdrs[17:],41)
massspec_12_1 = massspec_12_1.dropna(axis=0, how='all')
ms = massspec_12_1.iloc[5858:9895, :]
ms = ms.reset_index(drop=True)
y12 = ms[hdrs[target]]

X = ms[hdrs[idx]].interpolate(method='linear', order=1, limit_direction='both')

Xf = X.values
Xf_mean = Xf.mean(axis=0)
Xf_std = Xf.std(axis=0)
x = (Xf-Xf_mean)/Xf_std

#shorten dataset to split into pieces
x = x[0:len(x)-np.mod(len(x),n_input)]

train, test = split_dataset(x,n_input)
# evaluate model and get scores

model12 = build_model(train, n_input,verbose, epochs, batch_size)

#%%

###------------- CALIBRATE July 14 ----------------------- #####
massspec_14_1 = pd.read_csv(pth+'MSData_7_14_2017 21_13.dfData',sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1')
hdr14 = massspec_14_1.columns.values

#Shift to compensate for lage in UMS flow-thru system#
massspec_14_1 = lag_shift(massspec_14_1,hdr14[17:],41)

massspec_14_1 = massspec_14_1.dropna(axis=0, how='all')
ms = massspec_14_1.iloc[6950:11310, :]
ms = ms.reset_index(drop=True)
y14 = ms[hdr14[target]]

X = ms[hdr14[idx]].interpolate(method='linear', order=1, limit_direction='both')

Xf = X.values
Xf_mean = Xf.mean(axis=0)
Xf_std = Xf.std(axis=0)
x = (Xf-Xf_mean)/Xf_std

#shorten dataset to split into pieces
x = x[0:len(x)-np.mod(len(x),n_input)]

train, test = split_dataset(x,n_input)
model14 = build_model(train, n_input,verbose, epochs, batch_size)

#%%
###------------- CALIBRATE July 15 ----------------------- #####
massspec_15_1 = pd.read_csv(pth+'MSData_7_15_2017 22_44.dfData',sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1')
hdr15 = massspec_15_1.columns.values

#Shift to compensate for lage in UMS flow-thru system#
massspec_15_1 = lag_shift(massspec_15_1,hdr15[17:],41)


# Missing CTD data at lines 4287 4289
massspec_15_1.iloc[4202:4207, :] = np.nan;
massspec_15_1 = massspec_15_1.dropna(axis=0, how='all')
ms = massspec_15_1.iloc[3997:6885, :]
#ms = massspec_15_1
ms = ms.reset_index(drop=True)


ms[hdrs[3]] = wild_edit_2.despike(ms[hdrs[3]],2,10,20)
ms[hdrs[5]] = wild_edit_2.despike(ms[hdrs[5]],2,10,20)


X = ms[hdrs[idx]].interpolate(method='linear', order=1, limit_direction='both')
y15 = ms[hdr15[target]]

X_train = X
y_train = y15


Xf = X.values
Xf_mean = Xf.mean(axis=0)
Xf_std = Xf.std(axis=0)
x = (Xf-Xf_mean)/Xf_std

#shorten dataset to split into pieces
x = x[0:len(Xf)-np.mod(len(Xf),n_input)]

train, test = split_dataset(x,n_input)
#model15 = build_model(train, n_input)
test_x, test_y = to_supervised(test, n_input)

#model15, result = build_model(train, n_input,test_x,test_y)
train_x, train_y = to_supervised(train, n_input)
	# define parameters

n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

	# define model
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(Dropout(0.4))

model.add(LSTM(50, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(20, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')
	# fit network
#result = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
#   validation_data=(test_x, test_y),shuffle=False)
result = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
   validation_data=(test_x, test_y))
model15=model

train_x, train_y = to_supervised(train, n_input)


yhat = model15.predict(train_x,verbose=1)

mz = yhat*Xf_std[-1]+Xf_mean[-1]
#y = train_y[:,6]*Xf_std[-1]+Xf_mean[-1]

#yhat2 = model.predict(test_x,verbose=1)
plt.figure()
plt.plot(mz[:,1,:])
plt.plot(y15)

plt.figure()
plt.plot(result.history['loss'],label='Train RMSE')
plt.plot(result.history['val_loss'],label='Test RMSE')
plt.xlabel('Epoch');
plt.ylabel('% RMSE');
plt.legend()
plt.show()


#%%
#n_input=50
###------------- CALIBRATE July 17 ----------------------- #####
massspec_17_1 = pd.read_csv(pth+'MSData_7_17_2017 13_54.dfData',sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1')
hdr17 = massspec_17_1.columns.values

#Shift to compensate for lage in UMS flow-thru system#
massspec_17_1 = lag_shift(massspec_17_1,hdr17[17:],41)

massspec_17_1.iloc[13543:13546, :] = np.nan;
massspec_17_1 = massspec_17_1.dropna(axis=0, how='all')
ms = massspec_17_1.iloc[13525:15500, :]
#ms = massspec_17_1
ms = ms.reset_index(drop=True)


ms[hdr17[3]] = wild_edit_2.despike(ms[hdr17[3]],2,10,20)
ms[hdr17[5]] = wild_edit_2.despike(ms[hdr17[5]],2,10,20)


X = ms[hdr17[idx]].interpolate(method='linear', order=1, limit_direction='both')
y17 = ms[hdr17[target]]

Xf = X.values
Xf_mean = Xf.mean(axis=0)
Xf_std = Xf.std(axis=0)
x = (Xf-Xf_mean)/Xf_std

#shorten dataset to split into pieces
x = x[0:len(x)-np.mod(len(x),n_input)]

train, test = split_dataset(x,n_input)
train_x, train_y = to_supervised(train, n_input)

model17 = build_model(train, n_input,verbose, epochs, batch_size)

yhat = model17.predict(train_x,verbose=1)

mz = yhat*Xf_std[-1]+Xf_mean[-1]
#y = train_y[:,6]*Xf_std[-1]+Xf_mean[-1]

#yhat2 = model.predict(test_x,verbose=1)
plt.figure()
plt.plot(mz[:,1,:])
plt.plot(y17)


#%%
###------------- NNLS optimization fun ----------------------- #####
from scipy.optimize import minimize
def optocf(x0,ytrain,yp,o2):
    import Solubility as s
    import numpy as np
    den = (np.mean(ytrain-x0))
    CF = S.O2sol(35.5,23)/den
    yrect = (yp + x0)*CF
    
    return np.mean(np.abs(o2 - yrect))


fil = ['MSData_7_12_2017 11_36.dfData','MSData_7_12_2017 21_08.dfData',
       'MSData_7_13_2017 04_08.dfData','MSData_7_13_2017 10_08.dfData',
       'MSData_7_13_2017 16_24.dfData','MSData_7_13_2017 22_44.dfData',
       'MSData_7_14_2017 05_02.dfData','MSData_7_14_2017 11_31.dfData',
       'MSData_7_14_2017 21_13.dfData','MSData_7_15_2017 04_08.dfData',
       'MSData_7_15_2017 10_53.dfData','MSData_7_15_2017 22_44.dfData',
       'MSData_7_16_2017 05_04.dfData','MSData_7_16_2017 12_15.dfData',
       'MSData_7_16_2017 18_36.dfData','MSData_7_17_2017 00_07.dfData',
       'MSData_7_17_2017 06_53.dfData','MSData_7_17_2017 13_54.dfData']

#fil = fil[0:3];

# Make data set with progressive change in CF to cover changes in the calibration     
Yp = pd.DataFrame(); IC = np.array([]); CF = np.array([]);
cfi = 0; y = [y12,y14,y15,y17];
CFt = [massspec_14_1.iloc[0,0],massspec_15_1.iloc[0,0],massspec_17_1.iloc[0,0],massspec_17_1.iloc[0,0]]

lstm = [model12,model14,model15,model17]



for f in fil:
    ma = pd.read_csv(pth+f,sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1',
                       warn_bad_lines=True)

    ma = lag_shift(ma,hdrs[17:],41)    
    ma = ma.dropna(axis=0, how='all')
    
    # Trim the ends to remove possible bad data in files
    ma = ma.iloc[100:-100:1,:]
    
    ma[hdrs[7]] = wild_edit_2.despike(ma[hdrs[7]],2,10,20)
    ma[hdrs[5]] = wild_edit_2.despike(ma[hdrs[5]],2,10,20)
    ma[hdrs[8]] = wild_edit_2.despike(ma[hdrs[8]],2,10,20)
    
    ma[hdrs[8]] = ma[hdrs[8]].interpolate(method='linear', order=1, limit_direction='both')    
    
    #X = ma[hdrs[idx]].dropna(axis=0, how='any')
    X = ma[hdrs[idx]].interpolate(method='linear', order=1, limit_direction='both')
    
    Xf = X.values
    Xf_mean = Xf.mean(axis=0)
    Xf_std = Xf.std(axis=0)
    x = (Xf-Xf_mean)/Xf_std
    
    pad = n_input-np.mod(len(x),n_input)
    x = np.append(x,np.ones([pad+n_input,len(idx)]),axis=0)

    trane = array(split(x, len(x)/n_input))

    trane_x, trane_y = to_supervised(trane, n_input)

    yhat = lstm[cfi].predict(trane_x,verbose=1)
    yr = pd.DataFrame(yhat[0:-pad-1,1,:])*Xf_std[-1]+Xf_mean[-1]
        
    #yr = pd.Series(gam[cfi].predict(X))

    
    Os = pd.DataFrame(S.O2sol(ma[hdrs[7]],ma[hdrs[5]]),columns=['Os'])
    Os['Os'] = wild_edit_2.despike(Os['Os'])
    
    yo = pd.concat([ma[hdrs[target]],yr,ma[hdrs[8]]],axis=1).dropna(how='any')
    t = minimize(optocf,1e-15,args=(y[cfi],yo[hdrs[target]]-yo[0],yo[hdrs[8]]),method='Nelder-Mead',tol=1e-13)
    ic = t.x
    IC = np.append(IC,ic)
    cf = S.O2sol(35.5,23)/np.mean(y[cfi]-ic)
    CF = np.append(CF,cf);
    
        # Choose which inputs to use, based on timestamp
    if (ma.iloc[-1,0] > CFt[cfi]) & (cfi<=2):
        cfi+=1
    

    #Ycorr = pd.concat([ma[hdrs[0]],(ma[hdrs[target]]-yr+IC0)*(CF["CF"])],axis=1)
    yraw = yr; yraw.rename(columns={0:'yraw'},inplace=True)

    yrect = pd.DataFrame((ma[hdrs[target]]-yr.iloc[:,0]+ic)*cf,columns={'yrect'})
    Ycorr = pd.concat([ma,yraw,yrect,Os],axis=1)
    Yp = pd.concat([Yp, Ycorr])
    
    

Yp.reset_index(inplace=True)
#Yp.set_index(hdrs[0],inplace=True)
#Yp.drop(columns=('level_0','index','Index'),inplace=True)
Yp.drop(columns=('index'),inplace=True)


#Yp[hdrs[8]]= wild_edit_2.despike(Yp[hdrs[8]])
Yp['yrect'] = wild_edit_2.despike(Yp['yrect'])
Yp['Os'] = wild_edit_2.despike(Yp['Os'])

#%%

plt.figure()   

Yp['yrect'] = Yp['yrect'].interpolate(method='linear', order=1, limit_direction='both')

res = ((Yp['yrect'] - Yp[hdrs[8]]))/np.mean(Yp[hdrs[8]])
res.dropna(how='all',inplace=True)


plt.plot(np.cumsum(res**2)); plt.title('Cumulative residuals')
plt.show()

mse = np.round(np.sqrt(np.nansum((Yp['yrect'] - Yp[hdrs[8]])**2)/len(Yp[hdrs[8]])),3)


f,ax = plt.subplots(1,1)
ax.plot(Yp['yrect'],'r',label='SWIMS')
ax.plot(Yp[hdrs[8]],'g',label='SBE35DO')
ax.legend()
ax.set_ylim(100,300)
ax.set_title('RMSE='+mse.astype(str))
#axb = ax.twinx()
#axb.plot(Yp[hdrs[idx[-1]]])
#axb.set_ylim(100,350)
plt.show()


plt.figure(3)
#plt.hist(res*100,1000)
  # Density Plot and Histogram of all arrival delays
sns.distplot(res*100, hist=False, kde=True,
             kde_kws = {'shade': True, 'linewidth': 2},label='N_input='+str(n_input))
plt.xlabel('%Error')
plt.legend(); plt.yscale('log')

#plt.xlim(-50,50)
plt.show()


DelO2 = (Yp['yrect'].divide(Yp['Os'])-1)*100

#f,ax2 = plt.subplots(1,1)
#ax2.plot(DelO2)
#ax2.set_ylim(-20,20)



#axb = ax2.twinx()
#axb.plot(Yp[hdrs[idx[-2]]],'g')
#axb.set_ylim(0,300)
#
#Del_ms = (Yp['yrect']/Yp['Os']-1)*100
#Del_o2 = (Yp[hdrs[8]]/Yp['Os']-1)*100
#
#f,ax = plt.subplots(2,1)
#ax[0].plot(mse)
#ax[0].set_ylim(-50,50)
#ax[1].plot(Del_ms)
#ax[1].plot(Del_o2)
#ax[1].set_ylim(-30,30)
#axb = ax[1].twinx()
#axb.plot(Yp[hdrs[idx[-2]]],'r')
#axb.set_ylim(0,300)
#plt.show()


plt.show()

plt.figure()
plt.plot(result.history['loss'],label='loss')
plt.plot(result.history['val_loss'],label='valid_loss')
plt.xlabel('epoch');
plt.legend()
plt.show()
    
    
#%%
Yp.to_pickle('Z2_O2_opt.pkl')
    
