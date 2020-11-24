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



# multivariate multi-step encoder-decoder lstm
from numpy import split
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
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
 



verbose, epochs, batch_size = 1, 20, 80
n_input = 100

pth = '../../ZIPP2_EN602/EN602_Loose/science/UMS/MS Data/'
target = 64

idx = [62,5,36,28,57,3,64]
#idx = [36,28,57,64]


#%%

#%%

###------------- CALIBRATE July 15 ----------------------- #####

'''
massspec_15_1 = pd.read_csv(pth+'MSData_7_15_2017 22_44.dfData',sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1')
hdrs = massspec_15_1.columns.values

#Shift to compensate for lage in UMS flow-thru system#
massspec_15_1 = lag_shift(massspec_15_1,hdrs[17:-1],41)


# Missing CTD data at lines 4287 4289
massspec_15_1.iloc[4202:4207, :] = np.nan;
massspec_15_1 = massspec_15_1.dropna(axis=0, how='all')
ms = massspec_15_1.iloc[3997:6885, :]
#ms = massspec_15_1
ms = ms.reset_index(drop=True)


ms[hdrs[3]] = wild_edit_2.despike(ms[hdrs[3]],2,10,20)
ms[hdrs[5]] = wild_edit_2.despike(ms[hdrs[5]],2,10,20)


ms.to_csv('Insitu_cal.csv')
'''
ms = pd.read_csv(r'Insitu_cal.csv')


X = ms[hdrs[idx]].interpolate(method='linear', order=1, limit_direction='both')
y15 = ms[hdrs[target]]

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
result = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
   validation_data=(test_x, test_y))
model15=model

train_x, train_y = to_supervised(train, n_input)



yhat = model15.predict(train_x,verbose=1)

mz = yhat*Xf_std[-1]+Xf_mean[-1]


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
###------------- NNLS optimization fun ----------------------- #####
from scipy.optimize import minimize
def optocf(x0,ytrain,yp,o2):
    import Solubility as s
    import numpy as np
    den = (np.mean(ytrain-x0))
    CF = 210.68/den
    yrect = (yp + x0)*CF
    
    return np.mean(np.abs(o2 - yrect))


'''
fil = ['MSData_7_17_2017 00_07.dfData','MSData_7_17_2017 06_53.dfData']

#fil = fil[0:3];

# Make data set with progressive change in CF to cover changes in the calibration     
MA = pd.DataFrame(); IC = np.array([]); CF = np.array([]);
y = y15; lstm = model15


dro = hdrs[39:56]


for f in fil:
    ma = pd.read_csv(pth+f,sep='\t',parse_dates=[0], header=0, low_memory=False,encoding='latin1',
                       warn_bad_lines=True)
    
    ma[dro] = ma[dro].fillna(-999) 

    ma = lag_shift(ma,hdrs[17:],41)    
    ma = ma.dropna(axis= 0, how='any')
    
    # Trim the ends to remove possible bad data in files
    ma = ma.iloc[100:-100:1,:]
    
    ma[hdrs[7]] = wild_edit_2.despike(ma[hdrs[7]],2,10,20)
    ma[hdrs[5]] = wild_edit_2.despike(ma[hdrs[5]],2,10,20)
    ma[hdrs[8]] = wild_edit_2.despike(ma[hdrs[8]],2,10,20)
    
    ma[hdrs[8]] = ma[hdrs[8]].interpolate(method='linear', order=1, limit_direction='both')    

    Os = pd.DataFrame(S.O2sol(ma[hdrs[7]],ma[hdrs[5]]),columns=['Os'])
    Os['Os'] = wild_edit_2.despike(Os['Os'])
    
    mo = pd.concat([ma,Os],axis=1)
    
    MA = pd.concat([MA, mo])
    MA.reset_index(inplace=True)
    MA.drop(columns=('index'),inplace=True)
    MA.to_csv('SWIMS_Lat36_38.csv')
'''
MA = pd.read_csv(r'SWIMS_Lat36_38.csv')
    
    
#%%    
#X = ma[hdrs[idx]].dropna(axis=0, how='any')
X = MA[hdrs[idx]].interpolate(method='linear', order=1, limit_direction='both')

Xf = X.values
Xf_mean = Xf.mean(axis=0)
Xf_std = Xf.std(axis=0)
x = (Xf-Xf_mean)/Xf_std

pad = n_input-np.mod(len(x),n_input)
x = np.append(x,np.ones([pad+n_input,len(idx)]),axis=0)

trane = array(split(x, len(x)/n_input))

trane_x, trane_y = to_supervised(trane, n_input)

yhat = lstm.predict(trane_x,verbose=1)
#yr = pd.DataFrame(yhat[0:-pad-1,1,:])*Xf_std[-1]+Xf_mean[-1]
yr = pd.DataFrame(yhat[0:-pad-1,1,:])*ma[hdrs[target]].std()+ma[hdrs[target]].mean()

        
# Merge reconstructed target with the rest of the data      
MA = pd.concat([MA,yr],axis=1).dropna(how='any')

####  Calibration step.  ################

### Non-linear optimization to find best fit to SBE oxygen data #
## See Section 2.3 on calibration after bias removal ### 
t = minimize(optocf,1e-15,args=(y,MA[hdrs[target]]-MA[0],MA[hdrs[8]]),method='Nelder-Mead',tol=1e-13)
ic = t.x
IC = np.append(IC,ic)

# Equil. solubility at S=35.5 and T =23 C.
cf = 210.68/np.mean(y-ic)
CF = np.append(CF,cf);

yraw = yr; yraw.rename(columns={0:'yraw'},inplace=True)

# Apply
yrect = pd.DataFrame((MA[hdrs[target]]-yr.iloc[:,0]+ic)*cf,columns={'yrect'})

Ycorr = pd.concat([MA,yraw,yrect],axis=1)
    

#%%

Yp = Ycorr

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





plt.figure()
plt.plot(result.history['loss'],label='loss')
plt.plot(result.history['val_loss'],label='valid_loss')
plt.xlabel('epoch');
plt.legend()
plt.show()
    
