#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:09:36 2023

@author: jinani

This code is a working example of estimating the receptive fields 
of a V1 neuron recorded in cats for the fist pass-without any cropping. 
First part of the  code loads the cell responses for natural images and 
carryout different preprocessing steps. Second part of the code implements 
the 2-pass ConvNet to estimate the model parameters.

Required functions can be found in /SysIden_AvgMethod_SupportFiles/k_functions.py
Custom layers for gaussian map layer and power law exponent is defined in /SysIden_AvgMethod_SupportFiles/k_layers.py
Implementation of the model with defined layers can be found in /SysIden_AvgMethod_SupportFiles/k_model.py

"""

# Import Required Libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
from IPython import get_ipython
get_ipython().magic('reset -sf') # To Clear Variables Before Script Runs

import scipy.io as sio
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy
import sys

# Import Custom Written Functions and Model
sys.path.insert(0,'/Users/jinani/Desktop/RFestimate_CNNwithPReLU/Real V1 Neuron/SysIden_AvgMethod_SupportFiles/') # path to the support files folder
from SysIden_AvgMethod_SupportFiles.k_functions import arrange_stimuli, arrange_responses,conv_output_length,plotGaussMap,plotReconstruction
from SysIden_AvgMethod_SupportFiles.k_model import model_pass1, model_pass2


# INPUT DETAILS HERE !
dataset_path='/Users/jinani/Desktop/RFestimate_CNNwithPReLU/Real V1 Neuron/H6214.010_2_Ch44/'
dataset_name='H6214.010_2_Ch44'
crop_x1=0
crop_x2=30
crop_y1=0
crop_y2=30
kernel_size = int((crop_x2-crop_x1)/30)
num_timelags=7
Stride=1
Filter_Size=11
Pool_Size=1 


# Arrange the Stimuli Dataset
Stimuli_dataset = sio.loadmat('/Users/jinani/Desktop/RFestimate_CNNwithPReLU/Real V1 Neuron/Stimuli_Data/DataSets_McGillClips_hc_downsampled480to30.mat')
estSet,regSet,predSet,imSize=arrange_stimuli(Stimuli_dataset,crop_y1,crop_y2,crop_x1,crop_x2,kernel_size,num_timelags)

# Arrange the Response Dataset
y_est,y_reg,y_pred=arrange_responses(dataset_path,dataset_name)


# Calculate Shape of the Main Input and Intermediate Layer Inputs 
Input_Shape=estSet.shape[1:]   
numRows =  Input_Shape[0]  
numCols =  Input_Shape[1]  
assert numRows == numCols
numFrames = Input_Shape[2] 

convImageSize = conv_output_length(numRows,Filter_Size,'valid',Stride) # Input to Conv2D Layer
downsampImageSize = conv_output_length(convImageSize,Pool_Size,'valid',Pool_Size) # Input to Gaussian Map Layer

# Model for Pass 1 : Filter Estimate Pass
model=model_pass1(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize)
model.summary()
optimizerFunction = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizerFunction)    
earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
history=model.fit(estSet,y_est, validation_data=(regSet,y_reg), epochs = 500,
              batch_size=750,callbacks=[earlyStop,mc],verbose=1)

# Pass 1 Trained Weights
weights = model.get_weights()

# Initialize Filter Weights for Second Pass
Initial_Filter_Weights=[weights[0],weights[1]] # Receptive Field Estimates from Pass 1
Initial_exp=np.asarray([1]) # Intialize Power Law Exponet to 1

# Model for Pass 2 : Power Law Pass
model2=model_pass2(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize,Initial_Filter_Weights,Initial_exp)
model2.summary()
optimizerFunction = keras.optimizers.Adam(lr=0.001)
model2.compile(loss='mse', optimizer=optimizerFunction)    
earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
history=model2.fit(estSet,y_est, validation_data=(regSet,y_reg), epochs = 500,
              batch_size=750,callbacks=[earlyStop,mc],verbose=1)


# Calculate VAF
predicted_test_response = model2.predict(predSet)
predicted_test_response = predicted_test_response.reshape(-1)
respTest=y_pred.reshape(-1)
R=np.corrcoef(predicted_test_response,respTest)
diag=R[0,1]
VAF_test=diag*diag*100
print (VAF_test)

# Pass 2 Trained Weights
weights2 = model2.get_weights()

# Plot Results
# 1. Learning Curve
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Learning Curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid()
plt.show()

# 2. Actual/Predicted Response Data
plt.figure()
plt.plot(respTest[0:100],color='r',label='Actual')
plt.plot(predicted_test_response[0:100],color='b',label='Estimated')
plt.legend(loc='upper right')
plt.grid()
plt.title("Response Data")
plt.show()

# 3. PReLU
plt.figure()
alpha1 = np.squeeze(weights2[2])
x = np.arange(-100,101)
y = np.arange(-100,101)
y[y<=0] = alpha1*y[y<=0] 
plt.plot(x,y)
plt.title('PReLU, alpha = {}'.format(np.round(alpha1,2)))
plt.show()

# 4. Gaussian Map
plt.figure()
mapMean = weights2[3]
mapSigma = weights2[4]
mapVals = plotGaussMap(mapMean,mapSigma,downsampImageSize)
plt.title('Gaussian Map')
plt.show()

# 5. Receptive Field Filter Weights
plt.figure()
filterWeights = weights[0][:,:,:,0]
numFrames = filterWeights.shape[2]
vmin = np.min(filterWeights)
vmax = np.max(filterWeights)
vabs = np.abs(filterWeights)
vabs_max = np.max(vabs)
for i in range(numFrames):
    plt.subplot(1,numFrames,i+1)
    plt.imshow(filterWeights[:,:,i],vmin=-vabs_max,vmax=+vabs_max)
plt.suptitle(' cell filter')

# 6. Reconstructed Receptive Field Filter 
plt.figure()
reconFilter=plotReconstruction(filterWeights,mapVals,Stride,Pool_Size,imSize[0])
plt.suptitle('Reconstruction of the linear filter')

# Save results in a .mat file
scipy.io.savemat('Results_AvgMethod.mat', {'weights_pass1': weights,'weights_pass2': weights2, 
                                           'Final_Rf_Construst': reconFilter,'VAF':VAF_test, 
                                           'Predicted_response':predicted_test_response})






