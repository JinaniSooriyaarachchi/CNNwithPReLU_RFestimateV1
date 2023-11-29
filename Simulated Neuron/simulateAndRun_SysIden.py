#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:43:01 2023

@author: jinani

This code is a working example of estimating the receptive fields 
of a simulated simple and complex cell. 
First part of the  code simulate the cell responses for natural images and 
carryout different preprocessing steps. Second part of the code implements 
the 2-pass ConvNet to estimate the model parameters.

Required functions can be found in /SupportFiles/k_functions.py
Custom layers for gaussian map layer and power law exponent is defined in /SupportFiles/k_layers.py
Implementation of the model with defined layers can be found in /SupportFiles/k_model.py

"""


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from SupportFiles.k_functions import buildCalcAndScale, kConvNetStyle,conv_output_length,plotGaussMap,plotReconstruction, generateSimpleCellResponse, generateComplexCellResponse, generateGabors, getNatStim 
from SupportFiles.k_model import model_pass1, model_pass2


# Load natural images from NaturalImages.mat file. These are 30x30 pixels 11250 grey scale images
stim = getNatStim()
movieSize = np.shape(stim)
imSize = (movieSize[0],movieSize[1])
movieLength = movieSize[2]


# Set up neuron Gabors. These are the model RFs (ground truth)
xCenter = 20.0
yCenter= 8.0
sf = 0.10
ori = 90.0
env = 2.0
gabor0,gabor90 = generateGabors(imSize,xCenter,yCenter,sf,ori,env)

# Visualize ground truth RFs
plt.figure()
#mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(1005,0,600, 300)
plt.subplot(1,2,1)
plt.imshow(gabor0)
plt.title('gabor0')
plt.subplot(1,2,2)
plt.imshow(gabor90)
plt.title('gabor90')
plt.suptitle(' Actual Receptive Field', fontweight ='bold', fontsize = 15)


# Generate cell responses for a user-chosen cell type
cellQuery = input('Model Simple Cell? Model Complex Cell otherwise. (y/n) \n')
if cellQuery.lower() =='y' or cellQuery.lower()=='yes':
    print ('Modelling Simple Cell')
    CellResponse=generateSimpleCellResponse(gabor0, stim, noiseAmount=0.5, exponent=1.25)
else:
    print ('Modelling Complex Cell')
    CellResponse=generateComplexCellResponse(gabor0, gabor90, stim, noiseAmount=0.5, exponent=1.25)
    
    
# Set up training, validation and testing datasets.
stim = np.transpose(np.reshape(stim,(imSize[0]*imSize[1],movieLength))) 

estIdx = slice(0,int(0.8*movieLength))                     # 80% for training
regIdx = slice(int(0.8*movieLength),int(0.9*movieLength))  # 10% for validation
predIdx = slice(int(0.9*movieLength),movieLength)          # 10% for testing

estSet, regSet, predSet=stim[estIdx], stim[regIdx], stim[predIdx]   # stimuli data
y_est, y_reg, y_pred=CellResponse[estIdx], CellResponse[regIdx], CellResponse[predIdx] # response data    


# Preprocessing: Normalize the datasets
estSet, regSet, predSet = (buildCalcAndScale(dataSet) for dataSet in [estSet,regSet,predSet])


# Add temporal dynamics to data
Frames =list(range(4))
estSet, regSet, predSet = (kConvNetStyle(dataSet,Frames) for dataSet in [estSet,regSet,predSet])


# dataset sizes
Input_Shape=estSet.shape[1:]  
numRows =  Input_Shape[0] 
numCols =  Input_Shape[1] 
assert numRows == numCols
numFrames = Input_Shape[2]

Stride=1
Filter_Size=15
convImageSize = conv_output_length(numRows,Filter_Size,'valid',Stride) # Output from Conv2D Layer


# Model for Pass 1 : Filter Estimate Pass
model=model_pass1(Input_Shape,Filter_Size,Stride,convImageSize)
model.summary()
optimizerFunction = keras.optimizers.Adam(lr=0.005)
model.compile(loss='mse', optimizer=optimizerFunction)    
earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
history=model.fit(estSet,y_est, validation_data=(regSet,y_reg), epochs = 200,
              batch_size=750,callbacks=[earlyStop,mc],verbose=1)

# Pass 1 Trained Weights
weights = model.get_weights()

# Initialize Filter Weights for Second Pass
Initial_Filter_Weights=[weights[0],weights[1]] # Receptive Field Estimates from Pass 1
Initial_exp=np.asarray([1]) # Intialize Power Law Exponet to 1

# Model for Pass 2 : Power Law Pass
model2=model_pass2(Input_Shape,Filter_Size,Stride,convImageSize,Initial_Filter_Weights,Initial_exp)
model2.summary()
optimizerFunction = keras.optimizers.Adam(lr=0.005)
model2.compile(loss='mse', optimizer=optimizerFunction)    
earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
history=model2.fit(estSet,y_est, validation_data=(regSet,y_reg), epochs = 200,
              batch_size=750,callbacks=[earlyStop,mc],verbose=1)


# Calculate VAF
predicted_test_response = model2.predict(predSet)
predicted_test_response = predicted_test_response.reshape(-1)
respTest=y_pred.reshape(-1)
R=np.corrcoef(predicted_test_response,respTest)
diag=R[0,1]
VAF_test=diag*diag*100
print('\n')
print ('VAF = ',VAF_test)

# Pass 2 Trained Weights
weights2 = model2.get_weights()


# Plot Results
# 1. Learning Curve
plt.figure()
#mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(0,0,1000, 750)
plt.subplot(2,3,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Learning Curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.grid()
plt.show()

# 2. Actual/Predicted Response Data
plt.subplot(2,3,2)
plt.plot(respTest[0:100],color='r',label='Actual')
plt.plot(predicted_test_response[0:100],color='b',label='Estimated')
plt.legend(loc='upper right')
plt.grid()
plt.title("Response Data")
plt.show()

# 3. PReLU
plt.subplot(2,3,3)
alpha1 = np.squeeze(weights2[2])
x = np.arange(-100,101)
y = np.arange(-100,101)
y[y<=0] = alpha1*y[y<=0] 
plt.plot(x,y)
plt.title('PReLU, alpha = {}'.format(np.round(alpha1,2)))
plt.show()

# 4. Gaussian Map
plt.subplot(2,3,4)
mapMean = weights2[3]
mapSigma = weights2[4]
mapVals = plotGaussMap(mapMean,mapSigma,convImageSize)
plt.title('Gaussian Map')
plt.show()

# 5. PowerLaw exponent 
plt.subplot(2,3,5)
pl = np.squeeze(weights2[7])
x = np.arange(-100,101)
y = np.arange(-100,101)
y[y<=0] = 0
y[y>0] = x[x>0]**pl
plt.plot(x,y)
plt.title('Power Law = {}'.format(np.round(pl,2)))
plt.show()
plt.suptitle('Model parameters', fontweight ='bold', fontsize = 15)
plt.tight_layout()

# Receptive Field Filter Weights
plt.figure()
#mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(1005,0,600, 300)
filterWeights = weights[0][:,:,:,0]
numFrames = filterWeights.shape[2]
vmin = np.min(filterWeights)
vmax = np.max(filterWeights)
vabs = np.abs(filterWeights)
vabs_max = np.max(vabs)
for i in range(numFrames):
    plt.subplot(1,numFrames,i+1)
    plt.imshow(filterWeights[:,:,i],vmin=-vabs_max,vmax=+vabs_max)
plt.tight_layout()    
plt.suptitle(' Filter weights', fontweight ='bold', fontsize = 15)

# Reconstructed Receptive Field Filter 
plt.figure()
#mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(1005,400,600, 300)
reconFilter=plotReconstruction(filterWeights,mapVals,Stride,imSize[0])
plt.tight_layout()   
plt.suptitle('Reconstruction of the linear filter', fontweight ='bold', fontsize = 15)

