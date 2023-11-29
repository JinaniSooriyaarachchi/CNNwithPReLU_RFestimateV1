#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:42:31 2022

@author: jinani

Gaussian map layer and power law exponent layer

"""
# Import Required Libraries
import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras

# 1. Gaussian map

matrix_inverse = tf.linalg.inv
tensordot = tf.tensordot
matrix_determinant = tf.linalg.det
    

class gaussian2dMapLayer(Layer):
    ''' assumes 2-d  input
        Only Allows Square 
    '''
    def __init__(self,input_dim=(3,3),init = 'zero',
                                 init_mean= None,
                                 init_sigma = None,                              
                                 
                                 **kwargs):
        print ('input dim')
        print(input_dim)
        assert input_dim[0] == input_dim[1],"Input must be square"

        self.input_dim = input_dim
        self.inv_scale = input_dim[0]
        
        #map the space of inputs, the values for the dot product will be pulled from a gaussian density
        xSpace = np.linspace(0,input_dim[0]-1,input_dim[0])
        ySpace = np.linspace(0,input_dim[1]-1,input_dim[1])
        spaceMatrix = np.asarray((np.meshgrid(xSpace,ySpace)))
        self.spaceVector = spaceMatrix.reshape((2,input_dim[0]*input_dim[1]))
        
        self.init = initializers.get(init) 
        
        if init_mean is None:
            half_mean = (1/2.)
            init_mean = np.asarray([half_mean,half_mean])
          
        if init_sigma is None:
            one_sig = (np.asarray(1.0))
            init_sigma =np.asarray([one_sig,
                                         np.asarray(0.0),
                                         one_sig])

        self.init_mean = init_mean.astype('float32')
        self.init_sigma = init_sigma.astype('float32')
        self.tolerance = np.asarray(0.01) #use tolerance from stopping the matrix from being un-invertible

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim)
        super(gaussian2dMapLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #print(input_shape)
        
        self.mean = self.add_weight(shape=(2,),  #I added shape= -JINANI-
                                    initializer=self.init,
                                    name='mean')
        self.sigma = self.add_weight(shape=(3,),
                            initializer=self.init,
                            name='sigma')
        if self.init_mean is not None:
            K.set_value(self.mean,self.init_mean)
            del self.init_mean
        if self.init_sigma is not None:
            K.set_value(self.sigma,self.init_sigma)
            del self.init_sigma
        self.built = True

    def call(self,x, mask=None):
        #print(np.shape(x))
        x = K.reshape(x,(K.shape(x)[0],K.shape(x)[-3]*K.shape(x)[-2]))
        #print(np.shape(x))
        covar = K.sign(self.sigma[1])*K.switch(K.sqrt(K.abs(self.sigma[0]*self.sigma[2]))-self.tolerance >= K.abs(self.sigma[1]),
                                                 K.abs(self.sigma[1]),
                                                 K.sqrt(K.abs(self.sigma[0]*self.sigma[2]))-self.tolerance )
        
        #Below is just the calculations for a Gaussian
        inner = (self.spaceVector - self.inv_scale*K.expand_dims(self.mean))

        cov = self.inv_scale*K.stack([[self.sigma[0],covar],[covar,self.sigma[2]]])
        
        inverseCov = matrix_inverse(cov)
        
        firstProd =  tensordot(K.transpose(inner),inverseCov,axes=1)
        malahDistance = K.sum(firstProd*K.transpose(inner),axis =1)
        gaussianDistance = K.exp((-1./2.)*malahDistance)
        detCov = matrix_determinant(cov)
        denom = 1./(2*np.pi*K.sqrt(detCov))
        gdKernel = tensordot(x,denom*gaussianDistance,axes=1)

        return K.expand_dims(gdKernel)


    def get_output_shape_for(self, input_shape):
        return (input_shape[-1],1)
    def compute_output_shape(self, input_shape):
        return (input_shape[-1],1)
    def get_config(self):
        base_config = super(gaussian2dMapLayer, self).get_config()
        return dict(list(base_config.items()))

        
# 2. Power law exponent
        
class PowerLowNonLinearity(keras.layers.Layer):
    def __init__(self,init = 'zero',init_exp= None,init_gain= None,**kwargs):
        super().__init__(**kwargs)
        self.init = initializers.get(init) 
        
        if init_exp is None:
            exp = (1.0)
            init_exp = np.asarray([exp])
            
        if init_gain is None:
            gain = (1.0)
            init_gain = np.asarray([gain])   
          
        self.init_exp = init_exp.astype('float32')
        self.init_gain = init_gain.astype('float32')

    def build(self, input_shape):

        self.exp = self.add_weight(shape=(1,),  #I added shape= -JINANI-
                                    initializer=self.init,
                                    name='exp')
        self.gain = self.add_weight(shape=(1,),  #I added shape= -JINANI-
                                    initializer=self.init,
                                    name='gain')

        if self.init_exp is not None:
            K.set_value(self.exp,self.init_exp)
            del self.init_exp
            
        if self.init_gain is not None:
            K.set_value(self.gain,self.init_gain)
            del self.init_gain

        self.built = True

    def call(self,x):
        selectPositive=tf.math.maximum(tf.constant([0.]), x)
        return (self.gain)*selectPositive**(self.exp)
        
    def get_config(self):
        base_config = super(PowerLowNonLinearity, self).get_config()
        return dict(list(base_config.items()))
