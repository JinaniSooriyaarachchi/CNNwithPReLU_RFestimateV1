# RFestimate_CNNwithPReLU

# Introduction
This repository contains Python code for a working example of our receptive field estimation method, using a simple convolutional neural network approach. The method is described in detail, with results on many cortical neurons, in our paper currently submitted for publication (“Estimating receptive fields of simple and complex cells in early visual cortex: A convolutional neural network model with parameterized rectification”, by Philippe Nguyen, myself and Curtis Baker).
Data files to generate figures given in the paper are provided in /Paper_figureDataFiles folder.

This repository contains code to run system identification with a real V1 neuron and a simulated simple/complex neuron.

For the simulated neuron scenario: 
This generates responses of simulated model simple and complex V1 cells, to natural image. For each simulated neuron, it estimates the parameters of a receptive field model consisting of a single spatiotemporal filter, a parameterized rectified linear unit (PReLU), and a two-dimensional Gaussian "map" of the receptive field envelope. I've commented throughout the primary script, simulateAndRun_SysIden.py, so that you can follow what is occuring. The code generates two gabor filters, and uses those to create two model V1 neurons, to simulate a simple cell and a complex cell. Most of the model-building occurs in the k_* files. Look through them to see how the models are set up. The code also uses natural image stimuli from the McGill Colour Image Database (http://tabby.vision.mcgill.ca/) - in the latter case, the images have been subdivided, converted to greyscale, and normalized (see Talebi & Baker, 2012), and downsampled. Once the stimuli and neuron responses are generated, we estimate the model parameters, and then plot the results.

For the real neuron scenario: 
Run SystemIdentification_AvgMethod.py file. The stimuli data and neuronal response data are provided in sub folders.

# Requirements
Python 3., scipy, numpy, matplotlib tensorflow 2.0, Keras. Check the environment.ym file for detailed list of dependencies.
If you have an NVIDIA card, the libraries CUDA and CUDNN will be useful. https://developer.nvidia.com/cuda-zone https://developer.nvidia.com/cudnn 

# Notes
The complex cell response is generated using a simplified version of an "energy" model (Adelson & Bergen, 1975). Since our model architecture used for the estimation differs in some significant details from this energy-type model, our method does not exactly estimate its parameters. Nevertheless, our method still performs well on this model. One could generate a complex cell model by using spatially separated identical simple cell subunits, and replace the squaring operation with a full-wave rectification - in that case our method should perform even better.

Note that the values of the Gaussian Map Layer correspond to the convolved image (the input to the gaussian map layer), and not image space itself.
