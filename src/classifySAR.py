#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functions.raster import Raster
from sklearn.mixture import GaussianMixture


def ClassifySAR(sarData,sampleFraction,covarianceType='full',tolerance=1e-3,regularizationCovariance=1e-6,
				encoding={'ndv':0,'non-inundated':1,'inundated':2},verbose=False):

	if verbose:
		print("classifying SAR ...")

	# save important values of sar data
	booleanOfDatainArray = np.all(sarData.array != sarData.ndv,axis=0)
	sarDataNDV = sarData.ndv

	# intialize predicted inundation array
	predictedInundation = sarData.copy()
	predictedInundation.ndv = encoding['ndv']
	predictedInundation.array = np.zeros((predictedInundation.nrows,predictedInundation.ncols),dtype=np.uint8) + predictedInundation.ndv

	# reshape to number of samples, number of features
	# sarData = sarData.array.reshape(sarData.ncols*sarData.nrows,sarData.nbands)
	sarData = np.vstack((sarData.array[0,:,:].ravel(),sarData.array[1,:,:].ravel())).T

	# get indices of data
	indicesOfData = np.where(np.all(sarData != sarDataNDV,axis=1))[0]
	numberOfGoodDataPixels = len(indicesOfData)

	# get data samples
	sampleSize = int(numberOfGoodDataPixels * sampleFraction)
	indicesOfSampledData = np.random.choice(indicesOfData,size=sampleSize,replace=False)

	# initialize gmm
	gmm = GaussianMixture(n_components=2, covariance_type=covarianceType, tol=tolerance, reg_covar=regularizationCovariance,
					max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None,
					precisions_init=None, random_state=None, warm_start=False, verbose=int(verbose), verbose_interval=1)

	# fit and predict
	gmm.fit(sarData[indicesOfSampledData,:])
	predictedLabels = gmm.predict(sarData[indicesOfData,:])

	# fix flipped data and classify to encoding
	meanByComponent = gmm.means_.mean(axis=1)
	component0_boolean = predictedLabels == 0 ; component1_boolean = predictedLabels == 1

	if meanByComponent[0] > meanByComponent[1]:
		predictedLabels[component0_boolean] = encoding['non-inundated']
		predictedLabels[component1_boolean] = encoding['inundated']
	else:
		predictedLabels[component0_boolean] = encoding['inundated']
		predictedLabels[component1_boolean] = encoding['non-inundated']

	# fill predicted inundation array
	predictedInundation.array[booleanOfDatainArray] = predictedLabels

	return(predictedInundation)
