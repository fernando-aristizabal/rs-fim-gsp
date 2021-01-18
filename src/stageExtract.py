#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functions.raster import Raster
import geopandas as gpd
from tqdm import tqdm

def StageExtract(rem,catchmask,fim,streamNetwork,columnName,
				 encoding={'ndv':0,'non-inundated':1,'inundated':2},verbose=False):
	""" Stage extract from catchments and relative elevation """

	if verbose:
		print("Extracting stages ...")

	streamNetwork[columnName] = np.nan

	uniqueCatchmasks = np.unique(catchmask.array[catchmask.array != catchmask.ndv])
	lengthOfUniqueCatchmasks = len(uniqueCatchmasks)
	uniqueCatchmasks = iter(uniqueCatchmasks)

	for cm in tqdm(uniqueCatchmasks,disable=(not verbose),total=lengthOfUniqueCatchmasks):
		indicesOfCM = catchmask.array == cm
		CM_remValues = rem.array[indicesOfCM]
		CM_fimValues = fim.array[indicesOfCM]

		boolean_of_cm_fimValues = CM_fimValues == encoding['inundated']

		try:
			max_stage_value = np.max(CM_remValues[boolean_of_cm_fimValues])
		except ValueError:
			max_stage_value = 0

		#if np.any(boolean_of_cm_fimValues):
		#	max_stage_value = np.max(CM_remValues[boolean_of_cm_fimValues])
		#else:
		#	max_stage_value = 0

		# streamNetwork[columnName][streamNetwork['COMID'] == cm] = max_stage_value
		streamNetwork.loc[streamNetwork['COMID'] == cm,columnName] = max_stage_value
		#streamNetwork.loc[cm,columnName] = max_stage_value

	# filter out COMID's without catchments
	streamNetwork = streamNetwork.loc[~streamNetwork[columnName].isnull(),:]

	return(streamNetwork)
