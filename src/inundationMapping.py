#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from src.raster import Raster
import geopandas as gpd
from tqdm import tqdm

def InundationMapping(rem,catchmask,streamNetwork,stages_columnName,
                      encoding={'ndv':0,'non-inundated':1,'inundated':2},verbose=False):
    """ Stage extract from catchments and relative elevation """

    if verbose:
        print("Mapping inundation ...")

    # initialize inundation array and save parameters
    inundation = rem.copy()
    inundation.ndv = encoding['ndv']
    inundation.array = np.zeros((rem.nrows,rem.ncols),dtype=np.int8) + inundation.ndv

    # boolean of data values
    # booleanOfDataValues = rem.array != rem.ndv

    # get iterator of unique catchments
    uniqueCatchmasks = np.unique(catchmask.array[catchmask.array != catchmask.ndv])
    lengthOfUniqueCatchmasks = len(uniqueCatchmasks)
    uniqueCatchmasks = iter(uniqueCatchmasks)
    
    for cm in tqdm(uniqueCatchmasks,disable=(not verbose),total=lengthOfUniqueCatchmasks):

        # boolean of catchment pixels
        booleanCatchmask = catchmask.array == cm

        # get index of current comid
        current_comid_index = streamNetwork.index[streamNetwork['COMID'] == cm].to_list()

        # get stage values
        try:
            stageValue = streamNetwork.loc[current_comid_index,stages_columnName].values[0]
        except IndexError:
            stageValue = 0

        # boolean of below water pixels
        booleanBelowWater = rem.array <= stageValue
        booleanAboveWater = rem.array > stageValue

        # inundated and non-inundated booleans
        # inundatedBoolean = np.logical_and(booleanBelowWater,booleanCatchmask,booleanOfDataValues)
        # nonInundatedBoolean = np.logical_and(booleanAboveWater,booleanCatchmask,booleanOfDataValues)
        inundatedBoolean = np.logical_and(booleanBelowWater,booleanCatchmask)
        nonInundatedBoolean = np.logical_and(booleanAboveWater,booleanCatchmask)

        # assign values to inundation array
        inundation.array[nonInundatedBoolean] = encoding['non-inundated']
        inundation.array[inundatedBoolean] = encoding['inundated']


    return(inundation)
