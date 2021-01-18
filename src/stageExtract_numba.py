#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functions.raster import Raster
import geopandas as gpd
from tqdm import tqdm
from numba import jit

def StageExtract(rem,catchmask,fim,streamNetwork,columnName,
                 encoding={'ndv':0,'non-inundated':1,'inundated':2},verbose=False):
    """ Stage extract from catchments and relative elevation """

    if verbose:
        print("Extracting stages ...")

    streamNetwork[columnName] = np.nan


    @jit(nopython=True,parallel=True)
    def go_fast_extractions(remArray,catchmaskArray,fimArray,catchmaskNDV,inundationEncoding):
    
        uniqueCatchmasks = np.unique(catchmaskArray[catchmaskArray != catchmaskNDV])
        lengthOfUniqueCatchmasks = len(uniqueCatchmasks)
        #maxStages = np.zeros(lengthOfUniqueCatchmasks,dtype=np.float32)
        maxStages = [0] * lengthOfUniqueCatchmasks
        
        for idx,cm in enumerate(uniqueCatchmasks):
            
            if idx % 10 == 0: print(idx)
            
            indicesOfCM = catchmaskArray == cm
            CM_remValues = remArray[indicesOfCM]
            CM_fimValues = fimArray[indicesOfCM]

            boolean_of_cm_fimValues = CM_fimValues == inundationEncoding

            if np.any(boolean_of_cm_fimValues):
                max_stage_value = np.max(CM_remValues[boolean_of_cm_fimValues])
                maxStages[idx] = max_stage_value
            
        return(maxStages)

    maxStages = go_fast_extractions(rem.array.ravel(),catchmask.array.ravel(),fim.array.ravel(),catchmask.ndv,encoding['inundated'])
    
    for idx,cm in enumerate(uniqueCatchmasks):
        streamNetwork[streamNetwork['COMID']==cm,'COMID'] = maxStages[idx]
    
    # filter out COMID's without catchments
    streamNetwork = streamNetwork.loc[~streamNetwork[columnName].isnull(),:]

    return(streamNetwork)
