#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from itertools import product
from sys import path
import os
projectDirectory = os.path.join('/rs-fim-gsp')
scriptsDirectory = os.path.join(projectDirectory,'src')
path.append(projectDirectory)
from src.raster import Raster
import pandas as pd
import geopandas as gpd
from time import time
from src.classifySAR import ClassifySAR
from src.stageExtract import StageExtract
#from src.stageExtract_numba import StageExtract
from src.graphFilterStages import GraphFilterStages
from src.inundationMapping import InundationMapping

# directories
dataDirectory = os.path.join('/data','rs-fim-gsp','texas')

# input filenames
sarData_fileName = os.path.join(dataDirectory,'sar','processed','sar_brazos_upper.tif')
#validation_fileName = os.path.join(dataDirectory,'validation','processed','finalInundation_Goldsboro.tiff')
remData_fileName = os.path.join(dataDirectory,'hand','processed','hand_brazos_upper.tif')
catchmentsData_fileName = os.path.join(dataDirectory,'hand','processed','catchmask_brazos_upper.tif')
streamNetwork_fileName = os.path.join(dataDirectory,'hand','processed','flowlines_final.shp')
weightMatrix_fileName= os.path.join(dataDirectory,'hand','processed','weightMatrix_brazos_upper.npy')
output_version_name = "brazos_upper_test_1"

# code parameters
verbose = True
extractedStages_columnName = 'sar_extracted_stages'[0:10]
filteredStages_columnName = 'filtered_stages'[0:10]
inundationEncoding = {'ndv':0,'non-inundated':1,'inundated':2}

# output filenames
sar_fim_fileName = os.path.join(dataDirectory,'results','sar_fim_{}.tif')
streamNetwork_outfile_fileName = os.path.join(dataDirectory,'results','flowlines_brazos_upper.gpkg')
filtered_fim_fileName = os.path.join(dataDirectory,'results','filtered_fim_{}.tif')

# load data
if verbose: print("Loading SAR data ...")
sarData = Raster(sarData_fileName)

######## experiment parameters #########

# em-gmm
sampleFraction = 0.01
covarianceType = 'full'
tolerance = 1e-3
regularizationCovariance = 1e-6

#gmmParamCombos = product(sampleFraction,covarianceType,tolerance,regularizationCovariance)

# graph filtering
filterName = 'MexicanHat'
filterParameters = {'Nf': 1}

#graphParamCombos = product(filterName,filterParameters)

######################### STEP 1: CLASSIFY SAR #############################
startTime = time()
sar_fim = ClassifySAR(sarData,sampleFraction=sampleFraction,covarianceType=covarianceType,tolerance=tolerance,
                      regularizationCovariance=regularizationCovariance,encoding=inundationEncoding,verbose=verbose
                     )
classifySAR_time = time() - startTime

if verbose: 
    print("Classify SAR Time: {} \n Writing raster ...".format(classifySAR_time)) 

sar_fim.writeRaster(sar_fim_fileName.format(output_version_name))
#sar_fim = Raster(sar_fim_fileName.format('TEST'))

# load data
if verbose:
    print("Loading REM, Catchments, and Stream Networks ...")
remData = Raster(remData_fileName)
catchmentsData = Raster(catchmentsData_fileName)
streamNetwork = gpd.read_file(streamNetwork_fileName)
streamNetwork.set_index('COMID',drop=False,inplace=True)

######################### STEP 2: STAGE EXTRACTION ########################
startTime = time()
streamNetwork_outfile = StageExtract(rem=remData,catchmask=catchmentsData,fim=sar_fim,streamNetwork=streamNetwork,
                                     columnName=extractedStages_columnName,encoding=inundationEncoding,verbose=verbose
                                    )
stageExtract_time = time() - startTime

if verbose: 
    print("Stage Extract Time: {} \n Writing raster ...".format(stageExtract_time)) 

streamNetwork_outfile.to_file(streamNetwork_outfile_fileName.format(output_version_name),index=False,driver='GPKG')
#streamNetwork_outfile = gpd.read_file(streamNetwork_outfile_fileName.format('TEST'))

############################# STEP 3: GRAPH FILTERING #####################
startTime = time()
streamNetwork_outfile = GraphFilterStages(streamNetwork=streamNetwork_outfile,
                                          extractedStages_columnName=extractedStages_columnName,
                                          filteredStages_columnName=filteredStages_columnName,filterName=filterName,
                                          filterParameters=filterParameters,streamNetwork_vaa_fileName=None,
                                          weightMatrix_fileName=weightMatrix_fileName,useWeights=False,
                                          verbose=verbose
                                          )
graphFilterStages_time = time() - startTime

if verbose: 
    print("Filtering Stages: {} \n Writing raster ...".format(graphFilterStages_time)) 

streamNetwork_outfile.to_file(streamNetwork_outfile_fileName.format(output_version_name),index=False,driver='GPKG')

############################# STEP 4: MAPPING #############################
startTime = time()
filtered_fim = InundationMapping(rem=remData,catchmask=catchmentsData,streamNetwork=streamNetwork_outfile,
                                 stages_columnName=filteredStages_columnName,encoding=inundationEncoding,
                                 verbose=verbose
                                )
inundationMapping_time = time() - startTime

if verbose: 
    print("Inundation Mapping Time: {} \n Writing raster ...".format(inundationMapping_time)) 

filtered_fim.writeRaster(filtered_fim_fileName.format(output_version_name))
