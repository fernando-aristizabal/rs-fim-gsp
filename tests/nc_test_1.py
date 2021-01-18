#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from itertools import product
from raster import Raster
import os
import pandas as pd
import geopandas as gpd
from time import time
from functions.classifySAR import ClassifySAR
from functions.stageExtract import StageExtract
from functions.graphFilterStages import GraphFilterStages
from functions.inundationMapping import InundationMapping

# directories
projectDirectory = os.path.join('/home','lqc','Documents','research','graphSignals')
scriptsDirectory = os.path.join(projectDirectory,'scripts')
dataDirectory = os.path.join(projectDirectory,'data')

# input filenames
sarData_fileName = os.path.join(dataDirectory,'results','vv_vh_hand_Goldsboro.tiff')
validation_fileName = os.path.join(dataDirectory,'validation','processed','finalInundation_Goldsboro.tiff')
remData_fileName = os.path.join(dataDirectory,'results','vv_vh_hand_Goldsboro.tiff')
catchmentsData_fileName = os.path.join(dataDirectory,'hand','processed','catchmask_proj_Goldsboro_nodata_proj_scaled_Goldsboro.tiff')
streamNetwork_fileName = os.path.join(dataDirectory,'nhd','NHDFlowline_goldsboro_TEST.shp')
streamNetwork_vaa_fileName = os.path.join(dataDirectory,'nhd','PlusFlowlineVAA.dbf')

# code parameters
verbose = True
extractedStages_columnName = 'sar_extracted_stages'
filteredStages_columnName = 'filtered_stages'
inundationEncoding = {'ndv':0,'non-inundated':1,'inundated':2}

# output filenames
sar_fim_fileName = os.path.join(dataDirectory,'results','sar_fim_{}.tif')
streamNetwork_outfile_fileName = os.path.join(dataDirectory,'nhd','NHDFlowline_goldsboro_{}_outfile.shp')
filtered_fim_fileName = os.path.join(dataDirectory,'results','filtered_fim_{}.tif')

# load data
sarData = Raster(sarData_fileName)
remData = Raster(remData_fileName)
catchmentsData = Raster(catchmentsData_fileName)
streamNetwork = gpd.read_file(streamNetwork_fileName)

######## experiment parameters #########

# em-gmm
sampleFraction = 1
covarianceType = 'full'
tolerance = 1e-3
regularizationCovariance = 1e-6

# graph filtering
filterName = 'abspline'
filterParameters = {'Nf': 1}

# pd.DataFrame()

######################### STEP 1: CLASSIFY SAR #############################
sarData.array = sarData.array[0:2,:] ; sarData.nbands = 2
startTime = time()
sar_fim = ClassifySAR(sarData,sampleFraction=sampleFraction,covarianceType=covarianceType,tolerance=tolerance,
					  regularizationCovariance=regularizationCovariance,encoding=inundationEncoding,verbose=verbose)
classifySAR_time = time() - startTime
sar_fim.writeRaster(sar_fim_fileName.format('TEST'))

######################### STEP 2: STAGE EXTRACTION ########################
remData.array = remData.array[2,:] ; remData.nbands = 1
startTime = time()
streamNetwork_outfile = StageExtract(rem=remData,catchmask=catchmentsData,fim=sar_fim,streamNetwork=streamNetwork,
									 columnName=extractedStages_columnName,encoding=inundationEncoding,verbose=verbose)
stageExtract_time = time() - startTime
streamNetwork_outfile.to_file(streamNetwork_outfile_fileName.format('TEST'))

############################# STEP 3: GRAPH FILTERING #####################
startTime = time()
streamNetwork_outfile = GraphFilterStages(streamNetwork=streamNetwork_outfile,extractedStages_columnName=extractedStages_columnName,
										  filteredStages_columnName=filteredStages_columnName,filterName=filterName,filterParameters=filterParameters,
										  streamNetwork_vaa_fileName=None,useWeights=False,verbose=verbose)
graphFilterStages = time() - startTime
streamNetwork_outfile.to_file(streamNetwork_outfile_fileName.format('TEST'))

############################# STEP 4: MAPPING #############################
startTime = time()
filtered_fim = InundationMapping(rem=remData,catchmask=catchmentsData,streamNetwork=streamNetwork_outfile,
								 stages_columnName=filteredStages_columnName,encoding=inundationEncoding,verbose=verbose)
inundationMapping_time = time() - startTime
filtered_fim.writeRaster(filtered_fim_fileName.format('TEST'))
