#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from raster import Raster
import numpy as np
import os

# directories and fileNames
dataDirectory = os.path.join('/home','fernandoa','projects','graphSignals','data','texas')
vv_input_raster_fileName = os.path.join(dataDirectory,'sar','processed','vv_cal_spk_gcp_sphProj_db_nd_cProj_masked.tif')
vh_input_raster_fileName = os.path.join(dataDirectory,'sar','processed','vh_cal_spk_gcp_sphProj_db_nd_cProj_masked.tif')
hand_input_raster_fileName = os.path.join(dataDirectory,'hand','processed','hand_merged_ndv_scaled_clipped_masked.tif')
catchmask_input_raster_fileName = os.path.join(dataDirectory,'hand','processed','catchmask_merged_ndv_scaled_clipped.tif')

vv_output_raster_fileName = vv_input_raster_fileName.split('.')[0] + "_masked.tif"
vh_output_raster_fileName = vh_input_raster_fileName.split('.')[0] + "_masked.tif"
hand_output_raster_fileName = hand_input_raster_fileName.split('.')[0] + "_masked.tif"
catchmask_output_raster_fileName = catchmask_input_raster_fileName.split('.')[0] + "_masked.tif"

# load files
print('loading rasters ...')
vv_raster = Raster(vv_input_raster_fileName)
vh_raster = Raster(vh_input_raster_fileName)
hand_raster = Raster(hand_input_raster_fileName)
catchmask_raster = Raster(catchmask_input_raster_fileName)

# mask
print('generating mask ....')
mask = np.logical_and(vv_raster.array != vv_raster.ndv,vh_raster.array != vh_raster.ndv)
mask = np.logical_and(mask,hand_raster.array != hand_raster.ndv)
mask = np.logical_and(mask,catchmask_raster.array != catchmask_raster.ndv)

mask = np.reshape(mask,(vv_raster.nrows,vv_raster.ncols)) 

# stack
#mask = np.all(np.stack((vv_raster.array != vv_raster.ndv,vh_raster.array != vh_raster.ndv,hand_raster.array != hand_raster.ndv,catchmask_raster.array != catchmask_raster.ndv),axis=2),axis=2)


print('masking ....')
#vv_raster.array[~mask] = vv_raster.ndv
#vh_raster.array[~mask] = vh_raster.ndv
#hand_raster.array[~mask] = hand_raster.ndv
print(np.unique(catchmask_raster.array).shape,catchmask_raster.ndv,catchmask_raster.array.dtype,catchmask_raster.dt)
catchmask_raster.array[~mask] = catchmask_raster.ndv
print(np.unique(catchmask_raster.array).shape,catchmask_raster.array.dtype,catchmask_raster.dt);exit()

# write out rasters
print('writing out ...')
#if os.path.isfile(vv_output_raster_fileName): os.remove(vv_output_raster_fileName)
#if os.path.isfile(vh_output_raster_fileName): os.remove(vh_output_raster_fileName)
#if os.path.isfile(hand_output_raster_fileName): os.remove(hand_output_raster_fileName)
if os.path.isfile(catchmask_output_raster_fileName): os.remove(catchmask_output_raster_fileName)
#print(vv_raster.array.shape,vh_raster.array.shape,hand_raster.array.shape,catchmask_raster.array.shape)
#vv_raster.writeRaster(vv_output_raster_fileName,dtype=np.float32)
#vh_raster.writeRaster(vh_output_raster_fileName,dtype=np.float32)
#hand_raster.writeRaster(hand_output_raster_fileName,dtype=np.float32)
catchmask_raster.writeRaster(catchmask_output_raster_fileName)
