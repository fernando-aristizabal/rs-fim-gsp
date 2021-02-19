#!/usr/bin/env python3

from sys import path 
path.insert(0,'../') ; path.insert(0,'../../')

from src import gis
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def scatterPlots(vv_vh,rem,val,sampleFrac,xyzLim,catch=None,catchid=None,out_file=None):
    
    # import raster objects
    vv_vh = gis.rasterio.smart_open(vv_vh)
    rem = gis.rasterio.smart_open(rem)
    val = gis.rasterio.smart_open(val)

    # get common data boolean
    common_data_bool = np.all(np.stack([vv_vh.read_masks(1), rem.read_masks(1), val.read_masks(1)],axis=0),axis=0)

    if catchid is not None:
        catch = gis.rasterio.smart_open(catch)
        #catchid = np.random.choice(np.unique(catch.read(1)),1)
        #catchid = 3123708
        common_data_bool = np.all(np.stack([common_data_bool, catch.read(1) == catchid],axis=0),axis=0)

    # output this?

    # sample indices to plot
    sample_size = int(np.sum(common_data_bool) * sampleFrac)
    plotting_indices = np.random.choice( np.where(common_data_bool.ravel())[0],
                                         size= sample_size,
                                         replace=False
                                         )
    plotting_indices = np.sort(plotting_indices) # sort indices after random sampling

    # housekeeping
    del common_data_bool

    # read data
    vv_vals = np.take(vv_vh.read(1), plotting_indices)
    vh_vals = np.take(vv_vh.read(2), plotting_indices)
    rem_vals = np.take(rem.read(1),plotting_indices)
    val_vals = np.take(val.read(1),plotting_indices)

    # housekeeping
    del plotting_indices

    # subset by validation label
    dry_bool = val_vals == 1
    wet_bool = val_vals == 2

    dry_vv_vals = vv_vals[dry_bool]
    wet_vv_vals = vv_vals[wet_bool]
    del vv_vals

    dry_vh_vals = vh_vals[dry_bool]
    wet_vh_vals = vh_vals[wet_bool]
    del vh_vals

    dry_rem_vals = rem_vals[dry_bool]
    wet_rem_vals = rem_vals[wet_bool]
    del rem_vals
   
    del dry_bool, wet_bool

    """
    dataIndices = np.all(np.stack([np.array([vv_vh_hand_data.array[0,:,:] != vv_vh_hand_data.ndv]),
                                    np.array([vv_vh_hand_data.array[1,:,:] != vv_vh_hand_data.ndv]),
                                    np.array([vv_vh_hand_data.array[2,:,:] != vv_vh_hand_data.ndv]),
                                    np.array([vv_vh_hand_data.array[0,:,:] >= xyzLim[0]]), np.array([vv_vh_hand_data.array[0,:,:] <= xyzLim[1]]),
                                    np.array([vv_vh_hand_data.array[1,:,:] >= xyzLim[2]]), np.array([vv_vh_hand_data.array[1,:,:] <= xyzLim[3]]),
                                    np.array([vv_vh_hand_data.array[2,:,:] >= xyzLim[4]]), np.array([vv_vh_hand_data.array[2,:,:] <= xyzLim[5]])],axis=0),axis=0)
    dataIndices = dataIndices[0,:,:]
    dataIndices = np.where(dataIndices)

    indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=int(sampleFrac*len(dataIndices[0])),replace=False))
    rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
    columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
    dataToPlot = vv_vh_hand_data.array[:,rowIndices,columnIndices].T
    vv,vh,hand = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()
    val_vals = validationRaster.array[rowIndices,columnIndices]
    #landcoverLabel = landcoverRaster.array[rowIndices,columnIndices]
    #print(np.unique(landcoverLabel))
    """

    fig = plt.figure(figsize=(9,3))

    ax = fig.add_subplot(131)

    line1 = ax.scatter(dry_vv_vals, dry_vh_vals,c='brown',alpha=0.15,label="Non-Inundated")
    line2 = ax.scatter(wet_vv_vals, wet_vh_vals,c='blue',alpha=0.15,label="Inundated")

    ax.set_xlabel("VV Backscatter (dB)", rotation=0, size=12,labelpad=8)
    ax.set_ylabel("VH Backscatter (dB)", rotation=90, size=12,labelpad=8)

    ax.set_xlim(xyzLim[0:2])
    ax.set_ylim(xyzLim[2:4])

    ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
    ax.yaxis.set_ticks(np.arange(xyzLim[2], xyzLim[3], 5))

    ax.legend((line1,line2),'upper right')
    # fig1.savefig('manuscript/figures/2d_scatterPlot_vv_vh_{}_{}.tiff'.format(area,cl),format='tiff',dpi=300)

    ax = fig.add_subplot(132)

    line1 = ax.scatter(dry_vv_vals, dry_rem_vals,c='brown',alpha=0.15,label="Non-Inundated")
    line2 = ax.scatter(wet_vv_vals, wet_rem_vals,c='blue',alpha=0.15,label="Inundated")

    ax.set_xlabel("VV Backscatter (dB)", rotation=0, size=12,labelpad=8)
    ax.set_ylabel("HAND (m)", rotation=90, size=12,labelpad=8)

    ax.set_xlim(xyzLim[0:2])
    ax.set_ylim(xyzLim[4:6])

    ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
    ax.yaxis.set_ticks(np.arange(xyzLim[4], xyzLim[5], 5))

    ax.legend((line1,line2),'upper right')
    # fig2.savefig('manuscript/figures/2d_scatterPlot_vv_hand_{}_{}.tiff'.format(area,cl),format='tiff',dpi=300)

    ax = fig.add_subplot(133)
    fig.tight_layout(h_pad=2)

    line1 = ax.scatter(dry_vh_vals, dry_rem_vals,c='brown',alpha=0.15,label="Non-Inundated")
    line2 = ax.scatter(wet_vh_vals, wet_rem_vals,c='blue',alpha=0.15,label="Inundated")

    ax.set_xlabel("VH Backscatter (dB)", rotation=0, size=12,labelpad=8)
    ax.set_ylabel("HAND (m)", rotation=90, size=12,labelpad=8)

    ax.set_xlim(xyzLim[2:4])
    ax.set_ylim(xyzLim[4:6])

    ax.xaxis.set_ticks(np.arange(xyzLim[2], xyzLim[3], 5))
    ax.yaxis.set_ticks(np.arange(xyzLim[4], xyzLim[5], 5))

    ax.legend((line1,line2),'upper right')
    
    if out_file is not None:
        fig.savefig(out_file,format='jpeg',dpi=300)

    plt.close(fig)


if __name__ == "__main__":

    catch=gis.rasterio.smart_open("/data/rs-fim-gsp/texas/hand/processed/catchmask_brazos_upper.tif")
    catchids = np.unique(catch.read(1))
    catchids = np.append(catchids[catchids != catch.nodata],None)


    for catchid in tqdm(catchids):

        scatterPlots(vv_vh='/data/rs-fim-gsp/texas/sar/processed/sar_brazos_upper.tif',
                     rem="/data/rs-fim-gsp/texas/hand/processed/hand_brazos_upper.tif",
                     val="/data/rs-fim-gsp/texas/validation/processed/validation_brazos_upper.tif",
                     catch="/data/rs-fim-gsp/texas/hand/processed/catchmask_brazos_upper.tif",
                     catchid=catchid,
                     sampleFrac= 1,
                     xyzLim=[-30,-10,-20,-5,0,25],
                     out_file='/data/rs-fim-gsp/texas/results/figures/2d_scatterPlot_{}.jpg'.format(catchid))

