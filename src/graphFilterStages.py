#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functions.raster import Raster
import geopandas as gpd
from pygsp import graphs, filters
from os.path import isfile

__filter_dictionary = { 'abspline' : filters.Abspline }

def GraphFilterStages(streamNetwork,extractedStages_columnName,filteredStages_columnName,filterName='abspline',
                                          filterParameters = {'Nf': 1},streamNetwork_vaa_fileName=None,weightMatrix_fileName=None,useWeights=False,verbose=False):

    if verbose:
        print("Filtering stages ...")

    if (weightMatrix_fileName is None) | (not isfile(weightMatrix_fileName)):

        # initialize weight matrix
        numberOfCOMIDS = len(streamNetwork['COMID'])
        weightMatrix = np.zeros((numberOfCOMIDS,numberOfCOMIDS),dtype=np.int8)

        # generate weighted weight matrix
        if useWeights:

            ## generate weight matrix ##
            for rowIndex,current_comid in enumerate(streamNetwork['COMID']):

                current_comid_index = streamNetwork.index[streamNetwork['COMID'] == current_comid].to_list()

                # get to node
                toNode = streamNetwork.loc[current_comid_index,"ToNode"]
                # fromNode = streamNetwork.loc[current_comid,"FromNode"]

                toMatches = (streamNetwork.loc[:,"FromNode"] == toNode).astype(np.int8)
                # fromMatches = (streamNetwork.loc[:,"ToNode"] == fromNode).astype(np.int8)

                booleanOf_toMatches = toMatches == 1

                if booleanOf_toMatches.sum() > 0:
                    toMatches_DA = np.array([],dtype=np.float32)
                    for tm in toMatches[booleanOf_toMatches].index.values:
                        #print(current_comid,tm)
                        current_tm_index = streamNetwork.index[streamNetwork['COMID'] == tm].to_list()
                        toMatches_DA = np.append(toMatches_DA,[1/abs(streamNetwork.loc[current_comid_index,'TotDASqKM']-streamNetwork.loc[current_tm_index,'TotDASqKM'])])

                        # toMatches_normalized_DA = toMatches_DA / np.max(toMatches_DA)
                        # weightMatrix[rowIndex,booleanOf_toMatches] = toMatches_normalized_DA

                        weightMatrix[rowIndex,booleanOf_toMatches] = toMatches_DA

        else:
            ## generate weight matrix as adjacency ##
            for rowIndex,current_comid in enumerate(streamNetwork['COMID']):

                current_comid_index = streamNetwork.index[streamNetwork['COMID'] == current_comid].to_list()

                # get to node
                toNode = streamNetwork.loc[current_comid_index,"ToNode"].values[0]
                #toNode = streamNetwork.loc[current_comid,"ToNode"].values[0]
                # fromNode = streamNetwork.loc[current_comid,"FromNode"]

                # assign adjacency to weight matrix
                weightMatrix[rowIndex,:] = (streamNetwork.loc[:,"FromNode"] == toNode).astype(np.int8)

            # make undirected
            weightMatrix = np.maximum(weightMatrix,weightMatrix.T)
        
        np.save(weightMatrix_fileName,weightMatrix)
    else:
        weightMatrix = np.load(weightMatrix_fileName)

    # intialize graph and signals
    graphInstance = graphs.Graph(weightMatrix,coords=None)
    signal = streamNetwork[extractedStages_columnName].values

    # estimate lmax
    graphInstance.estimate_lmax()

    # graph filters
    filterClass = __filter_dictionary[filterName]
    filterInstance = filterClass(graphInstance,**filterParameters)

    # filter signal
    filteredSignal = filterInstance.filter(signal)

    # sum across eigenvalues
    if len(filteredSignal.shape) == 2:
        filteredSignal = filteredSignal.sum(axis=1)
        # filteredSignal = filteredSignal[:,:].sum(axis=1)

    # add filtered
    streamNetwork[filteredStages_columnName] = filteredSignal

    return(streamNetwork)
