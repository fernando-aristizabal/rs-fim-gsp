#!/usr/bin/env python3

import numpy as np

def generateDifferenceRaster(predictedRaster,observedRaster,maskingRaster=None,maskingValue=None,output_fileName=None,mapping={'TP':1,'FP':2,'FN':3,'TN':4,'ndv':0}):

	if (maskingRaster is not None) | (maskingValue is not None):
		predictedRaster.array[maskingRaster.array != maskingValue] = predictedRaster.ndv
		observedRaster.array[maskingRaster.array != maskingValue] = observedRaster.ndv

	differenceRaster = copy.deepcopy(observedRaster)
	differenceRaster.array = np.zeros((differenceRaster.nrows,differenceRaster.ncols),dtype=int)

	differenceRaster.array[np.all(np.stack((predictedRaster.array==2,observedRaster.array==2)),axis=0)] = mapping['TP']
	differenceRaster.array[np.all(np.stack((predictedRaster.array==2,observedRaster.array==1)),axis=0)] = mapping['FP']
	differenceRaster.array[np.all(np.stack((predictedRaster.array==1,observedRaster.array==2)),axis=0)] = mapping['FN']
	differenceRaster.array[np.all(np.stack((predictedRaster.array==1,observedRaster.array==1)),axis=0)] = mapping['TN']
	differenceRaster.ndv = mapping['ndv']

	if output_fileName is not None:
		gis.writeGeotiff(differenceRaster,output_fileName,gdal.GDT_Byte)

	return(differenceRaster)


def calculateBinaryClassificationStatistics(differenceRaster,mapping={'TP':1,'FP':2,'FN':3,'TN':4,'ndv':0}):

	TP = (differenceRaster.array == mapping['TP']).sum()
	FP = (differenceRaster.array == mapping['FP']).sum()
	TN = (differenceRaster.array == mapping['TN']).sum()
	FN = (differenceRaster.array == mapping['FN']).sum()

	totalPopulation = TP + FP + TN + FN

	TP_perc = (TP / totalPopulation) * 100
	FP_perc = (FP / totalPopulation) * 100
	TN_perc = (TN / totalPopulation) * 100
	FN_perc = (FN / totalPopulation) * 100

	cellArea = abs(differenceRaster.gt[1] * differenceRaster.gt[5])

	TP_area = TP * cellArea
	FP_area = FP * cellArea
	TN_area = TN * cellArea
	FN_area = FN * cellArea

	totalPopulation = TP + FP + TN + FN
	predPositive = TP + FP
	predNegative = TN + FN
	obsPositive = TP + FN
	obsNegative = TN + FP

	predPositive_perc = predPositive / totalPopulation
	predNegative_perc = predNegative / totalPopulation
	obsPositive_perc = obsPositive / totalPopulation
	obsNegative_perc = obsNegative / totalPopulation

	predPositive_area = predPositive * cellArea
	predNegative_area = predNegative * cellArea
	obsPositive_area =  obsPositive * cellArea
	obsNegative_area =  obsNegative * cellArea

	positiveDiff = predPositive - obsPositive
	positiveDiff_area = predPositive_area - obsPositive_area
	positiveDiff_perc = predPositive_perc - obsPositive_perc

	prevalance = (TP + FN) / totalPopulation
	PPV = TP / predPositive
	NPV = TN / predNegative
	TPR = TP / obsPositive
	TNR = TN / obsNegative
	ACC = (TP + TN) / totalPopulation
	F1_score = (2*TP) / (2*TP + FP + FN)
	BACC = np.mean([TPR,TNR])
	MCC = (TP_area*TN_area - FP_area*FN_area)/ np.sqrt((TP_area+FP_area)*(TP_area+FN_area)*(TN_area+FP_area)*(TN_area+FN_area))
	CSI = ( ( (TPR)**-1 ) + ( (PPV)**-1 ) - 1)**-1

	stats = { 'TP' : TP,'FP' : FP,'TN' : TN,'FN' : FN,
			  'TP_perc' : TP_perc,'FP_perc' : FP_perc,
			  'TN_perc' : TN_perc,'FN_perc' : FN_perc,
			  'TP_area' : TP_area,'FP_area' : FP_area,
			  'TN_area' : TN_area,'FN_area' : FN_area,
			  'totalPopulation' : totalPopulation,
			  'predPositive' : predPositive,
			  'predNegative' : predNegative,
			  'obsPositive' : obsPositive,
			  'obsNegative' : obsNegative,
			  'prevalance' : prevalance,
			  'predPositive_perc' : predPositive_perc,
			  'predNegative_perc' : predNegative_perc,
			  'obsPositive_perc' : obsPositive_perc,
			  'obsNegative_perc' : obsNegative_perc,
			  'predPositive_area' : predPositive_area,
			  'predNegative_area' : predNegative_area,
			  'obsPositive_area' : obsPositive_area,
			  'obsNegative_area' : obsNegative_area,
			  'positiveDiff' : positiveDiff,
			  'positiveDiff_area' : positiveDiff_area,
			  'positiveDiff_perc' : positiveDiff_perc,
			  'PPV' : PPV,
			  'NPV' : NPV,
			  'TPR' : TPR,
			  'TNR' : TNR,
			  'ACC' : ACC,
			  'F1_score' : F1_score,
			  'BACC' : BACC,
			  'MCC' : MCC,
			  'CSI' : CSI}

	return(stats)
