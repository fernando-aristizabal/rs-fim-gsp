import os
import sys
from importlib import reload
from getpass import getuser

# append path
projectDirectory = os.path.join('/home',getuser(),'Documents','research','nwcUnsupervised')
os.chdir(projectDirectory)
sys.path.append(os.path.join(projectDirectory ,'src'))

import gis
from analysis import generateDifferenceRaster, calculateBinaryClassificationStatistics
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import time
from sklearn.metrics import calinski_harabaz_score,accuracy_score,confusion_matrix
from itertools import product
import copy
from gdal import GDT_Byte
import matplotlib.pyplot as plt
from pprint import pprint
import geopandas as gpd
from pygsp import graphs, filters, plotting
from dbfread import DBF


"""
###################################################################################################################################
Import data, preprocess, and set experiment parameters
"""

version = "0"

# file names
resultsTableFilename = 'data/results/resultsTable_v{}.pkl'.format(version)
vv_vh_hand_filename = 'data/results/vv_vh_hand_Goldsboro.tiff'
validationRaster_filename = 'data/validation/processed/finalInundation_Goldsboro.tiff'
predictedInundation_filename_guide = os.path.join("data","results","predictedInundation_{}_{}_{}_{}_{}_v{}.tiff")
predictedInundation_HAND_filename_guide = os.path.join("data","results","predictedInundation_HAND_{}_{}_{}_{}_{}_v{}.tiff")
difference_filename_guide = os.path.join("data","results","difference_{}_{}_{}_{}_{}_v{}.tiff")
difference_HAND_filename_guide = os.path.join("data","results","difference_HAND_{}_{}_{}_{}_{}_v{}.tiff")
catchmask_filename = os.path.join("data","hand","processed","catchmask_proj_Goldsboro_nodata_proj_scaled_Goldsboro.tiff")
validationCatchMasksTable_filename = os.path.join("data","results","validationCatchMasksTable_v{}.pkl".format(version))
flowlinesvaa_filename = os.path.join("data","nhd","PlusFlowlineVAA.dbf")
flowlines_filename_guide = os.path.join("data","nhd","NHDFlowline_goldsboro.{}")
flowlines_heights_filename_guide = os.path.join("data","nhd","NHDFlowline_goldsboro_heights.{}")
flowlines_heights_filtered_filename_guide = os.path.join("data","nhd","NHDFlowline_goldsboro_heights_filtered.{}")
merged_dataframe_filename = os.path.join("data","nhd","merged_flowlines_dataframe_v{}.pkl").format(version)
merged_inundated_dataframe_filename = os.path.join("data","nhd","merged_inundated_flowlines_dataframe_v{}.pkl").format(version)
validationFromMaxHANDvalues_filename = os.path.join("data","results","validationFromMaxHANDvalues.tiff")
validationFromMaxHANDvalues_difference_filename = os.path.join("data","results","validationFromMaxHANDvalues_difference.tiff")
drainageAreas_fileName = os.path.join('data','nhd','NHDPlusAttributes','CumulativeArea.dbf')


## input data
vv_vh_hand = gis.raster(vv_vh_hand_filename)
validationRaster = gis.raster(validationRaster_filename)

## determine good data
goodIndices_array = np.where(validationRaster.array != validationRaster.ndv)
vv_vh_hand_data = np.vstack((vv_vh_hand.array[0,:,:].ravel(),vv_vh_hand.array[1,:,:].ravel(),vv_vh_hand.array[2,:,:].ravel())).T
validation_data = validationRaster.array.ravel()
goodIndices_data = np.all(vv_vh_hand_data!=vv_vh_hand.ndv,axis=1)
numberOfGoodDataPixels = goodIndices_data.sum()

# factor - level combinations
numberOfTrials = 4
sampleSizes = list(np.round(np.array([1])*numberOfGoodDataPixels))
tolerances = [1e-4] #[1e-2,1e-4,1e-6]
covarianceTypes = ['full','tied'] #['full','tied','diag','spherical']
regularizationCovariances = [1e-6]
trials = list(range(1,numberOfTrials+1))

# extract number of levels for each factor and group sample size
numOfLevelsOfSampleSizes = len(sampleSizes)
numOfLevelsOfTolerances = len(tolerances)
numOfLevelsOfCovarianceTypes = len(covarianceTypes)
numOfLevelsOfRegularizationCovariances = len(regularizationCovariances)
numOfTrialsPerFactorLevelCombination = numberOfTrials

# combinations and trials
combinations = list(product(range(numOfLevelsOfSampleSizes),range(numOfLevelsOfTolerances),
							range(numOfLevelsOfCovarianceTypes),range(numOfLevelsOfRegularizationCovariances)))
numberOfExperiments = len(combinations) * numOfTrialsPerFactorLevelCombination
numberOfFactorLevels = int(numberOfExperiments / numOfTrialsPerFactorLevelCombination)

table = pd.DataFrame(index = list(range(numberOfExperiments)),
					columns = ["trials","sampleSizes","tolerances","covarianceTypes",
					"regularizationCovariances","gmm","predictedLabels","time","accuracyCLUSTER","accuracyHAND","VRC","HAND"])

numberOfClusters = 2

"""
###################################################################################################################################
EM-GMM Prediction across sample sizes, tolerances, covarianceTypes, and regularizationCovariances
"""

"""
print("... EM-GMM Prediction ...")
c = -1 ; tr = -1
for index,row in table.iterrows():

	if (index % numOfTrialsPerFactorLevelCombination) == 0:
		c += 1
		tr = -1

	tr += 1

	ss = combinations[c][0]; to = combinations[c][1]; ct = combinations[c][2]; rc = combinations[c][3]
	sampleSize = sampleSizes[ss] ; tolerance = tolerances[to] ; covarianceType = covarianceTypes[ct] ; regularizationCovariance = regularizationCovariances[rc] ; trial = trials[tr]
	table["sampleSizes"][index] = sampleSize ; table["tolerances"][index] = tolerance ; table["covarianceTypes"][index] = covarianceType ; table["regularizationCovariances"][index] = regularizationCovariance ; table['trials'][index] = trial
	print("Experiment: {} / {} | F-L_Combo: {}/{} | SS_Prop: {} | Tol: {} | CovType: {} | RegCov: {} | Trial: {}".format(index+1,numberOfExperiments,c+1,numberOfFactorLevels,np.round(sampleSize/numberOfGoodDataPixels,2),tolerance,covarianceType,regularizationCovariance,trial))

	gmm = GaussianMixture(n_components=numberOfClusters, covariance_type=covarianceType, tol=tolerance, reg_covar=regularizationCovariance,
						max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None,
						precisions_init=None, random_state=None, warm_start=False, verbose=1, verbose_interval=1)

	## sample data
	indicesOfData = np.sort(np.random.choice(np.where(goodIndices_data)[0],size=sampleSize,replace=False))

	start = time.time()
	gmm.fit(vv_vh_hand_data[indicesOfData,0:2])
	predictedLabels = gmm.predict(vv_vh_hand_data[goodIndices_data,0:2]) + 1
	table['time'][index] = time.time() - start
	table['predictedLabels'][index] = predictedLabels
	table['accuracyCLUSTER'][index] = accuracy_score(validation_data[goodIndices_data],predictedLabels)

	table['gmm'][index] = gmm
	table['VRC'][index] = calinski_harabaz_score(vv_vh_hand_data[goodIndices_data,0:2], predictedLabels)

table.to_pickle(resultsTableFilename)
"""
"""
###################################################################################################################################
Write inundation and output rasters
"""
"""
table = pd.read_pickle(resultsTableFilename)
#plotSize = 5000

print("... Writing Predicted Inundation and Difference Rasters ...")
for index,row in table.iterrows():
	print("Experiment: {} / {} ".format(index+1,len(table)))
	predictedLabels = row['predictedLabels']

	if (row['gmm'].means_[0,:].mean() < row['gmm'].means_[1,:].mean()):
		inundated = [predictedLabels==2]
		noninundated = [predictedLabels==1]

		predictedLabels[noninundated] = 2
		predictedLabels[inundated] = 1

	predictedRaster = copy.deepcopy(validationRaster)
	goodIndices_array = np.where(predictedRaster.array != predictedRaster.ndv)
	predictedRaster.array[goodIndices_array[0],goodIndices_array[1]] = predictedLabels
	predictedRaster.ndv = -9999
	predictedInundation_filename = predictedInundation_filename_guide.format(row["trials"],row["sampleSizes"],
																			 row["tolerances"],row["covarianceTypes"],
																			 row["regularizationCovariances"],version)
	gis.writeGeotiff(predictedRaster,predictedInundation_filename,GDT_Byte)


	difference_filename = difference_filename_guide.format(row["trials"],row["sampleSizes"],
														   row["tolerances"],row["covarianceTypes"],
														   row["regularizationCovariances"],version)
	diffRaster = generateDifferenceRaster(predictedRaster,validationRaster)
	gis.writeGeotiff(diffRaster,difference_filename,GDT_Byte)


	# plotting
	xyzLim = [-25,0,-25,0,0,25]
	dataIndices = np.all(np.stack([np.array([vv_vh_hand.array[0,:,:] != vv_vh_hand.ndv]),
									np.array([vv_vh_hand.array[1,:,:] != vv_vh_hand.ndv]),
									np.array([vv_vh_hand.array[2,:,:] != vv_vh_hand.ndv]),
									np.array([vv_vh_hand.array[0,:,:] >= xyzLim[0]]), np.array([vv_vh_hand.array[0,:,:] <= xyzLim[1]]),
									np.array([vv_vh_hand.array[1,:,:] >= xyzLim[2]]), np.array([vv_vh_hand.array[1,:,:] <= xyzLim[3]]),
									np.array([vv_vh_hand.array[2,:,:] >= xyzLim[4]]), np.array([vv_vh_hand.array[2,:,:] <= xyzLim[5]])],axis=0),axis=0)
	dataIndices = dataIndices[0,:,:]
	dataIndices = np.where(dataIndices)

	indicesOfDataIndicesToSampleFrom = np.sort(np.random.choice(np.arange(len(dataIndices[0])),size=plotSize,replace=False))
	rowIndices = dataIndices[0][indicesOfDataIndicesToSampleFrom]
	columnIndices = dataIndices[1][indicesOfDataIndicesToSampleFrom]
	dataToPlot = vv_vh_hand.array[:,rowIndices,columnIndices].T
	vv,vh,hand = dataToPlot[:,0].ravel(),dataToPlot[:,1].ravel(),dataToPlot[:,2].ravel()

	diffLabel = diffRaster.array[rowIndices,columnIndices]


	fig1 = plt.figure()
	ax = fig1.add_subplot('111')

	line1 = ax.scatter(vv[diffLabel == 1], vh[diffLabel == 1],c='blue',alpha=0.3,marker='$TP$')
	line2 = ax.scatter(vv[diffLabel == 2], vh[diffLabel == 2],c='red',alpha=0.3,marker='$FP$')
	line3 = ax.scatter(vv[diffLabel == 3], vh[diffLabel == 3],c='black',alpha=0.3,marker='$FN$')
	line4 = ax.scatter(vv[diffLabel == 4], vh[diffLabel == 4],c='orange',alpha=0.3,marker='$TN$')

	#ax.set_title("{} {} Prediction".format(area,cl),pad=10)
	ax.set_xlabel("VV", rotation=0, size='large',labelpad=10)
	ax.set_ylabel("VH", rotation=90, size='large',labelpad=10)

	ax.set_xlim(xyzLim[0:2])
	ax.set_ylim(xyzLim[2:4])

	ax.xaxis.set_ticks(np.arange(xyzLim[0], xyzLim[1], 5))
	ax.yaxis.set_ticks(np.arange(xyzLim[2], xyzLim[3], 5))
	ax.set_title("Trial: {} | N: {} | Tol: {} | C: {}".format(row["trials"],row["sampleSizes"],
																		row["tolerances"],row["covarianceTypes"]),pad=10)

	fig1.legend((line1,line2,line3,line4),("TP","FP","FN","TN"),'upper right')
	fig1.savefig("diffScatterPlot_{}_{}_{}_{}_{}.tiff".format(row["trials"],row["sampleSizes"]/numberOfExperiments,
																		row["tolerances"],row["covarianceTypes"],
																		row["regularizationCovariances"]),
				format='tiff',dpi=300)

"""
"""
###################################################################################################################################
Runtime Plots
"""
"""
table = pd.read_pickle('resultsTable.pkl')
table[["time","accuracyCLUSTER","accuracyHAND","VRC"]] = table[["time","accuracyCLUSTER","accuracyHAND","VRC"]].apply(pd.to_numeric)

tol = table[["sampleSizes","tolerances","time","covarianceTypes"]].groupby(["sampleSizes",'tolerances','covarianceTypes']).mean()

lineStyles = {0.01 : '-',0.0001 : '--',1e-6 : ':'}
colorStyles = {'full': 'm','tied': 'y','spherical': 'r','diag': 'b'}

for i in product(tolerances,covarianceTypes):
	lineStyle = lineStyles[i[0]]
	colorStyle = colorStyles[i[1]]


	plt.plot(table[["sampleSizes","tolerances","time","covarianceTypes"]][table['tolerances']== i[0]][:][table['covarianceTypes']==i[1]].groupby(['sampleSizes']).mean()['time'],
		lineStyle+colorStyle,label="{} - {}".format(i[0],i[1]))

plt.title('Runtime By Tolerance and Covariance Type')
plt.xlabel('Sample Size')
plt.ylabel('Time (s)')
plt.legend()
plt.show()

"""
"""
###################################################################################################################################
Generate Catchmask Table
"""
"""
print(" ... Generating Catchmask Table ...")
table = pd.read_pickle(resultsTableFilename)

catchmask = gis.raster(catchmask_filename)
catchmaskList = np.sort(np.unique(catchmask.array))
catchmaskList = catchmaskList[catchmaskList!=catchmask.ndv]


numberOfCatchmasks = len(catchmaskList)
progressPoints = np.arange(0,101,5)/100
progressPoints = np.round(progressPoints * numberOfCatchmasks,0)


validationCatchMasksTable = pd.DataFrame(index = catchmaskList,columns = ['actualMaxHAND','actualHANDsampleSize',
																		  'indicesOfCatchmasks','predictedMaxHAND','allHANDValues',
																		  'HANDsampleSize'])

i = 0
for cm,row in validationCatchMasksTable.iterrows():

	if i in progressPoints:
		print("{} %".format(np.round((i/numberOfCatchmasks)*100,0)))

	indicesOfCatchmasks = np.where(catchmask.array==cm)
	handValues = vv_vh_hand.array[2,indicesOfCatchmasks[0],indicesOfCatchmasks[1]]
	validationInundatedHandValues = handValues[validationRaster.array[indicesOfCatchmasks[0],indicesOfCatchmasks[1]]==2]
	if validationInundatedHandValues.size == 0:
		actualMaxHAND = 0
	else:
		actualMaxHAND = np.amax(validationInundatedHandValues)

	validationCatchMasksTable['indicesOfCatchmasks'][cm] = indicesOfCatchmasks
	validationCatchMasksTable['actualHANDsampleSize'][cm] = validationInundatedHandValues.size
	validationCatchMasksTable['actualMaxHAND'][cm] = actualMaxHAND
	validationCatchMasksTable['allHANDValues'][cm] = handValues
	validationCatchMasksTable['predictedMaxHAND'][cm] = np.max(handValues)
	validationCatchMasksTable['HANDsampleSize'][cm] = handValues.size
	i+=1

validationCatchMasksTable.to_pickle(validationCatchMasksTable_filename)
"""

"""
###################################################################################################################################
Generate predicted HAND values per catchment and per experiment
"""
"""
print("... Generate predicted HAND values per catchment and by experiment ...")

table = pd.read_pickle(resultsTableFilename)
validationCatchMasksTable = pd.read_pickle(validationCatchMasksTable_filename)

numberOfCatchmasks = len(validationCatchMasksTable)
progressPoints = np.arange(0,101,5)/100
progressPoints = np.round(progressPoints * numberOfCatchmasks,0)

table['predictedCatchMasksTables'] = ""

for index,row in table.iterrows():
	print("Experiment: {} / {} ".format(index+1,len(table)))

	predictedInundation_filename = predictedInundation_filename_guide.format(row["trials"],row["sampleSizes"],
																			 row["tolerances"],row["covarianceTypes"],
																			 row["regularizationCovariances"],version)
	predictedRaster = gis.raster(predictedInundation_filename)

	table['predictedCatchMasksTables'][index] = pd.DataFrame(index = validationCatchMasksTable.index.values,columns = ['predictedMaxHAND','predictedHANDsampleSize'])

	i=0
	for cm,row2 in validationCatchMasksTable.iterrows():

		if i in progressPoints:
			print("{}%".format(np.round((i/numberOfCatchmasks)*100)))

		handValues = vv_vh_hand.array[2,row2['indicesOfCatchmasks'][0],row2['indicesOfCatchmasks'][1]]
		predictedInundatedHandValues = handValues[predictedRaster.array[row2['indicesOfCatchmasks'][0],row2['indicesOfCatchmasks'][1]]==2]
		if predictedInundatedHandValues.size == 0:
			predictedMaxHand = 0
		else:
			predictedMaxHand = np.amax(predictedInundatedHandValues)

		table['predictedCatchMasksTables'][index]['predictedHANDsampleSize'][cm] = predictedInundatedHandValues.size
		table['predictedCatchMasksTables'][index]['predictedMaxHAND'][cm] = predictedMaxHand
		i+=1

table.to_pickle(resultsTableFilename)
"""

"""
###################################################################################################################################
Generate predicted HAND values per catchment and per experiment
"""

"""
print("... Generate HAND inundation raster and difference rasters ...")

table = pd.read_pickle(resultsTableFilename)
validationCatchMasksTable = pd.read_pickle(validationCatchMasksTable_filename)

numberOfCatchmasks = len(validationCatchMasksTable)
progressPoints = np.arange(0,101,5)/100
progressPoints = np.round(progressPoints * numberOfCatchmasks,0)

for index,row in table.iterrows():
	print("Experiment: {} / {} ".format(index+1,len(table)))

	predictedInundation_filename = predictedInundation_filename_guide.format(row["trials"],row["sampleSizes"],
																			 row["tolerances"],row["covarianceTypes"],
																			 row["regularizationCovariances"],version)
	predictedRaster = gis.raster(predictedInundation_filename)

	i=0 ; meanPredictedInundatedHand = []; meanValidationInundatedHand = []
	for cm,row2 in validationCatchMasksTable.iterrows():

		if i in progressPoints:
			print("{}%".format(np.round((i/numberOfCatchmasks)*100)))

		handValues = vv_vh_hand.array[2,row2['indicesOfCatchmasks'][0],row2['indicesOfCatchmasks'][1]]
		predictedMaxHand = table['predictedCatchMasksTables'][index]['predictedMaxHAND'][cm]
		actualMaxHAND = validationCatchMasksTable['actualMaxHAND'][cm]

		newPredictedRaster = copy.deepcopy(predictedRaster)

		inundatedValues = np.zeros((handValues.shape),dtype=np.ubyte)
		inundatedValues[handValues <= predictedMaxHand] = 2
		inundatedValues[handValues > predictedMaxHand] = 1

		newPredictedRaster.array[row2['indicesOfCatchmasks'][0],row2['indicesOfCatchmasks'][1]] = inundatedValues


		i+=1
	predictedRaster_filename = predictedInundation_HAND_filename_guide.format(row["trials"],row["sampleSizes"],
																			  row["tolerances"],row["covarianceTypes"],
																			  row["regularizationCovariances"],version)
	gis.writeGeotiff(newPredictedRaster,predictedRaster_filename,GDT_Byte)

	difference_filename = difference_filename_guide.format(row["trials"],row["sampleSizes"],
														   row["tolerances"],row["covarianceTypes"],
														   row["regularizationCovariances"],version)
	diff_predictedRaster = gis.raster(difference_filename_guide.format(row["trials"],row["sampleSizes"],
																	   row["tolerances"],row["covarianceTypes"],
																	   row["regularizationCovariances"],version))

	diff_predictedRaster_HAND = generateDifferenceRaster(newPredictedRaster,validationRaster)

	difference_HAND_filename = difference_HAND_filename_guide.format(row["trials"],row["sampleSizes"],
																	 row["tolerances"],row["covarianceTypes"],
																	 row["regularizationCovariances"],version)
	gis.writeGeotiff(diff_predictedRaster_HAND,difference_HAND_filename,GDT_Byte)

	accuracy_predictedRaster = (diff_predictedRaster.array[diff_predictedRaster.array==4].size + diff_predictedRaster.array[diff_predictedRaster.array ==1].size)/(diff_predictedRaster.array[diff_predictedRaster.array!=0].size)
	accuracy_predictedRaster_HAND = (diff_predictedRaster_HAND.array[diff_predictedRaster_HAND.array==4].size + diff_predictedRaster_HAND.array[diff_predictedRaster_HAND.array ==1].size)/(diff_predictedRaster_HAND.array[diff_predictedRaster_HAND.array!=0].size)

	table['accuracyCLUSTER'][index] = accuracy_predictedRaster
	table['accuracyHAND'][index] = accuracy_predictedRaster_HAND

table.to_pickle(resultsTableFilename)
"""

"""
###################################################################################################################################
Hydrosequencing
"""
"""

# makes one panda dataframe from flowline and flowlineVAA DBF files
if not os.path.isfile(merged_dataframe_filename):

	print("... Performing Hydrosequencing ...")
	# import DBF files
	flowlinesvaa = DBF(flowlinesvaa_filename)
	flowlines = DBF(flowlines_filename_guide.format("dbf"))

	# convert to panda dataframes
	flowlinesvaa_dataframe = pd.DataFrame(iter(flowlinesvaa))
	flowlines_dataframe = pd.DataFrame(iter(flowlines))

	flowlines_dataframe = flowlines_dataframe.rename(columns={'COMID':'ComID'})

	merged_dataframe = pd.merge(flowlinesvaa_dataframe,flowlines_dataframe,on='ComID',how='right')
	merged_dataframe['StreamOrde'] = merged_dataframe['StreamOrde'].fillna(-1)
	merged_dataframe = merged_dataframe.astype({"ComID": int,"FromNode" : pd.Int64Dtype(),
												"ToNode" : pd.Int64Dtype(),"Hydroseq" : pd.Int64Dtype(),
												"LevelPathI" : pd.Int64Dtype(),"DnHydroseq" : pd.Int64Dtype(),
												"TerminalPa" : pd.Int64Dtype(),"DnLevelPat" : pd.Int64Dtype(),
												"UpLevelPat" : pd.Int64Dtype(), "UpHydroseq" : pd.Int64Dtype(),
												"DnMinorHyd" : pd.Int64Dtype(),"DnDrainCou" : pd.UInt16Dtype(),
												"TerminalFl" : pd.UInt16Dtype(),'StreamOrde': np.int16})
	merged_dataframe['predictedHeight'] = -1
	merged_dataframe['validationHeight'] = -1
	merged_dataframe['maxHAND'] = -1
	merged_dataframe['visited'] = False
	merged_dataframe['order'] = int

	merged_dataframe = merged_dataframe.set_index('ComID',drop = False)

	merged_dataframe.to_pickle(merged_dataframe_filename)
else:
	merged_dataframe = pd.read_pickle(merged_dataframe_filename)
"""

"""
# Check for intersection/union/etc of flowlines and flowlinesvaa

flowlines_list = list()
flowlinesvaa_list = list()

for flowline in flowlines:
	flowlines_list.append(flowline['COMID'])

for flowline in flowlinesvaa:
	flowlinesvaa_list.append(int(flowline['ComID']))


flowlines_set = set(flowlines_list)
flowlinesvaa_set = set(flowlinesvaa_list)

# intersection
intersection = flowlines_set & flowlinesvaa_set

# union
union = flowlines_set | flowlinesvaa_set

# in flowlines but not vaa
flowlinesONLY = flowlines_set - flowlinesvaa_set

# in vaa but not flowlines
vaaONLY = flowlinesvaa_set - flowlines_set
"""

def traverseNetwork(current_comid,m_dataframe,table,validationCatchMasksTable,order=0,verbose=True):

	order += 1

	# check if existing
	if current_comid not in set(m_dataframe.index):
		return(None,m_dataframe)

	# check if visited
	if m_dataframe.loc[current_comid,"visited"]:
		return(None,m_dataframe)

	# get to node
	toNode = m_dataframe.loc[current_comid,"ToNode"]
	fromNode = m_dataframe.loc[current_comid,"FromNode"]

	# get matching from node comids
	next_comid_list = list(m_dataframe.loc[:,"ComID"][m_dataframe.loc[:,"FromNode"] == toNode])

	# record height
	try:
		current_comid_predicted_height = table['predictedCatchMasksTables'][0]['predictedMaxHAND'][current_comid]
	except KeyError:
		current_comid_predicted_height = 0

	try:
		current_comid_validation_height = validationCatchMasksTable['actualMaxHAND'][current_comid]
		current_catchment_max_hand_height = validationCatchMasksTable['predictedMaxHAND'][current_comid]
	except KeyError:
		current_comid_validation_height = 0
		current_catchment_max_hand_height = 0

	if verbose:
		print("From {} with predicted={:.2f}, validation={:.2f}, max={:.2f} to {}, order {}".format(current_comid,current_comid_predicted_height,current_comid_validation_height,current_catchment_max_hand_height,next_comid_list,order))

	"""
	# zero filling ###
	if current_comid_predicted_height <= 1e-2:
		# get prev_comid
		print("CUrrent COMID 0")
		prev_comid_list = list(m_dataframe.loc[:,"ComID"][m_dataframe.loc[:,"ToNode"] == fromNode])

		# get prev_height
		try:
			prev_comid_predicted_height = np.array([])
			for prev_comid in prev_comid_list:
				prev_comid_predicted_height = np.append(prev_comid_predicted_height,table['predictedCatchMasksTables'][0]['predictedMaxHAND'][prev_comid])
		except KeyError("Previous COMID doesn't exist in catchmask table"):
			pass
			#current_comid_predicted_height = 0

		# get next_comid_height
		try:
			next_comid_predicted_height = np.array([])
			for next_comid in next_comid_list:
				next_comid_predicted_height = np.append(next_comid_predicted_height,table['predictedCatchMasksTables'][0]['predictedMaxHAND'][next_comid])
		except KeyError("Previous COMID doesn't exist in catchmask table"):
			print("Double zeros")
			exit()

		# average
		current_comid_predicted_height = np.average(np.append(next_comid_predicted_height,prev_comid_predicted_height))
		print(current_comid_predicted_height)
	"""

	# mark as visited and height
	m_dataframe.loc[current_comid,"visited"] = True
	m_dataframe.loc[current_comid,"predictedHeight"] = current_comid_predicted_height
	m_dataframe.loc[current_comid,"validationHeight"] = current_comid_validation_height
	m_dataframe.loc[current_comid,"maxHAND"] = current_catchment_max_hand_height
	m_dataframe.loc[current_comid,"order"] = order


	# return None if no comids are found
	if len(next_comid_list) == 0:
		return(None,m_dataframe)
	# operate recursively over downstream segements
	for next_comid in next_comid_list:
		next_comid_list,m_dataframe = traverseNetwork(next_comid,m_dataframe,table,validationCatchMasksTable,order,verbose)
		break

	return(next_comid_list,m_dataframe)
"""
table = pd.read_pickle(resultsTableFilename)
validationCatchMasksTable = pd.read_pickle(validationCatchMasksTable_filename)

# find headwaters
toNodes = set(merged_dataframe.loc[:,'ToNode'])
fromNodes = set(merged_dataframe.loc[:,'FromNode'])
headwater_nodes = fromNodes - toNodes
indices_headwater_nodes = np.nonzero(np.in1d(merged_dataframe.loc[:,'FromNode'],list(headwater_nodes)))[0]
headwater_streams = list(merged_dataframe.iloc[indices_headwater_nodes,:]['ComID'])

# find unconnected streams
unconnectedStreams = list(merged_dataframe.loc[:,"ComID"][merged_dataframe.loc[:,"FLOWDIR"] == "Uninitialized"])

# combine
streams = headwater_streams + unconnectedStreams

for stream in streams:
	_,merged_dataframe = traverseNetwork(stream,merged_dataframe,table,validationCatchMasksTable)

#for comid in merged_dataframe.index.values:
#	_,merged_dataframe = traverseNetwork(comid,merged_dataframe,table,validationCatchMasksTable)

merged_dataframe.to_pickle(merged_inundated_dataframe_filename)

# merge and write
flowlines_dataframe = gpd.read_file(flowlines_filename_guide.format('shp'))
flowlines_merged_dataframe = pd.merge(flowlines_dataframe,merged_dataframe[['predictedHeight','validationHeight','maxHAND','StreamOrde']],left_on='COMID',right_index=True)
flowlines_merged_dataframe.to_file(flowlines_heights_filename_guide.format('shp'))
"""

"""
###################################################################################################################################
Plot Hydrosequencing

"""
"""
flowlines_heights_dataframe = gpd.read_file(flowlines_heights_filename_guide.format('shp'))


flowlines_heights_dataframe['err'] = np.array(flowlines_heights_dataframe['predictedH']) - np.array(flowlines_heights_dataframe['validation'])
flowlines_heights_dataframe['mae'] = np.absolute(np.array(flowlines_heights_dataframe['validation']) - np.array(flowlines_heights_dataframe['predictedH']))

uniqueStreamOrders = np.unique(flowlines_heights_dataframe['StreamOrde'])
uniqueStreamOrders = uniqueStreamOrders[uniqueStreamOrders != -1]
colors = ['b', 'g', 'r', 'c', 'm']
"""

"""
fig1 = plt.figure()
ax = fig1.add_subplot('111')


points = []
for i,uso in enumerate(uniqueStreamOrders):
	points.append(ax.scatter(flowlines_heights_dataframe['validation'][flowlines_heights_dataframe['StreamOrde'] == uso], flowlines_heights_dataframe['predictedH'][flowlines_heights_dataframe['StreamOrde'] == uso],c=colors[i],alpha=0.3))

ax.set_title("Validation vs Prediction",pad=10)
ax.set_xlabel("Validation", rotation=0, size='large',labelpad=10)
ax.set_ylabel("Prediction", rotation=90, size='large',labelpad=10)

#graphMin,graphMax = np.amin(ax.get_xlim()+ax.get_ylim())) , np.amax(ax.get_xlim()+ax.get_ylim())
ranges = ((np.amin(ax.get_xlim()+ax.get_ylim())) , np.amax(ax.get_xlim()+ax.get_ylim()))
#ranges = [np.amin(flowlines_heights_dataframe[['validation','predictedH']]),np.amax(flowlines_heights_dataframe[['validation','predictedH']])]

#ranges)
ax.set_xlim(ranges)
ax.set_ylim(ranges)

ax.xaxis.set_ticks(np.linspace(ranges[0], ranges[1], 5))
ax.yaxis.set_ticks(np.linspace(ranges[0], ranges[1], 5))

# add line
ax.plot(ranges, ranges)

fig1.legend(points,uniqueStreamOrders,'upper right',title="Stream Order",ncol=2)
fig1.savefig('figures/validation_v_prediction',format='jpeg',dpi=300)


"""
"""
fig2 = plt.figure()
ax = fig2.add_subplot('111')
bar_width = 0.25
index = np.arange(len(uniqueStreamOrders))

meanValuesByStreamOrder = flowlines_heights_dataframe.groupby('StreamOrde').mean()
meanValuesByStreamOrder = meanValuesByStreamOrder.loc[uniqueStreamOrders,:]

stdValuesByStreamOrder = flowlines_heights_dataframe.groupby('StreamOrde').std()
stdValuesByStreamOrder = stdValuesByStreamOrder.loc[uniqueStreamOrders,:]

plt.bar(uniqueStreamOrders, meanValuesByStreamOrder['err'], bar_width,alpha=0.5,color='r',label=uniqueStreamOrders,zorder=5)
plt.errorbar(uniqueStreamOrders,meanValuesByStreamOrder['err'],yerr=2*stdValuesByStreamOrder['err'],fmt='none',capsize=4)

ax.set_title("Mean Error for Max Inundated HAND By Stream Order \n (Error Bars 2x Std)",pad=10)
ax.set_xlabel("Stream Order", rotation=0, size='large',labelpad=10)
ax.set_ylabel("Mean Error for Max Inundated HAND (m)", rotation=90, size='large',labelpad=10)
ax.yaxis.grid()

fig2.savefig('figures/meanErrorByStreamOrder',format='jpeg',dpi=300)

quantiles = ['1','2','3','4','5']
bar_width = 0.25

flowlines_heights_dataframe['quantilesOfMaxHAND'] = pd.qcut(flowlines_heights_dataframe['maxHAND'],[0,.20,0.4,0.6,0.8,1],labels=quantiles)

meanValuesByMaxHAND = flowlines_heights_dataframe.groupby('quantilesOfMaxHAND').mean()
stdValuesByMaxHAND = flowlines_heights_dataframe.groupby('quantilesOfMaxHAND').std()

fig3 = plt.figure()
ax = fig3.add_subplot('111')
plt.bar(quantiles, meanValuesByMaxHAND['err'], bar_width,alpha=0.5,color='r',label=quantiles,zorder=5)
plt.errorbar(quantiles,meanValuesByMaxHAND['err'],yerr=2*stdValuesByMaxHAND['err'],fmt='none',capsize=4)

ax.set_title("Mean Error for Max Inundated HAND By Max HAND Quantile\n (Error Bars 2x Std)",pad=10)
ax.set_xlabel("MaxHAND of Catchment Quantiles", rotation=0, size='large',labelpad=10)
ax.set_ylabel("Mean Error for Max Inundated HAND (m)", rotation=90, size='large',labelpad=10)
ax.yaxis.grid()

fig3.savefig('figures/meanErrorByMaxHANDQuantile',format='jpeg',dpi=300)

plt.show()
"""
"""
streams = [8790261,11235283,11239861]

table = pd.read_pickle(resultsTableFilename)
validationCatchMasksTable = pd.read_pickle(validationCatchMasksTable_filename)

for stream in streams:
	_,merged_inundated_dataframe = traverseNetwork(stream,merged_dataframe,table,validationCatchMasksTable,verbose=False)

	dataToPLOT = merged_inundated_dataframe.loc[merged_inundated_dataframe['visited']==True,['predictedHeight','validationHeight',"order"]].sort_values('order')

	fig4 = plt.figure()
	ax = fig4.add_subplot('111')
	pred = plt.plot(dataToPLOT['order'],dataToPLOT['predictedHeight'],alpha=0.5,color='b',label='Predicted')
	val = plt.plot(dataToPLOT['order'],dataToPLOT['validationHeight'],alpha=0.5,color='r',label='Validation')


	ax.set_title("Max HAND Inundated Starting At {}".format(stream),pad=10)
	ax.set_xlabel("COMID", rotation=0, size='large',labelpad=10)
	ax.set_ylabel("MaxHAND Inundated (m)", rotation=90, size='large',labelpad=10)
	ax.yaxis.grid()

	#fig4.legend((pred,val),("Predicted","Validation"),'upper right',title="Stream Order",ncol=1)
	fig4.legend([pred,val],loc='upper right',labels=['Predicted','Validation'],ncol=1)

	fig4.savefig('figures/flowlinePlot_sourceCOMID_{}.jpeg'.format(stream),format='jpeg',dpi=300)

plt.show()
"""

"""
###################################################################################################################################
MaxHAND From Validation
"""
"""
# input: HAND raster, validation max inundated HAND value by catchment, catchment raster, validation raster

# inundate raster by catchment
#vv_vh_hand
validationFromMaxHANDvalues = copy.deepcopy(validationRaster)
catchmask = gis.raster(catchmask_filename)
table = pd.read_pickle(resultsTableFilename)
validationCatchMasksTable = pd.read_pickle(validationCatchMasksTable_filename)

catchmasks = validationCatchMasksTable.index.values
validationFromMaxHANDvalues.array = np.zeros((validationFromMaxHANDvalues.nrows,validationFromMaxHANDvalues.ncols))

for cm in catchmasks:
	#predictedMAXHAND = table['predictedCatchMasksTables'][0]['predictedMaxHAND'][cm]
	validationHAND = validationCatchMasksTable['actualMaxHAND'][cm]
	indicesOfCM = validationCatchMasksTable['indicesOfCatchmasks'][cm]
	handValues = vv_vh_hand.array[2,indicesOfCM[0],indicesOfCM[1]]
	inundatedValues = np.zeros((handValues.shape),dtype=np.ubyte)
	inundatedValues[handValues <= validationHAND] = 2
	inundatedValues[handValues > validationHAND] = 1
	validationFromMaxHANDvalues.array[indicesOfCM] = inundatedValues

validationFromMaxHANDvalues.array = validationFromMaxHANDvalues.array.astype(np.ubyte)

gis.writeGeotiff(validationFromMaxHANDvalues,validationFromMaxHANDvalues_filename,GDT_Byte)

diffRaster = generateDifferenceRaster(validationFromMaxHANDvalues,validationRaster,writeTIFF=True,output_fileName=validationFromMaxHANDvalues_difference_filename)

binaryClassificationStats = calculateBinaryClassificationStatistics(diffRaster,mapping={'TP':1,'FP':2,'FN':3,'TN':4,'ndv':0})
print(binaryClassificationStats)
"""

"""
###################################################################################################################################
Filter Out Zeros (Outliers)
"""
"""
table = pd.read_pickle(resultsTableFilename)
validationCatchMasksTable = pd.read_pickle(validationCatchMasksTable_filename)

# find headwaters
toNodes = set(merged_dataframe.loc[:,'ToNode'])
fromNodes = set(merged_dataframe.loc[:,'FromNode'])
headwater_nodes = fromNodes - toNodes
indices_headwater_nodes = np.nonzero(np.in1d(merged_dataframe.loc[:,'FromNode'],list(headwater_nodes)))[0]
headwater_streams = list(merged_dataframe.iloc[indices_headwater_nodes,:]['ComID'])

# find unconnected streams
unconnectedStreams = list(merged_dataframe.loc[:,"ComID"][merged_dataframe.loc[:,"FLOWDIR"] == "Uninitialized"])

# combine
streams = headwater_streams + unconnectedStreams

for stream in streams:
	_,merged_dataframe = traverseNetwork(stream,merged_dataframe,table,validationCatchMasksTable)

print(merged_dataframe)

"""

"""
###################################################################################################################################
Filtering
"""

print('Graph filtering ...')

plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (10, 5)

"""
rs = np.random.RandomState(42)  # Reproducible results.
W = rs.uniform(size=(30, 30))  # Full graph
W[W < 0.93] = 0  # Sparse graph.
W = W + W.T  # Symmetric graph.
np.fill_diagonal(W, 0)  # No self-loops.
G = graphs.Graph(W)
print(G.__dict__)
print(G.W)
print(W)
print('{} nodes, {} edges'.format(G.N, G.Ne))
"""


flowlines_heights_dataframe = gpd.read_file(flowlines_heights_filename_guide.format('shp'))
merged_inundated_dataframe = pd.read_pickle(merged_inundated_dataframe_filename)
merged_dataframe = pd.read_pickle(merged_dataframe_filename)
predictedRaster = gis.raster('data/results/predictedInundation_1_747401_0.0001_full_1e-06_v0.tiff')

diffRaster = generateDifferenceRaster(predictedRaster,validationRaster,writeTIFF=False)
binaryClassificationStats = calculateBinaryClassificationStatistics(diffRaster,mapping={'TP':1,'FP':2,'FN':3,'TN':4,'ndv':0})
print('EMM-GMM Prediction Performance ...')
print("F-Score: {:0.3f} | MCC: {:0.3f} | BACC: {:0.3f} | ACC: {:0.3f} | TPR: {:0.3f} | TNR: {:0.3f} | PPV: {:0.3f} | NPV: {:0.3f} | TP: {:0.3f} | FP: {:0.3f} | TN: {:0.3f} | FN: {:0.3f}".format(binaryClassificationStats['F1_score'],
																										binaryClassificationStats['MCC'],
																										binaryClassificationStats['BACC'],
																										binaryClassificationStats['ACC'],
																										binaryClassificationStats['TPR'],
																										binaryClassificationStats['TNR'],
																										binaryClassificationStats['PPV'],
																										binaryClassificationStats['NPV'],
																										binaryClassificationStats['TP_area']/((10**3)**2),
																										binaryClassificationStats['FP_area']/((10**3)**2),
																										binaryClassificationStats['TN_area']/((10**3)**2),
																										binaryClassificationStats['FN_area']/((10**3)**2)))


allCOMIDS = merged_inundated_dataframe.index.values
numberOfCOMIDS = len(merged_inundated_dataframe)

adjacencyMatrix = np.zeros((numberOfCOMIDS,numberOfCOMIDS),dtype=np.int8)
coordinates = np.zeros((numberOfCOMIDS,2),dtype=np.float32)

### extract drainage areas to use as weights
drainageAreas = {}
allCOMIDS_set = set([int(i) for i in allCOMIDS])
for record in DBF(drainageAreas_fileName):
	comid = int(record['ComID'])
	if comid in allCOMIDS_set:
		drainageAreas[comid] = float(record['TotDASqKM'])

# print(merged_inundated_dataframe['TotDASqKM']);exit()
# print(flowlines_heights_dataframe.columns);exit()

for rowIndex,current_comid in enumerate(allCOMIDS):
	# get to node
	toNode = merged_inundated_dataframe.loc[current_comid,"ToNode"]
	fromNode = merged_inundated_dataframe.loc[current_comid,"FromNode"]

	# coordinates[rowIndex,:] = flowlines_heights_dataframe['geometry'][rowIndex].centroid.x,flowlines_heights_dataframe['geometry'][rowIndex].centroid.y
	toMatches = (merged_inundated_dataframe.loc[:,"FromNode"] == toNode).astype(np.int8)
	# fromMatches = (merged_inundated_dataframe.loc[:,"ToNode"] == fromNode).astype(np.int8)

	booleanOf_toMatches = toMatches==1

	if booleanOf_toMatches.sum() > 0:
		toMatches_DA = np.array([],dtype=np.float32)
		for tm in toMatches[booleanOf_toMatches].index.values:
			print(current_comid,tm)
			toMatches_DA = np.append(toMatches_DA,[1/abs(drainageAreas[current_comid]-drainageAreas[tm])])

		# toMatches_normalized_DA = toMatches_DA / np.max(toMatches_DA)

		# adjacencyMatrix[rowIndex,booleanOf_toMatches] = toMatches_normalized_DA
		# adjacencyMatrix[rowIndex,:] = toMatches
		adjacencyMatrix[rowIndex,booleanOf_toMatches] = toMatches_DA


	# adjacencyMatrix[rowIndex,:] = (merged_inundated_dataframe.loc[:,"FromNode"] == toNode).astype(np.int8)

np.set_printoptions(threshold=np.inf)
print(adjacencyMatrix)

# make undirected
adjacencyMatrix = np.maximum(adjacencyMatrix,adjacencyMatrix.T)
# adjacencyMatrix = adjacencyMatrix/np.max(adjacencyMatrix)

indicesOfWeightsAsTupleOfArrays = np.where(adjacencyMatrix)
indicesOfWeights = list(zip(*indicesOfWeightsAsTupleOfArrays))

# print(indicesOfWeights)

# for idx in indicesOfWeights:
	# print(allCOMIDS[idx[0]]," ",allCOMIDS[idx[1]])
	# print(merged_inundated_dataframe.loc[allCOMIDS[idx[0]],'ToNode'],' ',merged_inundated_dataframe.loc[allCOMIDS[idx[1]],'FromNode'])

GRAPH = graphs.Graph(adjacencyMatrix,coords=None)
SIGNAL = merged_inundated_dataframe['predictedHeight'].values
validationValues = merged_inundated_dataframe['validationHeight'].values

GRAPH.estimate_lmax()

mae = lambda x,y: np.mean(np.absolute(x-y))
bias = lambda x,y: np.mean(x)-np.mean(y)
snr = lambda x: np.absolute(np.mean(x)/np.std(x))

print('Unfiltered Signal Metrics')
print("  MAE:  {:0.3f} | Bias: {:0.3f} | SNR: {:0.3f}".format(mae(SIGNAL,validationValues),bias(SIGNAL,validationValues),snr(SIGNAL)))

filters_list = [(filters.Abspline,{'Nf': 1})] #[(filters.Abspline,{'Nf': 2}),(filters.HalfCosine,{}),(filters.Heat,{'tau':1})]
filters_list_names = ['abspline']

for i,currentFilter in enumerate(filters_list):

	print(currentFilter[0],currentFilter[1])
	my_filter = currentFilter[0](GRAPH,**currentFilter[1])

	FILTERED_SIGNAL = my_filter.filter(SIGNAL)

	# sum across eigenvalues
	if len(FILTERED_SIGNAL.shape) == 2:
		# FILTERED_SIGNAL = FILTERED_SIGNAL.sum(axis=1)
		FILTERED_SIGNAL = FILTERED_SIGNAL[:,:].sum(axis=1)

	print("  MAE:  {:0.3f} | Bias: {:0.3f} | SNR: {:0.3f}".format(mae(FILTERED_SIGNAL,validationValues),bias(FILTERED_SIGNAL,validationValues),snr(FILTERED_SIGNAL)))
	merged_inundated_dataframe['filteredPredictedHeights'] = FILTERED_SIGNAL
	flowlines_heights_dataframe[filters_list_names[i]] = FILTERED_SIGNAL


	# GRAPH.plot_signal(SIGNAL)
	# GRAPH.plot_signal(FILTERED_SIGNAL[:,0])
	# plotting.show()

	# print(GRAPH.__dict__)
	# plotting.plot_graph(GRAPH)
	# plotting.show()

	streams = [8790261,11235283,11239861]
	plotFiltered = True

	table = pd.read_pickle(resultsTableFilename)
	validationCatchMasksTable = pd.read_pickle(validationCatchMasksTable_filename)

	for stream in streams:

		merged_inundated_dataframe['visited'] = np.repeat(False,len(merged_inundated_dataframe))
		merged_inundated_dataframe['order'] = np.repeat(int(),len(merged_inundated_dataframe))

		_,merged_inundated_dataframe = traverseNetwork(stream,merged_dataframe,table,validationCatchMasksTable,verbose=False)

		try:
			merged_inundated_dataframe['filteredPredictedHeights'] = FILTERED_SIGNAL[:,int(rowToUse)]
		except NameError:
			merged_inundated_dataframe['filteredPredictedHeights'] = FILTERED_SIGNAL

		#with pd.option_context('display.max_rows', None):  # more options can be specified also
		#	print(flowlines_heights_dataframe)

		#print(flowlines_heights_dataframe)
		#### ADD VISITED COLUMN TO flowlines_heights_dataframe AND WRITE OUT TO DIFFERENT FILE NAMES #############
		tempDataframe = merged_inundated_dataframe[['visited','ComID']]
		tempDataframe = tempDataframe.rename(columns={'ComID':'COMID','visited':str(stream)})
		#tempDataframe = tempDataframe.insert(0,'COMID',tempDataframe.index.values,True)
		#print(tempDataframe)
		flowlines_heights_dataframe = flowlines_heights_dataframe.merge(tempDataframe,on='COMID')

		dataToPLOT = merged_inundated_dataframe.loc[merged_inundated_dataframe['visited']==True,['predictedHeight','validationHeight','filteredPredictedHeights',"order","ComID"]].sort_values('order')
		xLabels = [str(i) for i in dataToPLOT['ComID']]

		fig4 = plt.figure()
		ax = fig4.add_subplot('111')
		pred = plt.plot(xLabels,dataToPLOT['predictedHeight'],alpha=0.5,color='b',label='SAR',linewidth=3)
		val = plt.plot(xLabels,dataToPLOT['validationHeight'],alpha=0.5,color='r',label='Validation',linewidth=3)
		if plotFiltered:
			fil = plt.plot(xLabels,dataToPLOT['filteredPredictedHeights'],alpha=0.5,color='g',label='Filtered',linewidth=3)

		ax.set_title("Stages For Flow Path Starting At COMID {}".format(stream),pad=10)
		ax.set_xlabel("COMID", rotation=0, size='large',labelpad=10)
		ax.set_ylabel("Stage (m)", rotation=90, size='large',labelpad=10)

		fig4.tight_layout()
		fig4.subplots_adjust(bottom=0.25)
		plt.xticks(rotation=90)
		ax.yaxis.grid()
		ax.xaxis.grid()

		# fig4.legend((pred,val,fil),loc='upper right',ncol=1,labels=('Predicted','Validation','Filtered'))
		if plotFiltered:
			fig4.legend((pred,val,fil),loc='upper right',ncol=1,labels=('SAR','Validation','Filtered'))
		else:
			fig4.legend((pred,val),loc='upper right',ncol=1,labels=('SAR','Validation'))

		if plotFiltered:
			fig4.savefig('figures/flowlinePlot_sourceCOMID_{}_filtered.jpeg'.format(stream),format='jpeg',dpi=300)
		else:
			fig4.savefig('figures/flowlinePlot_sourceCOMID_{}_noFiltered.jpeg'.format(stream),format='jpeg',dpi=300)


	plt.show()

	# inundate raster by catchment
	#vv_vh_hand
	validationFromMaxHANDvalues = copy.deepcopy(validationRaster)
	# catchmask = gis.raster(catchmask_filename)
	table = pd.read_pickle(resultsTableFilename)
	validationCatchMasksTable = pd.read_pickle(validationCatchMasksTable_filename)

	catchmasks = validationCatchMasksTable.index.values
	validationFromMaxHANDvalues.array = np.zeros((validationFromMaxHANDvalues.nrows,validationFromMaxHANDvalues.ncols))

	# print(merged_inundated_dataframe['filteredPredictedHeights'])
	# print(len(np.unique(catchmasks)))
	# print(len(np.unique(merged_inundated_dataframe.index.values)))
	# print(validationCatchMasksTable)
	# exit()
	for cm in catchmasks:
		#predictedMAXHAND = table['predictedCatchMasksTables'][0]['predictedMaxHAND'][cm]

		try:
			validationHAND = merged_inundated_dataframe['filteredPredictedHeights'][cm]
		except KeyError:
			continue

		indicesOfCM = validationCatchMasksTable['indicesOfCatchmasks'][cm]
		handValues = vv_vh_hand.array[2,indicesOfCM[0],indicesOfCM[1]]
		inundatedValues = np.zeros((handValues.shape),dtype=np.ubyte)
		inundatedValues[handValues <= validationHAND] = 2
		inundatedValues[handValues > validationHAND] = 1
		validationFromMaxHANDvalues.array[indicesOfCM] = inundatedValues

	validationFromMaxHANDvalues.array = validationFromMaxHANDvalues.array.astype(np.ubyte)

	gis.writeGeotiff(validationFromMaxHANDvalues,'data/results/predictedInundation_filtered_{}.tif'.format(filters_list[i]),GDT_Byte)

	diffRaster = generateDifferenceRaster(validationFromMaxHANDvalues,validationRaster,writeTIFF=True,output_fileName='data/results/difference_filteredHeights_{}.tif'.format(filters_list[i]))

	binaryClassificationStats = calculateBinaryClassificationStatistics(diffRaster,mapping={'TP':1,'FP':2,'FN':3,'TN':4,'ndv':0})
	print("Filtered Height FIM Performance ...")
	print("F-Score: {:0.3f} | MCC: {:0.3f} | BACC: {:0.3f} | ACC: {:0.3f} | TPR: {:0.3f} | TNR: {:0.3f} | PPV: {:0.3f} | NPV: {:0.3f} | TP: {:0.3f} | FP: {:0.3f} | TN: {:0.3f} | FN: {:0.3f}".format(binaryClassificationStats['F1_score'],
																											binaryClassificationStats['MCC'],
																											binaryClassificationStats['BACC'],
																											binaryClassificationStats['ACC'],
																											binaryClassificationStats['TPR'],
																											binaryClassificationStats['TNR'],
																											binaryClassificationStats['PPV'],
																											binaryClassificationStats['NPV'],
																											binaryClassificationStats['TP_area']/((10**3)**2),
																											binaryClassificationStats['FP_area']/((10**3)**2),
																											binaryClassificationStats['TN_area']/((10**3)**2),
																											binaryClassificationStats['FN_area']/((10**3)**2)))


flowlines_heights_dataframe.to_file(flowlines_heights_filtered_filename_guide.format('shp'))
