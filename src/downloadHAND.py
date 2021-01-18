from urllib.request import urlretrieve
import os

hucList = [120401,120402,120702,120701,120302,120903,120702,120200,120904,121001,121002]
extensions=['dbf','shx','shp','prj']
streamNetwork_linkFrameWork = "https://web.corral.tacc.utexas.edu/nfiedata/HAND/{0}/{0}-flows.{1}"
wbd_linkFrameWork = "https://web.corral.tacc.utexas.edu/nfiedata/HAND/{0}/{0}-wbd.{1}"
hand_linkFrameWork = "https://web.corral.tacc.utexas.edu/nfiedata/HAND/{0}/{0}hand.tif"
catchments_linkFrameWork = "https://web.corral.tacc.utexas.edu/nfiedata/HAND/{0}/{0}catchmask.tif"
targetDirectory = os.path.join("/home","fernandoa","data","graphSignals","texas","hand","original")
wbd_targetFileName = "{0}-wbd.{1}"
hand_targetFileName = "{0}hand.tif"
catchments_targetFileName = "{0}catchmask.tif"
streamNetwork_targetFileName = "{0}-flows.{1}"

for i in hucList:
    
    print(i)
    for ii in extensions:
        wbd_url = wbd_linkFrameWork.format(i,ii)
        wbd_filePath = os.path.join(targetDirectory,wbd_targetFileName.format(i,ii))
  #      urlretrieve(wbd_url,wbd_filePath)

        streamNetwork_url = streamNetwork_linkFrameWork.format(i,ii)
        #print(streamNetwork_url)
        streamNetwork_filePath = os.path.join(targetDirectory,streamNetwork_targetFileName.format(i,ii))
 #       urlretrieve(streamNetwork_url,streamNetwork_filePath)
#exit()
for i in hucList:
    print(i)
    hand_url = hand_linkFrameWork.format(i)
    hand_filePath = os.path.join(targetDirectory,hand_targetFileName.format(i))
    urlretrieve(hand_url,hand_filePath)

    catchments_url = catchments_linkFrameWork.format(i)
    catchments_filePath = os.path.join(targetDirectory,catchments_targetFileName.format(i))
#    urlretrieve(catchments_url,catchments_filePath)
    


