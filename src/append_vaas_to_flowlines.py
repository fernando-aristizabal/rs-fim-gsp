#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import geopandas as gpd
from dbfread import DBF
import sys


"""
USAGE:
    ./append_vaas_to_flowlines.py <streamNetwork_fileName> <streamNetwork_vaa_fileName> <streamNetwork_outfile_fileName>

"""

streamNetwork_fileName = sys.argv[1]
streamNetwork_vaa_fileName = sys.argv[2]
streamNetwork_outfile_fileName = sys.argv[3]

streamNetwork = gpd.read_file(streamNetwork_fileName)

streamNetwork['TotDASqKM'] = np.nan ; streamNetwork['ToNode'] = int(0) ; streamNetwork['FromNode'] = int(0)
    
for i,record in enumerate(DBF(streamNetwork_vaa_fileName)):
    if i % 1000 == 0:
    print(i)
    current_comid = int(record['ComID'])
    drainageArea = float(record['TotDASqKM'])
    toNode = int(record['ToNode'])
    fromNode = int(record['FromNode'])

    current_comid_index = streamNetwork.index[streamNetwork['COMID'] == current_comid].to_list()

    streamNetwork.loc[current_comid_index,'TotDASqKM'] = drainageArea
    streamNetwork.loc[current_comid_index,'ToNode'] = toNode
    streamNetwork.loc[current_comid_index,'FromNode'] = fromNode

print(streamNetwork.columns.list)

streamNetwork.to_file(streamNetwork_outfile_fileName)
