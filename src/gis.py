#!/usr/bin/env python3

import rasterio 
import geopandas

def smart_open(in_obj):
    
    """ smart open for rasterio objects for use within functions """

    if isinstance(in_obj,rasterio.DatasetReader):
        return(in_obj)
    elif isinstance(in_obj,str):
        return(rasterio.open(in_obj))
    else:
        raise TypeError("Pass Rasterio Dataset Reader or filepath to raster as a string")

rasterio.smart_open = smart_open


def smart_write(filename,driver=None,layer=None,index=False):
    
    """ smart write for geopandas geodataframes for use with file extensions as driver """

    # sets driver
    driverDictionary = {
                        '.gpkg' : 'GPKG',
                        '.geojson' : 'GeoJSON',
                        '.shp' : 'ESRI Shapefile'
                       }

    if driver is not None:
        driver = driverDictionary[splitext(fileName)[1]]

    self.to_file(fileName, driver=driver, layer=layer, index=index)

geopandas.GeoDataFrame.smart_write = smart_write
