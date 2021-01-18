#!/bin/bash

file=$1
sourceDirectory = $2

for i in $sourceDirectory/*.shp
do

	if [ -f “$file” ]
	then
		echo “creating outfile”
		ogr2ogr -f ‘ESRI Shapefile’ -update -append $file $i -nln merge
	else
	        echo “merging……”
	ogr2ogr -f ‘ESRI Shapefile’ $file $i
fi
done
