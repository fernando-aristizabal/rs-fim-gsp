#!/bin/bash -e

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo $parent_path
cd $parent_path

fullFile=aristizabal.tex
baseName="${fullFile%%.*}"
extension="${fullFile#*.}"

pdflatex "$fullFile"
bibtex "$baseName".aux
pdflatex "$fullFile"
pdflatex "$fullFile"




