#!/bin/bash -e

codeDir=$1

docker run --rm -v "$1":/project fernandoaristizabal/latex-full:rs_fim_gsp_20210117 /project/Makefile.sh
