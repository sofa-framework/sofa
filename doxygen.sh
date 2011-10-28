#! /bin/sh
outputdir=doc/doxygen
packageslist=`cat doc/doxygen-packages.txt`
tags=""

for package in ${packageslist}
do
    mkdir -p ${outputdir}/${package}
    cat Doxyfile.in | sed -e "s#@OUTPUT_DIR@#${outputdir}#g" -e "s#@PACKAGE@#${package}#g" -e "s#@TAGS@#${tags}#g" | doxygen - 2> ${outputdir}/${package}.log
    tags="${tags} ${outputdir}/${package}.tag=$PWD/${outputdir}/${package}/html"
done
