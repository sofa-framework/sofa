#!/bin/bash
if [ ! -d $1/boost ]; then
echo Please specify the directory containing boost, it should contain a boost sub-directory.
exit 1
fi
rm -rf oldboost newboost
mv boost oldboost
cp -pr $1/boost boost
./gen-all-includes.sh
mkdir newboost
for f in $(cat all-includes.txt); do echo new${f%/*}; done | sort | uniq | xargs mkdir -p
for f in $(cat all-includes.txt); do mv $f new${f%/*}; done
rm -rf boost
mv oldboost boost
../../scripts/svn-copy.sh newboost boost
