#!/bin/bash
for i in *files.txt; do
echo $i:
echo -n "    Files: "
wc -l < $i
echo -n "    Lines: "
(cd ../src ; cat `cat ../doc/$i` | wc -l)
echo -n "    Bytes: "
(cd ../src ; cat `cat ../doc/$i` | wc -c)
done
