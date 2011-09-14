#!/bin/bash
DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
cd -
size=0

for g in $(find ${1:-.} -name '*.scn' -o -name '*.xml') 
do
	size=$(($size+1))
done

echo "$size"
for g in $(find ${1:-.} -name '*.scn' -o -name '*.xml' ) 
do
	err=$(xmlconvert-displayflagsd "$g" > "$g".tmp )
  if [[ $? == 0 ]]
  then
	    if [ -s "$g".tmp ]
	    then
		      cat "$g".tmp > "$g"
          rm -f "$g".tmp
		      current=$(($current+1))
		#rate=$(echo "$current / $size" | bc -l)
		      rate=$(($current*100 / $size))
		      echo -ne "Progression : $rate%\r"
	    else
		      echo "Error in $g"
	    fi
  else
      echo "Error with file "$g""
      echo "Reverting file..."
      svn revert "$g"
  fi
done
