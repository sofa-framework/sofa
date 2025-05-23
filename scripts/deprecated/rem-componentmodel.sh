#!/bin/bash
DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
cd -
#echo $SCRIPTS

size=0

for g in $(find ${1:-.} -name '*.h' -o -name '*.cpp' -o -name '*.inl') 
do
	size=$(($size+1))
done

echo "$size"

current=0
for g in $(find ${1:-.} -name '*.h' -o -name '*.cpp' -o -name '*.inl') 
do
	#echo "$g"
	awk -f "$SCRIPTS/rem-componentmodel.awk" "$g" > buff
 
	if [ -s buff ]
	then
		cat buff > "$g"
		current=$(($current+1))
		#rate=$(echo "$current / $size" | bc -l)
		rate=$(($current*100 / $size))
		echo -ne "Progression : $rate%\r"
	else
		echo "Erreur in $g"
	fi
done

#rm -f buff
