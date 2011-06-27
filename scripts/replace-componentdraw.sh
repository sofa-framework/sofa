#!/bin/bash
DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
cd -
size=0

for g in $(find ${1:-.} -name '*.h' -o -name '*.cpp' -o -name '*.inl') 
do
	size=$(($size+1))
done

echo "$size"
for g in $(find ${1:-.} -name '*.h' -o -name '*.cpp' -o -name '*.inl') 
do
	#echo "$g"
	sed -f "$SCRIPTS/replace-componentdraw.sed" "$g" > buff
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
