size=0

for g in $(find $1 -name *.h -o -name *.cpp -o -name *.inl) 
do
	size=$(($size+1))
done

#echo "$size"

current=0
for g in $(find $1 -name *.h -o -name *.cpp -o -name *.inl) 
do
	./rem-componentmodel.awk $g > buff
	cat buff > $g
	current=$(($current+1))
	#rate=$(echo "$current / $size" | bc -l)
	rate=$(($current*100 / $size))
	echo -ne "Progression : $rate%\r"
done

rm -f buff
