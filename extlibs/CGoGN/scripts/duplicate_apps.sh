#!/bin/bash

if test $# -lt 2; then
	echo $0 application_directory_source application_directory_destination
	exit 2
fi

if test -d $2; then
	echo Directory $2 already exist
	exit 3
fi

if test ! -d $1; then
	echo Directory $1 does not exist
	exit 4
fi


echo "copying ..."
cp -r $1 $2
echo "cleanin ..."

cd $2
find . -name "CMakeFiles" -exec rm -rf {} \;  2> /dev/null
find . -name "cmake_install.cmake" -exec rm -f {} \; 2> /dev/null
find . -name "CMakeCache.txt" -exec rm -f {} \; 2> /dev/null
find . -name "Makefile" -exec rm -f {} \; 2> /dev/null
cd bin
rm -rf *
cd ..


app1=`basename $1`
app2=`basename $2`



app1_maj=`echo $app1 | awk '
BEGIN { upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lower = "abcdefghijklmnopqrstuvwxyz"
}
{
	FIRSTCHAR = substr($1, 1, 1)
	if (CHAR = index(lower, FIRSTCHAR))
		$1 = substr(upper, CHAR, 1) substr($1, 2)
	print $0
}' `


app2_maj=`echo $app2 | awk '
BEGIN { upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lower = "abcdefghijklmnopqrstuvwxyz"
}
{
	FIRSTCHAR = substr($1, 1, 1)
	if (CHAR = index(lower, FIRSTCHAR))
		$1 = substr(upper, CHAR, 1) substr($1, 2)
	print $0
}' `

echo remplace $app1 by $app2 in CMakeLists.txt ...
find . -name "CMakeLists.txt" -exec sed -i s/$app1/$app2/g {} \;

echo "renaming files:"
list_of_files=`find . -name $app1\*`

for f in $list_of_files; do
	nf=`echo $f | sed s/$app1/$app2/`
	echo  "   "$f -\> $nf
	mv $f $nf
	# search into files for inclusion
	bf=`basename $f`
	bnf=`basename $nf`
	list2=`fgrep -wl $bf *`
#	if test -n """$list2"""; then
#		echo "    files to modify (includes)" $list2
		for xx in $list2; do
			sed -i s/$bf/$bnf/g $xx
		done
#	fi
done

echo Modify contents ...
find . -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -exec sed -i s/${app1_maj}/${app2_maj}/g {} \;
find . -name "*.cpp" -exec sed -i s/${app1_maj}/${app2_maj}/g {} \;
find . -name "*.h" -exec sed -i s/${app1_maj}/${app2_maj}/g {} \;

	
echo finished

