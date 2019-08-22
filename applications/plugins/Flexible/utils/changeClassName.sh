#!/bin/sh

# Replace a class name.
# Arguments: OldClassName NewClassName
# This script:
# - replaces the old class name by the new class name in all the files;
# - changes all file names containing the old class name 
#   by new names with the new class name;
#

# number of arguments must be 2
if ! test $# -eq 2 
then
	echo arguments: OldClassName NewClassName
	exit 1
fi

# replace the strings in the files
for i in * 
do
	if [ -f $i ]
	then
		sed -i s/$1/$2/g $i
	fi
done

# change the file names
for i in `ls *$1*`
do
	mv $i `echo $i | sed s/$1/$2/g`
done


