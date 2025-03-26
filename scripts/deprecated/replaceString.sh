#!/bin/sh

# Replace a string by another in all the files of the current directory
# Arguments: OldString NewString

# number of arguments must be 2
if ! test $# -eq 2 
then
	echo arguments: OldString NewString
	exit 1
fi

# replace the strings in the files
for i in * 
do
	if  [ -f $i ]
	then
		sed -i s/$1/$2/g $i
	fi
done



