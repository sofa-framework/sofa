#!/bin/bash
# use: ./replace-object12-IO.sh directoryPath
# This script changes in all the scenes the deprecated structure for mapping:
# object1 --> input
# object2 --> output

# WARNING: this script may modify your scene
# make sure you saved/commited all local changes before apply this script

DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
cd -


#while read filename
#(taking into account spaces in the name)

find ${1:-.} '(' -name '*.scn' -o -name '*.pscn' -o -name '*.xml' -o -name '*.php' -o -name '*.tex' ')' -print0 | while read -rd $'\0' g;
do
#to visualize all file read: echo "$g"

  # check if file is not empty
  if [ -s "$g" ]
  then
    # apply the script sed: including input/output format
    "$SCRIPTS/replace-object12-IO.sed" < "$g" > "$g".tmp
    if [ -s "$g".tmp ]
    then
      cat "$g".tmp > "$g"
      rm -f "$g".tmp
    else
      echo "Error in $g"
    fi

    # apply the script sed: only adding the @-path
    "$SCRIPTS/replace-object12-IO-pathOnly.sed" < "$g" > "$g".tmp
    if [ -s "$g".tmp ]
    then
      cat "$g".tmp > "$g"
      rm -f "$g".tmp
    else
      echo "Error(2) in $g"
    fi

  # if file is empty
  else
    echo "File $g is empty"
  fi

done
