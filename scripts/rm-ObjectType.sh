#!/bin/bash
# use: ./rm-ObjectType.sh directoryPath
# This script removes in all the scenes the deprecated structure <Object type="MyComponent"

# WARNING: this script may modify your scene
# make sure you saved/commited all local changes before apply this script

DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
cd -


#while read filename
#(taking into account spaces in the name)

find ${1:-.} '(' -name '*.scn' -o -name '*.pscn' -o -name '*.xml' ')' -print0 | while read -rd $'\0' g;
do
#to visualize all file read: echo "$g"

  # check if file is not empty
  if [ -s "$g" ]
  then
    # apply the scrit sed
    "$SCRIPTS/rm-ObjectType.sed" < "$g" > "$g".tmp
    if [ -s "$g".tmp ]
    then
      cat "$g".tmp > "$g"
      rm -f "$g".tmp
    else
      echo "Error in $g"
    fi

  # if file is empty
  else
    echo "File $g is empty"
  fi

done