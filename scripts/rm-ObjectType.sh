#!/bin/bash
# use: ./rm-ObjectType.sh directoryPath
# Warning do not use current directory or script will be changed as well


grep -rl 'Object type=' ${1} | grep -v svn >> fileObject.log

while read filename
do
  while read line
  do
    component=$(echo ${line} | awk -F 'Object type="' '{print $2}')
    component=$(echo ${component} | awk -F '"' '{print $1}')
#     echo "compo: $component"

    echo ${filename} | xargs sed -i 's/Object type="'"$component"'"/'"$component"'/'
  
  done < ${filename}
done < "fileObject.log"

rm fileObject.log
