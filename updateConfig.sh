#! /bin/bash
#define the directories were to apply the script
defaultDirectories=(framework modules applications extlibs)
#array containing the error, and the time for each scene
declare -a results
#counter of file touched
counter=0

#---------------------------------------------------------------------------
echo Touch files containing option \"$*\" inside ${defaultDirectories[@]}

for options in $*
do
    pattern="-e $options $pattern"
done

for dir in $(seq 0 $((${#defaultDirectories[@]} - 1)))
do

    echo "   "Searching in ${defaultDirectories[$dir]}
    result=$(rgrep $pattern ${defaultDirectories[$dir]} --include=*.h --include=*.inl --include=*.cpp -l)
    for file in $result
    do
        echo "      "found in $file
        touch $file
        counter=$(($counter+ 1))
    done
done

echo $counter files were touched
