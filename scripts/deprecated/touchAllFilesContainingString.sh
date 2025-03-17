#! /bin/bash
#define the directories where we will apply the script
defaultDirectories=(framework modules applications extlibs)
defaultFiles=("*.h" "*.inl" "*.cpp")
declare -a results
#counter of file touched
counter=0

#---------------------------------------------------------------------------
echo Touch files containing option \"$*\" inside [ ${defaultDirectories[@]} ]

for options in $*
do
    pattern="-e $options $pattern"
done

for file in $(seq 0 $((${#defaultFiles[@]} - 1)))
do    
    includeFiles="--include=${defaultFiles[$file]} $includeFiles"
done

for dir in $(seq 0 $((${#defaultDirectories[@]} - 1)))
do

    echo "   "Searching in ${defaultDirectories[$dir]}
    result=$(rgrep $pattern ${defaultDirectories[$dir]} $includeFiles -l)
    for file in $result
    do
        echo "      "found in $file
        touch $file
        counter=$(($counter+ 1))
    done
done

echo $counter files were touched
