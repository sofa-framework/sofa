#! /bin/bash

#----------------------------------------------------------------------------------------------
#exit value matches the number of test which output a non-zero error value.

#----------------------------------------------------------------------------------------------
#default values
iniFile=initTimer.ini
resetMode=0
addMode=0
defaultDirectory=1
defaultPlugins="libsofaframe.so"

#----------------------------------------------------------------------------------------------
#Interpreting the command line
#options available:
# -r : to save new references for the scenes: just record the positions of the dofs
# -f : to specify the reference set of files. By default, we use "verification.ini"
# -a : to add set of files to the reference set of files, and record them

while [[ $# -gt 0 ]]
do
    case "$1" in
    -f) iniFile=$2    ; shift ;;
    -a) addFiles=$2
        addMode=1
	resetMode=1
        shift ;;
    -p) plugins=$2 ; shift ;;
    -r) resetMode=1 ;;
    -d) directory=$2 ; shift ;;
    *) echo "Invalid argument $1"; exit 1 ;;
    esac
    shift
done

#----------------------------------------------------------------------------------------------
#array containing only text
declare -a textResults=(Error: Time:)
#array containing the error, and the time for each scene
declare -a results
#array counter of scenes
declare -i counter=0
declare -i nerrors=0 
#the script performs a negative count of the error number

#building the parameters passed to sofaVerification
if [[ $resetMode == 1 ]] 
then
    arguments=$(echo $arguments -r)
    echo Reseting the InitTimer of files
else
    echo Run InitTimer with $iterations iterations: Set of files $iniFile
fi

if [[ -n $directory ]]
then
    arguments=$(echo $arguments -d $directory)
    echo Using directory: $directory
else
    echo Using default directory
fi

if [[ -n $plugins ]]
then
    arguments=$(echo $arguments -p $plugins)
    echo Using plugins: $plugins
else
    arguments=$(echo $arguments -p $defaultPlugins)
    echo Using default plugins: $defaultPlugins
fi

#select the file containing the set of simulation to be processed
if [[ $addMode == 1 ]] 
then 
    setFiles=$addFiles
else
    setFiles=$iniFile
fi


#----------------------------------------------------------------------------------------------
for file in $(cat $setFiles)
do 
    if [[ $addMode == 1 ]] 
    then
	#add mode, we have to add the list of files contained 
	presence=$(cat $iniFile | grep $file )
	if [[ $presence != $file ]]
	then
	    #file not present
	    echo $file >> $iniFile
	    echo Adding $file in $iniFile
	fi
    fi    

    #output=$(sofaInitTimer $file $arguments 2> /dev/null)
    #output=$(sofaInitTimer $file $arguments)
    echo $(sofaInitTimer $file $arguments)

    #test if the scene did load
    if [[ $? != 0 ]]
    then
	lineFormatation=$(echo "|  SegFault")
	nerrors=$(($nerrors + 1))
    else
	declare -i displayByDof=0
	declare -i displayTime=0
	for words in $output
	do
#----------------------------------------------------------------------------------------------
#find the information about the time spent
	    if [[ $displayTime == 1 ]]
	    then
		line=$(echo $words ${results[$counter]})
		results[$counter]=$line
		displayTime=0
	    fi
	    if [[ $words == TIME ]] 
	    then
		displayTime=1
	    fi
#----------------------------------------------------------------------------------------------
	    if [[ $displayByDof == 1 ]]
	    then 
 		if test "$words" != "0"  
	  then 
	nerrors=$(($nerrors+1))
	  fi
		line=$(echo $words ${results[$counter]})
		results[$counter]=$line
		displayByDof=0
	    fi

	    if [[ $words == ERRORBYDOF ]] 
	    then
		displayByDof=1
	    fi
#----------------------------------------------------------------------------------------------
	done

	declare -i idx=0
	lineFormatation=$( echo "| ")

#no reset mode: print the result for the error and time spent
	if [[ $resetMode == 0 ]] 
	then
	    for data in ${results[$counter]}
	    do
		lineFormatation=$(echo $lineFormatation ${textResults[$idx]} $data " | " )
		idx=$(( $idx+1 ))
	    done
	fi
    fi
    lineFormatation=$(echo $lineFormatation  File: $file)
    echo $lineFormatation	
    counter=$(( $counter+1 ))
done
exit $nerrors

