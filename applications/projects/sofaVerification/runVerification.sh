#! /bin/bash

#----------------------------------------------------------------------------------------------
#exit value matches the number of test which output a non-zero error value.

#----------------------------------------------------------------------------------------------
#default values
iterations=100
iniFile=verification.ini
resetMode=0
addMode=0


#----------------------------------------------------------------------------------------------
#Interpreting the command line
#options available:
# -i : to specify the number of iterations. By default, we use "100"
# -r : to save new references for the scenes: just record the positions of the dofs
# -f : to specify the reference set of files. By default, we use "verification.ini"
# -a : to add set of files to the reference set of files, and record them
readIterations=0
readFile=0
readAddFiles=0

for param in $*
do
#-----------------------------------------------
#read the -i option
    if [[ $readIterations == 1 ]] 
    then
	readIterations=0
	iterations=$param
    fi
    if [[ $param == -i ]] 
    then
	readIterations=1
    fi
#-----------------------------------------------
#read the -f option
    if [[ $readFile == 1 ]] 
    then
	readFile=0
	iniFile=$param
    fi  
    if [[ $param == -f ]] 
    then
	readFile=1
    fi


#-----------------------------------------------
#read the -a option
    if [[ $readAddFiles == 1 ]] 
    then
	readAddFiles=0
	addFiles=$param
	addMode=1
	resetMode=1
    fi
    if [[ $param == -a ]] 
    then
	readAddFiles=1
    fi

#-----------------------------------------------
#read the -r option
    if [[ $param == -r ]] 
    then
	resetMode=1
    fi
    
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
arguments=$(echo)
arguments=$(echo -i $iterations)


if [[ $resetMode == 1 ]] 
then
    arguments=$(echo $arguments "-r")
    echo Reseting the Verification of files
else
    echo Run Verification with $iterations iterations: Set of files $iniFile
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
	presence=$( cat $iniFile | grep $file )
	if [[ $presence != $file ]]
	then
	    #file not present
	    echo $file >> $iniFile
	    echo Adding $file in $iniFile
	fi
    fi    

    output=$( sofaVerificationd $file $arguments  2> /dev/null)

    #test if the scene did load
    if [[ $? != 0 ]]
    then
	lineFormatation=$( echo "|  SegFault")
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

