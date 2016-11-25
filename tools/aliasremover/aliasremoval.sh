#!/bin/bash
EXAMPLE_FOLDER=../../examples
TEMP_FILE=post_checking_list.log

## Pre-check
which runSofa

if [ $? -ne 0 ]
then 
    echo "no runSofa application found"
    exit 1
fi

## Usage
display_usage() 
{ 
    echo "This script replaces alias components in any SOFA scene (XML)." 
    echo -e "\nUsage:\n$0 [name of the dictionary file containing aliases] \n" 
} 

## Check argument
if [  $# -ne 1 ] 
then 
    display_usage
    exit 1
fi

## Here is where the magic happens
cat $1 | while read sofaAlias sofaRealComponent; do    
    # Save list for post-checking
    grep -rl '<'$sofaAlias $EXAMPLE_FOLDER > $TEMP_FILE

    # Do the substitution
    # Pattern 1 = <Alias ???
    find $EXAMPLE_FOLDER -type f -name '*.scn' -exec sed -i '' 's/<'$sofaAlias' /<'$sofaRealComponent' /' {} +

    # Post-checking
    cat $TEMP_FILE | while read line; do
        $SOFA_EXEC -g batch -n 10 $line
    done
done

## Clean temporary file 
rm -f $TEMP_FILE
