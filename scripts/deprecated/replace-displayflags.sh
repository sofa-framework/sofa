#!/bin/bash
DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
cd -
size=0

for g in $(find ${1:-.} -name '*.h' -o -name '*.cpp' -o -name '*.inl') 
do
	size=$(($size+1))
done

echo "$size"
for g in $(find ${1:-.} -name '*.h' -o -name '*.cpp' -o -name '*.inl') 
do
	  #echo "$g"
    #var=$( echo "$g" | sed  s/^\./sofa/ )
    #var=$( echo "$var" | sed s/\.inl$/\.h/)
    headerfile=$( echo "$g" | sed s/\.cpp$/\.h/)
    inlinefile=$( echo "$g" | sed s/\.cpp$/\.inl/ )
    if [ $headerfile != $g ]  # test if the current file is a .cpp file
    then
        #echo  "$headerfile"

        if [ -f "$inlinefile" ] # test if there exist a matching .inl file for template class
        then
            includetest=$( echo "$headerfile" | sed s/^\.// | sed -e 's/\//\\\//g' )
            #echo "$includetest"
            sed '/'"$includetest"'/ a\#include <sofa/core/visual/VisualParams.h>' < "$inlinefile" > "$inlinefile".tmp
            if [ -s "$inlinefile".tmp ]
            then
                cat "$inlinefile".tmp > "$inlinefile"
                rm -f "$inlinefile".tmp
            fi
        else
            if [ -f "$headerfile" ] # not a template a class
            then
                includetest=$( echo "$headerfile" | sed s/^\.// | sed -e 's/\//\\\//g' )
                #echo "$includetest"
                sed '/'"$includetest"'/ a\#include <sofa/core/visual/VisualParams.h>' < "$g" > "$g".tmp
                if [ -s "$g".tmp ]
                then
                    cat "$g".tmp > "$g"
                    rm -f "$g".tmp
                fi
                
            fi
        fi
    fi
    
    "$SCRIPTS/replace-displayflags.sed" < "$g" > "$g".tmp
	  if [ -s "$g".tmp ]
	  then
		    cat "$g".tmp > "$g"
        rm -f "$g".tmp
		    current=$(($current+1))
		#rate=$(echo "$current / $size" | bc -l)
		    rate=$(($current*100 / $size))
		    echo -ne "Progression : $rate%\r"
	  else
		    echo "Error in $g"
	  fi

done
