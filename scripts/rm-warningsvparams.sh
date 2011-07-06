#!/bin/bash
DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
cd -
#usage 
echo 'This script will perform the modifications to automatically remove warning: unused parameter ‘vparams’'
echo 'It performs make clean && make -j5 to get the gcc stderr, and parses that output
to remove the unused parameter `vparams’'
echo 'If you already know you have few warnings it is probably faster to remove them by hand'
echo 'You will have to run make again to update your binaries'
echo 'WARNING: Your files will be modified. Use at your own risk !'
echo 'Press Enter to continue, or Ctrl+C to abort.'
read || exit 1

echo 'clean project'
make clean &> /dev/null
echo 'run make...'
# make_err=$((
#         (
#             make -j5  2>&2
#         ) 1>/dev/null
#     ) 2>&1)
make_err=$( make -j5 2>&1 >/dev/null )
echo '...done'
#echo -e "$make_err" > make_err_file
#echo "$make_err" > make_err_file2
vparams_warnings=$(echo "$make_err" | grep vparams | uniq | sed s/‘vparams’/‘vparams’\\n/g)
#echo "$vparams_warnings" > vparams_warnings_file
echo 'processing vparams warnings...'
counter=0
i=0
for line in $vparams_warnings
do
    for f in $line
    do
    mod=$(($counter % 5))
    if [ $mod -eq 0 ]
    then
       parse=$(echo "$f" | awk 'BEGIN { FS = ":" } ; {  for(i=1;i<=NF;++i) print $i  }' | awk 'BEGIN { FS = "/" } ; {  for(i=NF;i>NF-1;--i) print $i  }' )
       file=$(echo $parse | cut -d ' ' -f 1)
       line=$(echo $parse | cut -d ' ' -f 2)
       file=$(find . -name $file)
       echo "removing warning in $file at line $line"
       sed_arg=$(echo $(( $line-1 )),$(( $line+1 )) s/vparams//)
       sed "$sed_arg" < "$file" > "$file".tmp 
	     if [ -s "$file".tmp ]
	     then
		       cat "$file".tmp > "$file"
           rm -f "$file".tmp
       fi
    fi
    counter=$(( $counter+1 ))
    done
done 
echo '...done'
