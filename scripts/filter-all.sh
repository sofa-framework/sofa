#!/bin/bash
echo "WARNING: This script will delete part of the files in the current directory."
echo "Make sure that you don't have any local changes not copied somewhere else."
echo "Press Enter to continue, or Ctrl-C to cancel."
read || exit 0

shopt -s nullglob

DIR0=$PWD
cd ${0%/*}
SCRIPTS=$PWD
echo $SCRIPTS

let npro=0
let nsrc=0
let nrmdir=0
let nrmfile=0
let nrmline=0

echo
echo "STEP 1: Filter project files and removing referenced files"
echo

function qmake_process_dir {
    cd $1
    echo "Entering $PWD"
    for f in *.pro *.pri *.cfg; do
	echo "Processing qmake project file $f"
	let npro+=1
	cp -pf $f $f.bak
	$SCRIPTS/filter-qmake-pro.awk < $f.bak > $f 2> $f.dev
	for g in $(cat $f.dev); do
	    if [ -d "$g" ]; then
		echo "Remove directory $g"
		let nrmdir+=1
		svn rm --force $g || (echo "Failed to remove directory $g"; mv -f $f.bak $f ; exit 1)
	    elif [ -f "$g" ]; then
		echo "Remove file $g"
		let nrmfile+=1
		svn rm --force $g || (echo "Failed to remove directory $g"; mv -f $f.bak $f ; exit 1)
	    else
		echo "$g already removed."
	    fi
	done
    done
    for f in *; do
	if [ -d "$f" ]; then
	    qmake_process_dir $f
	fi
    done
    echo "Leaving  $PWD"
    cd ..
}

qmake_process_dir $DIR0


echo
echo "STEP 2: Filter source code files"
echo

function code_process_dir {
    cd $1
    echo "Entering $PWD"
    for f in *.h *.hpp *.hxx *.inl *.cpp *.c *.cu *.cxx; do
	let nsrc+=1
	$SCRIPTS/filter-code.awk < $f > $f.new
	if [ $(wc -c < $f) != $(wc -c < $f.new) ]; then
	    nl=$(($(wc -l < $f)-$(wc -l < $f.new)))
	    let nrmline+=$nl
	    echo $nl "lines removed in source file" $f
	    cp -pf $f $f.bak
	    cat $f.new > $f
	fi
	rm -f $f.new
    done
    for f in *; do
	if [ -d "$f" ]; then
	    code_process_dir $f
	fi
    done
    echo "Leaving  $PWD"
    cd ..
}

code_process_dir $DIR0

echo
echo "Filtering complete."
echo

if [ $npro    -gt 0 ]; then echo -e $npro    "\tqmake project files processed."; fi
if [ $nsrc    -gt 0 ]; then echo -e $nsrc    "\tsource code files processed."; fi
if [ $nrmdir  -gt 0 ]; then echo -e $nrmdir  "\tdirectories removed."; fi
if [ $nrmfile -gt 0 ]; then echo -e $nrmfile "\tfiles removed."; fi
if [ $nrmline -gt 0 ]; then echo -e $nrmline "\tlines removed."; fi
