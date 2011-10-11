#!/bin/bash
if [ "$1" != "--force" ]; then
echo "WARNING: This script will delete part of the files in the current directory."
echo "Make sure that you don't have any local changes not copied somewhere else."
echo "Press Enter to continue, or Ctrl-C to cancel."
read || exit 0
fi

shopt -s nullglob

DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
echo $SCRIPTS

let npro=0
let npriv=0
let nsrc=0
let nrmdir=0
let nrmfile=0
let nrmline=0

echo
echo "STEP 1: Filter project files and remove referenced files"
echo

function qmake_process_dir {
    cd "$1"
    echo "Entering $PWD"
    for f in *.pro *.pri *.cfg *.prf; do
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
	rm -f $f.dev
    done
    for f in *; do
	if [ -d "$f" ]; then
	    qmake_process_dir "$f"
	fi
    done
    echo "Leaving  $PWD"
    cd ..
}

qmake_process_dir "$DIR0"


echo
echo "STEP 2: Remove files listed in private.txt files"
echo

function private_process_dir {
    cd "$1"
    echo "Entering $PWD"
    if [ -f "private.txt" ]; then
	echo "Processing private.txt"
	let npriv+=1
	for g in $(grep -v '^#' "private.txt"); do
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
    fi
    for f in *; do
	if [ -d "$f" ]; then
	    private_process_dir "$f"
	fi
    done
    echo "Leaving  $PWD"
    cd ..
}

private_process_dir "$DIR0"

echo
echo "STEP 3: Filter source code files"
echo

function code_process_dir {
    cd "$1"
    echo "Entering $PWD"
    for f in *.h *.hpp *.hxx *.inl *.cuh *.cpp *.cxx *.c *.cu; do
	let nsrc+=1
	if grep -q 'SOFA_DEV' $f; then
	$SCRIPTS/filter-code.awk < $f > $f.new
	if [ $(wc -c < $f) != $(wc -c < $f.new) ]; then
	    nl=$(($(wc -l < $f)-$(wc -l < $f.new)))
	    let nrmline+=$nl
	    echo $nl "lines removed in source file" $f
	    cp -pf $f $f.bak
	    cat $f.new > $f
	fi
	rm -f $f.new
	fi
    done
    for f in *; do
	if [ -d "$f" ]; then
	    code_process_dir "$f"
	fi
    done
    echo "Leaving  $PWD"
    cd ..
}

code_process_dir "$DIR0"

echo
echo "Filtering complete."
echo

if [ $npro    -gt 0 ]; then echo -e $npro    "\tqmake project files processed."; fi
if [ $nsrc    -gt 0 ]; then echo -e $nsrc    "\tsource code files processed."; fi
if [ $nrmdir  -gt 0 ]; then echo -e $nrmdir  "\tdirectories removed."; fi
if [ $nrmfile -gt 0 ]; then echo -e $nrmfile "\tfiles removed."; fi
if [ $nrmline -gt 0 ]; then echo -e $nrmline "\tlines removed."; fi
