#!/bin/bash

opt_force=0
opt_release=0
opt_clean=0

RMCMD=${RMCMD:-svn rm --force}

while [ $# -gt 0 ]
do
    case "$1" in
        --force )
            opt_force=1
            ;;
        --release )
            opt_release=1
            ;;
        --rm )
            RMCMD="rm -rf"
            ;;
        --clean )
            opt_clean=1
            ;;
    esac
    shift
done

if [ $opt_force -eq 0 ]; then
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

function qmake_remove_project {
    dir="${1%/*}"
    pro="${1##*/}"
    pushd "$dir"
    echo "Removing project $pro in $dir"
	let npro+=1
    for g in $($SCRIPTS/files-from-qmake-pro.awk < $pro); do
	    if [ -d "$g" ]; then
		    echo "Remove directory $g"
		    let nrmdir+=1
		    $RMCMD $g || echo "Failed to remove directory $g";
	    elif [ -f "$g" ]; then
            if [ "${g: -4}" == ".pro" ]; then
                qmake_remove_project "$g"
            else
		        echo "Remove file $g"
		        let nrmfile+=1
		        $RMCMD $g || echo "Failed to remove file $g";
            fi
	    else
		    echo "$g already removed."
	    fi
	done
    # note that we can't use dir and pro variables here, as then might have been erased by recursion
    popd
	echo "Remove file $1"
	let nrmfile+=1
	$RMCMD $1 || echo "Failed to remove file $1";
}

function qmake_process_dir {
    cd "$1"
    echo "Entering $PWD"
    for f in *.pro *.pri *.cfg *.prf; do
	    echo "Processing qmake project file $f"
	    let npro+=1
	    cp -pf $f $f.bak
	    $SCRIPTS/filter-qmake-pro.awk -v FILTER_TAG=SOFA_DEV < $f.bak > $f 2> $f.dev
        if [ $opt_release -gt 0 ]; then
	        $SCRIPTS/filter-qmake-pro.awk -v FILTER_TAG=SOFA_RELEASE < $f > $f.release 2> $f.unstable
            if [ $(wc -l < "$f".release) -ne $(wc -l < "$f") ]; then
                echo "Project file $f filtered for release"
                mv -f $f.release $f
            else
                rm -f $f.release
            fi
            if [ $(wc -l < "$f".unstable) -gt 0 ]; then
                echo $(wc -l < "$f".unstable) " files and directory removed from release"
                cat $f.unstable >> $f.dev
            fi
            rm -f "$f".unstable
        fi
#        cp -pf $f $f.nodev
	    for g in $(cat $f.dev); do
	        if [ -d "$g" ]; then
		        echo "Remove directory $g"
		        let nrmdir+=1
		        $RMCMD $g || (echo "Failed to remove directory $g"; mv -f $f.bak $f ; exit 1)
	        elif [ -f "$g" ]; then
                if [ "${g: -4}" == ".pro" ]; then
                    qmake_remove_project "$g"
                else
		            echo "Remove file $g"
		            let nrmfile+=1
		            $RMCMD $g || (echo "Failed to remove file $g"; mv -f $f.bak $f ; exit 1)
                fi
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

    for f in *.pro *.pri *.cfg *.prf; do
        if [ "$f" == "sofa-dependencies.prf" ]; then
	        echo "Post-processing qmake project file $f"
            cp -pf $f $f.tmp
            $SCRIPTS/post-filter-qmake-prf.awk -v features="$DIR0"/features pass=1 $f.tmp pass=2 $f.tmp > $f
            rm -f $f.tmp
        fi
        if [ $(wc -w < "$f") -eq 0 -a $(wc -w < "$f".bak) -gt 0 ]; then
		    echo "Remove file $f"
		    let nrmfile+=1
		    $RMCMD $f || (echo "Failed to remove file $f"; mv -f $f.bak $f; exit 1)
        elif [ $(wc -l < $f.bak) -gt $(wc -l < $f) ]; then
	        nl=$(($(wc -l < $f.bak)-$(wc -l < $f)))
	        let nrmline+=$nl
	        echo $nl "lines removed in project file" $f
        fi
        if [ $opt_clean -gt 0 ]; then
	        rm -f $f.bak
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
		$RMCMD $g || (echo "Failed to remove directory $g"; exit 1)
	    elif [ -f "$g" ]; then
		echo "Remove file $g"
		let nrmfile+=1
		$RMCMD $g || (echo "Failed to remove file $g"; exit 1)
	    else
		echo "$g already removed."
	    fi
	done
	echo "Remove file private.txt"
	let nrmfile+=1
	$RMCMD private.txt || (echo "Failed to remove file $f"; exit 1)
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
        if [ $opt_clean -eq 0 ]; then
	        cp -pf $f $f.bak
        else
            rm -f $f.bak
        fi
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
