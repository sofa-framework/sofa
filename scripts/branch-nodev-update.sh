#!/bin/bash

SVN="svn --non-interactive"

if [ $# -ne 5 ]; then
echo USAGE: $# $0 source_url source_rev0 source_rev1 source_dir dest_dir
exit 1
fi
echo "WARNING: This script will delete part of the files in the $4 and $5 directories."
echo "Make sure that you don't have any local changes not copied somewhere else."
echo "Press Enter to continue, or Ctrl-C to cancel."
read || exit 1

SVN_URL="$1"
SVN_REVA="$2"
SVN_REVB="$3"

shopt -s nullglob

DIR0=$PWD
cd ${0%/*} || exit 1
SCRIPTS=$PWD
echo "SCRIPTS:" $SCRIPTS
cd -
cd $4 || exit 1
SOURCE=$PWD
echo "SOURCE:" $SOURCE
cd -
cd $5 || exit 1
DEST=$PWD
echo "DEST:" $DEST
cd -

#exit 0

cd $SOURCE

echo
echo ========== Clean Source Directory ==========
echo

$SVN revert -R .
$SVN status --no-ignore | grep '^?' | colrm 1 7 | xargs -d '\n' rm -rf

echo
echo ========== Update Source Directory to $SVN_REVB ==========
echo

$SVN update -r $SVN_REVB
$SVN status --no-ignore | grep '^?' | colrm 1 7 | xargs -d '\n' rm -rf

echo
echo ========== Filter Source Directory ==========
echo

$SCRIPTS/filter-all.sh --force

cd $DEST

echo
echo ========== Clean Branch Directory ==========
echo

$SVN revert -R .
$SVN status --no-ignore | grep '^?' | colrm 1 7 | xargs -d '\n' rm -rf

echo
echo ========== Update Branch Directory ==========
echo

$SVN update || exit 1
$SVN status --no-ignore | grep '^?' | colrm 1 7 | xargs -d '\n' rm -rf

echo
echo ========== Merge r$SVN_REVA:$SVN_REVB, ignoring conflicts ==========
echo

$SVN merge --accept theirs-full -r $SVN_REVA:$SVN_REVB $SVN_URL || exit 1

echo
echo ========== SVN copy all missing directories and files ==========
echo

function svncopy_process_dir {
    CUR="$1"
    DEL=`LC_ALL=C svn info "$SOURCE/$1" | grep -c '^Schedule: delete$'`
    if [ $DEL -gt 0 ]; then
	true # ignore to-be-deleted directories
    elif [ -f "$DEST/$1" ]; then
        echo "ERROR: Directory $1 conflicts with existing file." >&2
        echo "The file will be removed, but the directory will only be created in a later commit." >&2
        $SVN rm --force "$DEST/$1" || exit 1
	rm -rf "$DEST/$1"
    elif [ ! -d "$DEST/$1" ]; then
        C_URL=`LC_ALL=C $SVN info $SOURCE/$1 | awk '$1=="URL:" { print $2 }'`
        C_REV=`LC_ACC=C $SVN info $SOURCE/$1 | awk '$1=="Last" && $2=="Changed" && $3=="Rev:" { print $4 }'`
        echo Copy directory $1 '@' $C_REV
        $SVN cp $C_URL'@'$C_REV "$DEST/$1" || exit 1
    else
	cd "$SOURCE/$1"
	for f in *; do
	    if [ -f "$f" ]; then
		if [ "${f##*.}" == "bak" ]; then
		    true # ignore backup files
		elif [ -d "$DEST/$2$f" ]; then
		    echo "ERROR: File $2$f conflicts with existing directory." >&2
		    echo "The directory will be removed, but the file will only be created in a later commit." >&2
		    $SVN rm --force "$DEST/$2$f" || exit 1
		elif [ ! -f "$DEST/$2$f" ]; then
		    C_URL=`LC_ALL=C $SVN info $SOURCE/$1 | awk '$1=="URL:" { print $2 }'`
		    C_REV=`LC_ALL=C $SVN info $SOURCE/$1 | awk '$1=="Last" && $2=="Changed" && $3=="Rev:" { print $4 }'`
		    echo Copy file $1 '@' $C_REV
		    $SVN cp $C_URL'@'$C_REV "$DEST/$2$f" || exit 1
		fi
	    fi
	done
	for f in *; do
	    if [ -d "$f" ]; then
		svncopy_process_dir "$2$f" "$2$f/"
	    fi
	done
	cd ..
    fi
}

svncopy_process_dir "" ""


echo
echo ========== SVN remove all deleted directories and files ==========
echo

function svnrm_process_dir {
    CUR="$1"
    if [ ! -d "$SOURCE/$1" ]; then
        echo Removing directory $1
        $SVN rm --force "$DEST/$1" || exit 1
    else
	DEL=`LC_ALL=C svn info "$SOURCE/$1" | grep -c '^Schedule: delete$'`
	if [ $DEL -gt 0 ]; then
            echo Removing directory $1
            $SVN rm --force "$DEST/$1" || exit 1
	else
	    cd "$DEST/$1"
	    for f in *; do
		if [ -f "$f" ]; then
		    if [ ! -f "$SOURCE/$2$f" ]; then
			echo Removing file "$DEST/$2$f"
			$SVN rm --force "$DEST/$2$f" || exit 1
			rm -f "$DEST/$2$f"
		    else
			DEL=`LC_ALL=C svn info "$SOURCE/$2$f" | grep -c '^Schedule: delete$'`
			if [ $DEL -gt 0 ]; then
			    echo Removing file "$DEST/$2$f"
			    $SVN rm --force "$DEST/$2$f" || exit 1
			    rm -f "$DEST/$2$f"
			fi
		    fi
		fi
	    done
	    for f in *; do
		if [ -d "$f" ]; then
		    svnrm_process_dir "$2$f" "$2$f/"
		fi
	    done
	    cd ..
	fi
    fi
}

svnrm_process_dir "" ""

echo
echo ========== COPY all files ==========
echo

function cp_process_dir {
    CUR="$1"
    if [ -d "$SOURCE/$1" ]; then
	DEL=`LC_ALL=C svn info "$SOURCE/$1" | grep -c '^Schedule: delete$'`
	if [ $DEL -eq 0 ]; then
	    cd "$DEST/$1"
	    for f in *; do
		if [ -f "$f" ]; then
		    if [ -f "$SOURCE/$2$f" ]; then
			cp -pf "$SOURCE/$2$f" "$DEST/$2$f" || exit 1
		    fi
		fi
	    done
	    for f in *; do
		if [ -d "$f" ]; then
		    cp_process_dir "$2$f" "$2$f/"
		fi
	    done
	    cd ..
	fi
    fi
}

cp_process_dir "" ""

echo
echo ========== MERGE complete ==========
echo

# Check concistency

diff -qwrU3 -x '.svn' -x '*.bak' $SOURCE $DEST
