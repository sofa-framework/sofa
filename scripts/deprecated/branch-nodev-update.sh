#!/bin/bash

SVN="svn --non-interactive"

if [ $# -ne 6 ]; then
echo USAGE: $0 source_url source_rev0 source_rev1 source_dir dest_dir tmp_dir
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
cd $6 || exit 1
TMPD=$PWD
cd -

SVNV=`$SVN --version | head -1 | awk '{ print $3 }'`
if [ ${SVNV:0:1} -gt 1 -o ${SVNV:2:1} -gt 5 ]; then
echo SVN version 1.6+ "($SVNV)"
COLRMSTATS="colrm 1 8"
else
echo SVN version 1.5 or less "($SVNV)"
COLRMSTATS="colrm 1 7"
fi

#exit 0

cd $SOURCE

echo
echo ========== Clean Source Directory ==========
echo

$SVN revert -R .
$SVN status --no-ignore | grep '^\(I\|?\)' | $COLRMSTATS | xargs -d '\n' rm -rf

echo
echo ========== Update Source Directory to $SVN_REVB ==========
echo

$SVN update -r $SVN_REVB
$SVN status --no-ignore | grep '^\(I\|?\)' | $COLRMSTATS | xargs -d '\n' rm -rf

echo
echo ========== Filter Source Directory ==========
echo

$SCRIPTS/filter-all.sh --force

cd $DEST

echo
echo ========== Clean Branch Directory ==========
echo

$SVN revert -R .
$SVN status --no-ignore | grep '^\(I\|?\)' | $COLRMSTATS | xargs -d '\n' rm -rf

echo
echo ========== Update Branch Directory ==========
echo

$SVN update || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
$SVN status --no-ignore | grep '^\(I\|?\)' | $COLRMSTATS | xargs -d '\n' rm -rf

echo
echo ========== Merge r$SVN_REVA:$SVN_REVB, ignoring conflicts ==========
echo

echo $SVN merge --accept theirs-full -r $SVN_REVA:$SVN_REVB $SVN_URL
$SVN merge --accept theirs-full -r $SVN_REVA:$SVN_REVB $SVN_URL || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1

if [ ${SVNV:0:1} -gt 1 -o ${SVNV:2:1} -gt 5 ]; then

echo
echo '========== Deleting tree conflicts (SVN 1.6+) =========='
echo
$SVN status --no-ignore | grep '^\(!\|      C\)' | $COLRMSTATS | xargs -d '\n' $SVN revert -R

fi

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
        $SVN rm --force "$DEST/$1" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
	rm -rf "$DEST/$1"
    elif [ ! -d "$DEST/$1" ]; then
        C_URL=`LC_ALL=C $SVN info $SOURCE/$1 | awk '$1=="URL:" { print $2 }'`
        C_REV=`LC_ACC=C $SVN info $SOURCE/$1 | awk '$1=="Last" && $2=="Changed" && $3=="Rev:" { print $4 }'`
        echo Copy directory $1 '@' $C_REV from $C_URL
        $SVN cp $C_URL'@'$C_REV "$DEST/$1" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
    else
	cd "$SOURCE/$1"
	for f in *; do
	    if [ -f "$f" ]; then
		if [ "${f##*.}" == "bak" ]; then
		    true # ignore backup files
		elif [ "$f" == "private.txt" ]; then
		    true # ignore private.txt files
		elif [ -d "$DEST/$2$f" ]; then
		    echo "ERROR: File $2$f conflicts with existing directory." >&2
		    echo "The directory will be removed, but the file will only be created in a later commit." >&2
		    $SVN rm --force "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
		elif [ ! -f "$DEST/$2$f" ]; then
		    C_URL=`LC_ALL=C $SVN info $SOURCE/$2$f | awk '$1=="URL:" { print $2 }'`
		    C_REV=`LC_ALL=C $SVN info $SOURCE/$2$f | awk '$1=="Last" && $2=="Changed" && $3=="Rev:" { print $4 }'`
		    echo Copy file $2$f '@' $C_REV from $C_URL
		    $SVN cp $C_URL'@'$C_REV "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
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
        $SVN rm --force "$DEST/$1" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
    else
	DEL=`LC_ALL=C svn info "$SOURCE/$1" | grep -c '^Schedule: delete$'`
	if [ $DEL -gt 0 ]; then
            echo Removing directory $1
            $SVN rm --force "$DEST/$1" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
	else
	    cd "$DEST/$1"
	    for f in *; do
		if [ -f "$f" ]; then
		    if [ ! -f "$SOURCE/$2$f" ]; then
			echo Removing file "$DEST/$2$f"
			$SVN rm --force "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
			rm -f "$DEST/$2$f"
		    elif [ "$f" == "private.txt" ]; then
			echo Removing file "$DEST/$2$f"
			$SVN rm --force "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
			rm -f "$DEST/$2$f"
		    else
			DEL=`LC_ALL=C svn info "$SOURCE/$2$f" | grep -c '^Schedule: delete$'`
			if [ $DEL -gt 0 ]; then
			    echo Removing file "$DEST/$2$f"
			    $SVN rm --force "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
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
echo ========== COPY all files content and properties ==========
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
			cp -pf "$SOURCE/$2$f" "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
			$SVN pl "$SOURCE/$2$f" | tail +2 | colrm 1 2 | grep -v 'svn:mergeinfo' > "$TMPD/.source.plist"
			$SVN pl "$DEST/$2$f" | tail +2 | colrm 1 2 > "$TMPD/.dest.plist"
			for p in `cat "$TMPD/.source.plist"`; do
			    $SVN pg $p --strict "$SOURCE/$2$f" > "$TMPD/.source.prop"
			    if grep -q '^'$p'$' "$TMPD/.dest.plist"; then
				$SVN pg $p --strict "$DEST/$2$f" > "$TMPD/.dest.prop"
				if cmp -s "$TMPD/.source.prop" "$TMPD/.dest.prop"; then
				    true
				else
				    echo "Change property $p to $2$f";
				    $SVN ps $p --file "$TMPD/.source.prop" "$DEST/$2$f"
				fi
				rm -f "$TMPD/.dest.prop"
			    else
				echo "Adding property $p to $2$f";
				$SVN ps $p --file "$TMPD/.source.prop" "$DEST/$2$f"
			    fi
			    rm -f "$TMPD/.source.prop"
			done
			for p in `cat "$TMPD/.dest.plist"`; do
			    if grep -q '^'$p'$' "$TMPD/.source.plist"; then
				true
			    else
				echo "Removing property $p to $2$f";
				$SVN pd $p "$DEST/$2$f"
			    fi
			done
			rm -f "$TMPD/.dest.plist" "$TMPD/.source.plist"
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

# Check consistency

diff -qwrU3 -x '.svn' -x '*.bak' $SOURCE $DEST

echo
echo "After verifications, the following commands can be used to commit the merge :"
echo

echo cd $5
echo svn propset sofa:merged-rev $SVN_REVB .
echo svn commit -m '"'"SCRIPT: Merging trunk revisions r$SVN_REVA:$SVN_REVB to /branches/Sofa-nodev"'"'
echo cd -
