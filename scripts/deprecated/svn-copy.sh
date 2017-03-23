#!/bin/bash

SVN="svn --non-interactive"

if [ $# -lt 2 ]; then
echo USAGE: $0 source_dir dest_dir [tmp_dir]
echo Specify tmp_dir if you also want to duplicate svn properties
exit 1
fi
echo "WARNING: This script will delete part of the files in the $2 directory."
echo "Make sure that you don't have any local changes not copied somewhere else."
echo "Press Enter to continue, or Ctrl-C to cancel."
read || exit 1

shopt -s nullglob

DIR0=$PWD
cd ${0%/*} || exit 1
SCRIPTS=$PWD
echo "SCRIPTS:" $SCRIPTS
cd -
cd $1 || exit 1
SOURCE=$PWD
echo "SOURCE:" $SOURCE
cd -
cd $2 || exit 1
DEST=$PWD
echo "DEST:" $DEST
cd -
TMPD=$PWD
if [ $# -gt 2 ]; then
cd $3 || exit 1
TMPD=$PWD
cd -
fi

#exit 0

echo
echo ========== Copy all missing directories and files ==========
echo

function svncopy_process_dir {
    CUR="$1"
    if [ -f "$DEST/$1" ]; then
        echo "ERROR: Directory $1 conflicts with existing file." >&2
        echo "The file will be removed, but the directory will only be created in a later commit." >&2
        $SVN rm --force "$DEST/$1" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
	rm -rf "$DEST/$1"
        echo Copy directory $1
        cp -prf "$SOURCE/$1" "$DEST/$1" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
    elif [ ! -d "$DEST/$1" ]; then
        echo Copy and SVN add directory $1
        cp -prf "$SOURCE/$1" "$DEST/$1" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
	find "$DEST/$1" -name .svn -prune -exec rm -rf '{}' ';'
        $SVN add "$DEST/$1" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
    else
	cd "$SOURCE/$1"
	for f in *; do
	    if [ -f "$f" ]; then
		if [ "${f##*.}" == "bak" ]; then
		    true # ignore backup files
		elif [ -d "$DEST/$2$f" ]; then
		    echo "ERROR: File $2$f conflicts with existing directory." >&2
		    echo "The directory will be removed, but the file will only be created in a later commit." >&2
		    $SVN rm --force "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
		elif [ ! -f "$DEST/$2$f" ]; then
		    echo Copy and SVN add file $2$f
		    cp -pf  "$SOURCE/$2$f" "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
		    $SVN add "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
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
	cd "$DEST/$1"
	for f in *; do
	    if [ -f "$f" ]; then
		if [ ! -f "$SOURCE/$2$f" ]; then
		    echo Removing file "$DEST/$2$f"
		    $SVN rm --force "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
		    rm -f "$DEST/$2$f"
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
}

svnrm_process_dir "" ""

echo
echo ========== COPY all files content ==========
echo

function cp_process_dir {
    CUR="$1"
    if [ -d "$SOURCE/$1" ]; then
	cd "$DEST/$1"
	for f in *; do
	    if [ -f "$f" ]; then
		if [ -f "$SOURCE/$2$f" ]; then
		    cp -pf "$SOURCE/$2$f" "$DEST/$2$f" || read -p "Press Enter to continue, or Ctrl-C to cancel." || exit 1
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
}

cp_process_dir "" ""

if [ $# -gt 2 ]; then

echo
echo ========== COPY all files properties ==========
echo

function props_process_dir {
    CUR="$1"
    if [ -d "$SOURCE/$1" ]; then
	cd "$DEST/$1"
	for f in *; do
	    if [ -f "$f" ]; then
		if [ -f "$SOURCE/$2$f" ]; then
		    $SVN pl "$SOURCE/$2$f" | tail +2 | colrm 1 2 > "$TMPD/.source.plist"
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
		props_process_dir "$2$f" "$2$f/"
	    fi
	done
	cd ..
    fi
}

props_process_dir "" ""

fi

echo
echo ========== COPY complete ==========
echo

# Check concistency

diff -qwrU3 -x '.svn' -x '*.bak' $SOURCE $DEST
