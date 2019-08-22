#!/bin/bash

# find UNIQUE initData calls
# detect file, member and description
# find Data in file.h
# add ///< description after the member

usage() {
    echo "Usage: doxygenDataComments.sh [-f|--force] <src-dir>"
    echo "  src-dir: all files in this folder will be affected"
    echo "  [-f|--force]: existing comments will be overwritten"
}

# init variables
echo "Init..."
SRC_DIR=""
FORCE=false
ADDED_COMMENTS=0
TMP_FILE="doxygenDataComments.tmp.txt"

while [[ "$#" > 0 ]]; do
    case $1 in
        -h|--help)
            usage; exit 0;;
        -f|--force)
            FORCE=true;;
        *)
            SRC_DIR="$(cd "$1" && pwd)";;
    esac
    shift
done

if [[ "$SRC_DIR" == "" ]]; then
    usage; exit 0;
fi



force-one-lined-data-declarations() {
    rm -f "$TMP_FILE"
    local grep_pattern='^[	 A-Za-z:_-]*Data[	 ]*<.*>[	 ]*[A-Za-z_-]+[	 ]*,[	 ]*.*;.*$'
    grep -Er --include \*.h "$grep_pattern" "$SRC_DIR" | sort | uniq > "$TMP_FILE"
    count="$(wc -l < "$TMP_FILE")"

    i=1
    while read -r line; do
        (>&2 echo -ne "Updating: $i / $count\r")
        ((i++))

        file="$(echo "$line" | sed -e 's/:.*$//')"

        if [ ! -f "$file" ]; then
            continue
        fi

        # CONVERT multiple Data declaration to one-lined Data declarations.
        while grep -Eq "$grep_pattern" "$file"; do
            sed -ie 's/^\(.*Data[	 ]*<.*>[	 ]*\)\([A-Za-z_-]*\)[	 ]*,[	 ]*\(.*\);\(.*\)$/\1\2;\4\n\1\3;\4/g' "$file"
            rm -f "$file"e 2> /dev/null # Created by Windows only
        done
    done < "$TMP_FILE"
    
    rm -f "$TMP_FILE"
}

escape-for-sed() {
    #                       \n become space             \ are removed          / become \/                ( become \(                ) become \)
    echo "$( perl -p -e 's/\\n/ /g' $1 | perl -p -e 's/\\//g' | perl -p -e 's/\//\\\//g' | perl -p -e 's/\(/\\\(/g' | perl -p -e 's/\)/\\\)/g' )"
}

fix-inline-comment() {
    local file_h="$1"
    local member="$2"

    # FIX inline comments: "// DataComment" and "/// DataComment" to "///< DataComment"
    if grep -q '^[	 A-Za-z:_-]*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;[	 ]*///*[	 ][	 ]*' "$file_h"; then
        sed -ie 's/^\([	 A-Za-z:_-]*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;[	 ]*\)\/\/\/*[	 ][	 ]*\(.*\)$/\1\/\/\/< \2/g' "$file_h"
        rm -f "$file_h"e 2> /dev/null # Created by Windows only
    fi
}

add-comment() {
    local file_h="$1"
    local member="$2"
    local comment="$3"

    # Get the line of member declaration
    # Warning: if two similar member declarations are detected, we only care about the first one
    # TODO: handle the others
    if [[ "$FORCE" == "true" ]]; then # PERMISSIVE PATTERN
        line_number="$(grep -n '^[	 A-Za-z:_-]*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;' "$file_h" | grep -Eo '^[^:]+' | cut -f1 -d: | head -1)"
    else # STRICT PATTERN
        line_number="$(grep -n '^[	 A-Za-z:_-]*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;[	 ]*$' "$file_h" | grep -Eo '^[^:]+' | cut -f1 -d: | head -1)"
    fi
    previous_line_number=$((line_number-1)) # get previous line
    if [[ $previous_line_number < 1 ]]; then
        # Ignore not pattern-matching member declaration
        # if FORCE==false, it means that the declaration is already commented
        return 1
    fi

    # Search if there is a Doxygen comment on previous line
    if sed "${previous_line_number}q;d" "$file_h" | grep -q "^[	 ]*///"; then
        #echo "Comment found above $member"
        # TODO: in FORCE mode we should remove this comment
        return 1
    fi

    # Rewrite the member declaration with Doxygen comment at the end.
    local escaped_comment="$(echo "$comment" | escape-for-sed)"
    if [[ "$FORCE" == "true" ]]; then # PERMISSIVE PATTERN
        sed -ie 's/^\(.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;\)/\1 \/\/\/< '"$escaped_comment"'/g' "$file_h"
    else # STRICT PATTERN
        sed -ie 's/^\(.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;\)[	 ]*$/\1 \/\/\/< '"$escaped_comment"'/g' "$file_h"
    fi
    rm -f "$file_h"e 2> /dev/null # Created by Windows only

    echo "$file_h:$line_number: $member: $escaped_comment"
    return 0
}

generate-doxygen-data-comments() {
    rm -f "$TMP_FILE"
    # Count initData calls
    echo "Counting initData calls..."
    grep -Er --include \*.h --include \*.inl --include \*.cpp "^[^/]*initData[	 ]*\(.*\)[	 ]*\).*$" "$SRC_DIR" | sort | uniq > "$TMP_FILE"
    count="$(wc -l < "$TMP_FILE")"
    echo "$count calls counted."

    # For each initData call that is on one line
    i=1
    while read -r line; do
        (>&2 echo -ne "Updating: $i / $count\r")
        ((i++))

        file="$(echo "$line" | sed -e 's/:.*$//')"
        call="$(echo "$line" | sed -e 's/^[^:]*:[[:space:],:]*//')"
        member="$(echo "$call" | sed 's/ *(.*//')"
        description="$(echo "$call" | sed 's/.*, *"//' | sed 's/\([^\\]\)".*/\1/')"
        file_h="${file%.*}.h"

        if [ ! -f "$file_h" ]; then
            continue
        fi
        if echo "$description" | grep -q "^[^A-Za-z0-9_-]*$"; then
            continue
        fi

        fix-inline-comment "$file_h" "$member"

        add-comment "$file_h" "$member" "$description"
    done < "$TMP_FILE"

    echo ""
    rm -f "$TMP_FILE"
}


echo "Force one-lined data declarations..."
force-one-lined-data-declarations
echo "Done."

echo "Generate Doxygen Data comments..."
generate-doxygen-data-comments
echo "Done."
