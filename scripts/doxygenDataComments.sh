#!/bin/bash

# find UNIQUE initData calls
# detect file, member and description
# find Data in file.h
# add ///< description after the member
#
# WARNING: sed errors are printed in log.txt, search for "sed:"

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
INITDATA_CALLS_FILE="initDataCalls.txt"

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




files-to-update() {
    /usr/bin/find "$SRC_DIR" -regex ".*\.\(h\|cpp\|inl\|c\|cu\|h\.in\)$"
}


escape-for-sed() {
    #                       \n become space             \ are removed          / become \/                ( become \(                ) become \)
    echo "$( perl -p -e 's/\\n/ /g' $1 | perl -p -e 's/\\//g' | perl -p -e 's/\//\\\//g' | perl -p -e 's/\(/\\\(/g' | perl -p -e 's/\)/\\\)/g' )"
}

one-data-declaration-per-line() {
    local file_h="$1"
    local member="$2"

    # CONVERT one line Data declarations to separated lines Data declarations.
    if grep -q '^[	 ]*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*,.*$' "$file_h"; then
        sed -ie 's/^\([	 ]*\)\(Data[	 ]*<.*>[	 ]*\)\('"$member"'\)[	 ]*[,][	 ]*\(.*\)$/\1\2\3;\n\1\2\4/g' "$file_h"
        rm -f "$file_h"e 2> /dev/null # Created by Windows only
    fi
}

fix-inline-comment() {
    local file_h="$1"
    local member="$2"

    # FIX inline comments: "// DataComment" and "/// DataComment" to "///< DataComment"
    if grep -q '^.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;[	 ]*///? ?$' "$file_h"; then
        sed -ie 's/^\(.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;[	 ]*\)\/\/\/? ?\(.*\)$/\1\/\/\/< \2/g' "$file_h"
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
        line_number="$(grep -n '^.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;' "$file_h" | grep -Eo '^[^:]+' | cut -f1 -d: | head -1)"
    else # STRICT PATTERN
        line_number="$(grep -n '^.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;[	 ]*$' "$file_h" | grep -Eo '^[^:]+' | cut -f1 -d: | head -1)"
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





# Count initData calls
echo "Counting initData calls..."
rm -f "$INITDATA_CALLS_FILE"
grep -Er "^[^/]*initData[	 ]*\(.*\)[	 ]*\).*$" "$SRC_DIR" | sort | uniq > "$INITDATA_CALLS_FILE"
count="$(wc -l < "$INITDATA_CALLS_FILE")"
echo "$count calls counted."

echo "Starting detection..."
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

    one-data-declaration-per-line "$file_h" "$member"

    fix-inline-comment "$file_h" "$member"

    add-comment "$file_h" "$member" "$description"

done < "$INITDATA_CALLS_FILE"

echo "Update done."
rm -f "$INITDATA_CALLS_FILE"
