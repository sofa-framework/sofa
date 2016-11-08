#!/bin/bash

# find UNIQUE initData calls 
# detect file, member and description
# find Data in file.h
# add ///< description after the member
#
# WARNING: sed errors are printed in log.txt, search for "sed:"

usage() {
    echo "Usage: generateDataComments.sh [-f|--force] <src-dir>"
    echo "  src-dir: all files in this folder will be affected"
    echo "  [-f|--force]: existing comments will be overwritten"    
}

# init variables
echo "Init..."
SRC_DIR=""
FORCE=false
ADDED_COMMENTS=0

while [[ "$#" > 0 ]]
do
    arg="$1"
    case $arg in
        -h|--help)
            usage; exit 0;;
        -f|--force)
            FORCE=true;;
        *)
            SRC_DIR="$arg";;
    esac
    shift
done

if [[ "$SRC_DIR" == "" ]]; then
    usage; exit 0;
fi




escape-for-sed() {
    #                       \n become space             \ are removed          / become \/                ( become \(                ) become \)
    echo "$( perl -p -e 's/\\n/ /g' $1 | perl -p -e 's/\\//g' | perl -p -e 's/\//\\\//g' | perl -p -e 's/\(/\\\(/g' | perl -p -e 's/\)/\\\)/g' )"
}

one-data-declaration-per-line() {
    local file_header="$1"
    local member="$2"
    
    # CONVERT one line Data declarations to separated lines Data declarations.
    sed -ie 's/^\([	 ]*\)\(Data[	 ]*<.*>[	 ]*\)\('"$member"'\)[	 ]*[,][	 ]*\(.*\)$/\1\2\3;\n\1\2\4/g' "$file_header"
    rm -f "$file_header"e 2> /dev/null # Created by Windows only
}

fix-inline-comment() {
    local file_header="$1"
    local member="$2"
        
    # FIX inline comments: "// DataComment" and "/// DataComment" to "///< DataComment"
    sed -ie 's/^\(.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;[	 ]*\)\/\/\/? \(.*\)$/\1\/\/\/< \2/g' "$file_header"
    rm -f "$file_header"e 2> /dev/null # Created by Windows only
}

add-comment() {
    local file_header="$1"
    local member="$2"
    local comment="$3"
    
    local escaped_comment="$(echo "$comment" | escape-for-sed)"

    # Get the line of member declaration
    # Warning: if two similar member declarations are detected, we only care about the first one
    # TODO: handle the others
    if [ "$FORCE" = true ]; then # PERMISSIVE PATTERN
        line_number="$(grep -n '^.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;' "$file_header" | grep -Eo '^[^:]+' | cut -f1 -d: | head -1)"
    else # STRICT PATTERN
        line_number="$(grep -n '^.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;[	 ]*$' "$file_header" | grep -Eo '^[^:]+' | cut -f1 -d: | head -1)"
    fi
    previous_line_number=$((line_number-1)) # get previous line
    if [[ $previous_line_number < 1 ]]; then
        # Ignore not pattern-matching member declaration
        # if FORCE==false, it means that the declaration is already commented
        return 1
    fi 
    
    # Search if there is a Doxygen comment on previous line
    if sed "${previous_line_number}q;d" "$file_header" | grep -q "^[	 ]*///"; then
        #echo "Comment found above $member"
        # TODO: in FORCE mode we should remove this comment
        return 1
    fi
    
    # Rewrite the member declaration with Doxygen comment at the end.
    if [ "$FORCE" = true ]; then # PERMISSIVE PATTERN
        sed -ie 's/^\(.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;\)/\1 \/\/\/< '"$escaped_comment"'/g' "$file_header"
    else # STRICT PATTERN
        sed -ie 's/^\(.*Data[	 ]*<.*>[	 ]*'"$member"'[	 ]*;\)[	 ]*$/\1 \/\/\/< '"$escaped_comment"'/g' "$file_header"
    fi
    rm -f "$file_header"e 2> /dev/null # Created by Windows only
    
    echo "$file_header:$line_number : $member : $escaped_comment"
    return 0
}





# Count initData calls
count="$(grep -Er "^[^/]*initData[	 ]*\(.*\)[	 ]*\).*$" "$SRC_DIR" | sort | uniq | wc -l)"
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
    file_header="${file%.*}.h"
    
    if [ ! -f "$file_header" ]; then
        continue
    fi
    if echo "$description" | grep -q "^[^A-Za-z0-9_-]*$"; then
        continue
    fi

    one-data-declaration-per-line "$file_header" "$member"
    
    fix-inline-comment "$file_header" "$member"
    
    if add-comment "$file_header" "$member" "$description"; then
        ((ADDED_COMMENTS++))
    else
        continue
    fi
done < <(grep -Er "^[^/]*initData[	 ]*\(.*\)[	 ]*\).*$" "$SRC_DIR" | sort | uniq)

echo "Update done: $ADDED_COMMENTS members commented."
