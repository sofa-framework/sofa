#!/bin/bash

usage() {
    echo "Usage: licenceUpdater.sh <src-dir> <year> <version>"
}

if [[ "$#" = 3 ]]; then
    SRC_DIR="$1"
    YEAR="$2"
    VERSION="$3"

    VERSION_TAG="____VERSION_NUMBER____"
    YEAR_TAG="YYYY"
else
    usage; exit 1
fi

if [ ${#VERSION} -gt ${#VERSION_TAG} ]; then
    echo "ERROR: <version> length can be ${#VERSION_TAG} chars max."; exit 1
elif [ ${#YEAR} -gt ${#YEAR_TAG} ]; then
    echo "ERROR: <year> length can be ${#YEAR_TAG} chars max."; exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
    usage; exit 1
fi

files-to-update() {
    /usr/bin/find "$SRC_DIR" -regex ".*\.\(h\|cpp\|inl\|c\|cu\|h\.in\)$"
}

get-licence() {
    local LGPL_count="$(grep -R "GNU Lesser General Public License" $1 | wc -l)"
    local GPL_count="$(grep -R "GNU General Public License" $1 | wc -l)"

    if [ "$LGPL_count" -gt 0 ] && [ "$GPL_count" -gt 0 ]; then
        echo "both";
    elif [ "$LGPL_count" -gt 0 ]; then
        echo "LGPL";
    elif [ "$GPL_count" -gt 0 ]; then
        echo "GPL";
    else
        echo "other";
    fi
}

set-year() {
    echo "$(perl -p -e "s/$YEAR_TAG/$YEAR/g" $1)"
}

set-version() {
    local tag_size=${#VERSION_TAG}
    local version="$VERSION"
    local version_size=${#version}

    while [ $version_size -lt $tag_size ]; do
        version+=' '
        version_size=${#version}
    done

    echo "$(perl -p -e "s/$VERSION_TAG/$version/g" $1)"
}

escape-for-perl() {
    #                   * becomes \*                  / becomes \/               @ becomes \@                ( becomes \(               ) becomes \)
    echo "$(perl -p -e 's/\*/\\\*/g' $1 | perl -p -e 's/\//\\\//g' | perl -p -e 's/\@/\\\@/g' | perl -p -e 's/\(/\\\(/g' | perl -p -e 's/\)/\\\)/g')"
}

# prepare_header <file>
prepare_header() {
    set-year "$1" | set-version | escape-for-perl
}

main() {
    LGPL_HEADER="$(prepare_header ./LGPL_header.template)"
    GPL_HEADER="$(prepare_header ./GPL_header.template)"

    local files_count="$(files-to-update | wc -l)"
    local i=1

    files-to-update | while read file; do
        if [ ! -e "$file" ]; then
            continue
        fi

        local licence="$(get-licence $file)"
        case "$licence" in
            "LGPL")
                # search for /***(78)***...is free software...***(78)***/ and replace with escaped header
                perl -0777 -i -pe "s/\/(\*){78}(.*)is free software(.*)(\*){78}\//$LGPL_HEADER/s" "$file" && rm "$file.bak"

                echo "$file updated with LGPL"
                ;;
            "GPL")
                # search for /***(78)***...is free software...***(78)***/ and replace with escaped header
                perl -0777 -i -pe "s/\/(\*){78}(.*)is free software(.*)(\*){78}\//$GPL_HEADER/s" "$file" && rm "$file.bak"

                echo "$file updated with GPL"
                ;;
            "both")
                echo "WARNING: $file not changed: both licences detected"
                ;;
            "other")
                echo "WARNING: $file not changed: no licence detected"
                ;;
        esac

        (>&2 echo -ne "Updating: $i / $files_count\r")
        ((i++))
    done
}

# Start script
main