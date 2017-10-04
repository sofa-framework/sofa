#!/bin/bash

usage() {
    echo "Usage: licenseUpdater.sh <src-dir> <license> <year> <version>"
    echo "  src-dir: all files in this folder will be affected"
    echo "  license: choose in ['auto', 'LGPL', 'GPL']"
    echo "  year: YYYY formatted year for the copyright"
    echo "  version: SOFA version"
}

if [[ "$#" = 4 ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC_DIR="$(cd "$1" && pwd)"
    LICENSE="$2"
    YEAR="$3"
    VERSION="$4"

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

get-license() {
    file="$1"
    if grep -q "*       SOFA, Simulation Open-Framework Architecture," "$file"; then
        if grep -q "GNU Lesser General Public License" "$file" && grep -q "GNU General Public License" "$file"; then
            echo "multiple";
        elif grep -q "GNU Lesser General Public License" "$file"; then
            echo "LGPL";
        elif grep -q "GNU General Public License" "$file"; then
            echo "GPL";
        else
            echo "other";
        fi
    else
        echo "none";
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

# prepare-header <file>
prepare-header() {
    if [ ! -e "$1" ]; then
        echo "$1: file not found."; exit 1
    fi
    set-year "$1" | set-version
}

escape-for-perl() {
    #                   * becomes \*                  / becomes \/               @ becomes \@                ( becomes \(               ) becomes \)
    echo "$(perl -p -e 's/\*/\\\*/g' $1 | perl -p -e 's/\//\\\//g' | perl -p -e 's/\@/\\\@/g' | perl -p -e 's/\(/\\\(/g' | perl -p -e 's/\)/\\\)/g')"
}

update-header() {
    header="$1"
    file="$2"
    escaped_header="$(echo "$header" | escape-for-perl)"
    # search for /***(78)***...is free software...***(78)***/ and replace with escaped header
    perl -0777 -i -pe "s/(\/)(\*){78}(.*?)is free software(.*?)(\*){78}(\/)/$escaped_header/s" "$file"
    rm -f "$file.bak" 2> /dev/null # Created by Windows only
}

main() {
    if [ "$LICENSE" == "auto" ]; then
        if [ ! -e "$SCRIPT_DIR/LGPL_header.template" ] || [ ! -e "$SCRIPT_DIR/GPL_header.template" ]; then
            echo "ERROR: missing LGPL_header.template and/or GPL_header.template in $SCRIPT_DIR"; exit 1
        fi
        LGPL_HEADER="$(prepare-header $SCRIPT_DIR/LGPL_header.template)"
        GPL_HEADER="$(prepare-header $SCRIPT_DIR/GPL_header.template)"
    else
        file="${SCRIPT_DIR}/${LICENSE}_header.template"
        if [ ! -e "$file" ]; then
            echo "ERROR: missing $file in $SCRIPT_DIR"; exit 1
        fi
        LICENSE_HEADER="$(prepare-header "$file")"
    fi

    local files_count="$(files-to-update | wc -l)"
    local i=1

    files-to-update | while read file; do
        if [ ! -e "$file" ]; then
            echo "$file: file not found."
            continue
        fi

        current_license="$(get-license "$file")"

        if [ "$LICENSE" == "auto" ]; then
            case "$current_license" in
                "LGPL")
                    update-header "$LGPL_HEADER" "$file"
                    echo "$file updated with LGPL"
                    ;;
                "GPL")
                    update-header "$GPL_HEADER" "$file"
                    echo "$file updated with GPL"
                    ;;
                *)
                    echo "WARNING: $file not changed. Licence detected: $current_license"
                    ;;
            esac
        else # [ $LICENSE != "auto" ]
            case "$current_license" in
                "none")
                    # add new license header
                    file_before=`cat "$file"`
                    echo "${LICENSE_HEADER}" > "$file"
                    echo "${file_before}" >> "$file"
                    # echo "$file set to $LICENSE"
                    ;;
                "multiple")
                    echo "WARNING: $file not changed. Licence detected: multiple"
                    ;;
                *)
                    update-header "$LICENSE_HEADER" "$file"
                    echo "$file updated with $LICENSE"
                    ;;
            esac
        fi        
        (>&2 echo -ne "Updating: $i / $files_count\r")
        ((i++))
    done
}

# Start script
main