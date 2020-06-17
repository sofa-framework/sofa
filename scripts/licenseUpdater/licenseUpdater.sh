#!/bin/bash

usage() {
    echo "Usage: licenseUpdater.sh <src-dir> <license>"
    echo "  src-dir: all files in this folder will be affected"
    echo "  license: choose in ['auto', 'LGPL', 'GPL']"
}

if [[ "$#" = 2 ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC_DIR="$(cd "$1" && pwd)"
    LICENSE="$2"
else
    usage; exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
    usage; exit 1
fi

files-to-update() {
    /usr/bin/find "$SRC_DIR" -regex ".*\.\(h\|cpp\|inl\|c\|cu\|h\.in\)$"
}

get-license() {
    file="$1"
    if grep -q -E "\*[ ]+SOFA, Simulation Open-Framework Architecture" "$file"; then
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
        LGPL_HEADER="$(cat $SCRIPT_DIR/LGPL_header.template)"
        GPL_HEADER="$(cat $SCRIPT_DIR/GPL_header.template)"
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
                    echo "WARNING: $file not changed. License detected: $current_license"
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
                    echo "WARNING: $file not changed. License detected: multiple"
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