#!/bin/bash

usage() {
    echo "Usage: update-mass-data.sh <src-dir>"
    echo "  src-dir: all scenes in this folder will be affected"
}

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters ($#, should be one: <src-dir>)"
    usage; exit 1
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC_DIR="$(cd "$1" && pwd)"
fi


if [ ! -d "$SRC_DIR" ]; then
    echo "Wrong src-dir";
    usage; exit 1
fi

files_to_update() {
    /usr/bin/find "$SRC_DIR" -regex ".*\.\(scn\|xml\)$"
}



update_mass_data() {
    file="$1"
    
    sed -i -e "s/^\([     ]*<[     ]*UniformMass.*[     ]\)mass\([     ]*=.*\)$/\1vertexMass\2/g" "$file"
    sed -i -e "s/^\([     ]*<[     ]*UniformMass.*[     ]\)totalmass\([     ]*=.*\)$/\1totalMass\2/g" "$file"
    sed -i -e "s/^\([     ]*<[     ]*DiagonalMass.*[     ]\)mass\([     ]*=.*\)$/\1vertexMass\2/g" "$file"


    rm -f "$file.bak" 2> /dev/null # Created by Windows only
}

main() {
    local files_count="$(files_to_update | wc -l)"
    local i=0

    export filesmodified=0

    files_to_update | while read file; do
        i=$((i+1))

        if [ ! -e "$file" ]; then
            echo "$file: file not found."
            continue
        fi
        
        update_mass_data "$file"
        filesmodified=$((filesmodified+1))
    done

    echo ""
    unset filesmodified
}

# Start script
main