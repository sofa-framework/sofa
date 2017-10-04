#!/bin/bash

usage() {
    echo "Usage: update-scenes-rayleighcoeff.sh <src-dir>"
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

get_rayleigh() {
    file="$1"
    local found="0";

    if grep -q "<EulerImplicit" "$file"; then
        solver="EulerImplicit";
        found="1";
    elif grep -q "<VariationalSymplecticSolver" "$file"; then
        solver="VariationalSymplecticSolver";
        found="1";
    elif grep -q "<NewmarkImplicitSolver" "$file"; then
        solver="NewmarkImplicitSolver";
        found="1";
    elif grep -q "<CentralDifferenceSolver" "$file"; then
        solver="CentralDifferenceSolver";
        found="1";
        isstiffness="1";
    fi

    if [[ $found == "1" ]]; then

        if grep -q "rayleighMass" "$file"; then
            ismass="1";
        fi
        if grep -q "rayleighStiffness" "$file"; then
            isstiffness="1";
        fi
    else
        solver="none";
    fi
}


update_rayleigh() {
    file="$1"

    if [[ $ismass == "1" && $isstiffness == "0" ]]; then
        perl -0777 -i -pe "s/($solver.*?)\/>/\1 rayleighStiffness=\"0.1\" \/>/s" "$file"
        echo "$file updated : rayleighStiffness added"
    elif [[ $ismass == "0" && $isstiffness == "1" ]]; then
        perl -0777 -i -pe "s/($solver.*?)\/>/\1 rayleighMass=\"0.1\" \/>/s" "$file"
        echo "$file updated : rayleighMass added"
    elif [[ $ismass == "0" && $isstiffness == "0" ]]; then
        perl -0777 -i -pe "s/($solver.*?)\/>/\1 rayleighStiffness=\"0.1\" rayleighMass=\"0.1\" \/>/s" "$file"
        echo "$file updated : rayleighStiffness and rayleighMass added"
    else
        echo "********** This case shouldn't occur **********"
    fi

    rm -f "$file.bak" 2> /dev/null # Created by Windows only
}

main() {
    local files_count="$(files_to_update | wc -l)"
    local i=0

    export filesmodified=0
    export solver="none"    
    export solver="none"
    export ismass="0"

    files_to_update | while read file; do
        i=$((i+1))

        if [ ! -e "$file" ]; then
            echo "$file: file not found."
            continue
        fi

        ismass="0"
        isstiffness="0"


        get_rayleigh "$file"


        if [[ $solver == "none" ]]; then
            echo "$file : nothing to do"
            if [ $i == $files_count ]; then
                (>&2 echo -ne "total : $filesmodified file(s) modified")
            fi
            continue

        elif [[  $ismass == "1" && $isstiffness == "1" ]]; then
            echo "$file : nothing to do"
            if [ $i == $files_count ]; then
                (>&2 echo -ne "total : $filesmodified file(s) modified")
            fi
            continue

        else
            update_rayleigh "$file"

            if [ $i == $files_count ]; then
                (>&2 echo -ne "total : $filesmodified file(s) modified")
            fi
            filesmodified=$((filesmodified+1))

        fi

    done

    echo ""
    unset filesmodified
    unset solver
    unset ismass
    unset isstiffness
}

# Start script
main