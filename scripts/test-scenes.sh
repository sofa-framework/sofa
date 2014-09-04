#!/bin/bash

# Execute runSofa in batch mode on a series of scenes

# Usage: ./test-scenes.sh <build-dir> <scenes>
# 
# where 
# - <build-dir> is... a build directory,
# - <scenes> is either:
#   - a path to a directory containing scenes, relative to the build directory
#   - a file containing a list of paths to scenes, relative to the build
#     directory
#
# If you want to filter out some scenes, the script accepts a list of patterns
# to ignore with -i, as understood by grep -x

set -o nounset # Error when referencing undefined variables
set -o errexit # Exit on error

usage() {
    echo "Usage: ./test-scenes.sh build-directory (scene-directory|scene-list)"
    echo "                        [-i|--ignore-list file)]"
    echo "                        [-n|--iterations number]"
}

# Log file containing the 'exit status' of runSofa for each scene
STATUS_LOG_FILE=test-scenes-status.log
# Log file containing the output of runSofa for each scene
OUTPUT_LOG_FILE=test-scenes-output.log

ITERATIONS=100

while [[ "$#" > 0 ]]
do
    arg="$1"
    shift
    case $arg in
        -h|--help)
            usage; exit 0;;
        -i|--ignore-list)
            IGNORE_LIST="$1"
            if [[ ! -f "$IGNORE_LIST" ]]; then
                echo "$IGNORE_LIST: no such file"; exit 1;
            fi
            shift;;
        -n|--iterations)
            ITERATIONS="$1"
            if [[ ! "$ITERATIONS" =~ ^[0-9]+$ ]]; then
                echo "Invalid argument '$1': the number of iterations must be an integer"; exit 1;
            fi
            shift;;
        -*)
            echo "WHAT DO YOU WANT FROM ME? (unknown option: $arg)"
            usage; exit 1;;
        *)
            if [[ -z ${BUILD_DIR+x} ]]; then
                BUILD_DIR="$arg"
            elif [[ -z ${SCENES+x} ]]; then
                SCENES="$arg"
            else
		echo $arg
                echo "Superfluous argument: $arg"
                usage; exit 1
            fi;;
    esac
done

if [[ -z ${BUILD_DIR+x} || -z ${SCENES+x} ]]; then
    usage; exit 1
elif [[ ! -d "$BUILD_DIR" ]]; then
    echo "$BUILD_DIR: no such directory"; exit 1;
elif [[ ! -f "$SCENES" && ! -d "$BUILD_DIR/$SCENES" ]]; then
    echo "$SCENES: no such file or directory"; exit 1;
fi

run_single_scene() {
    local scene_file=$1
    pushd $BUILD_DIR > /dev/null
    echo -e "[ Output for $scene_file ]\n" >> $OUTPUT_LOG_FILE
    runSofa_cmd="bin/runSofa -g batch -n $ITERATIONS $scene_file &>> $OUTPUT_LOG_FILE"
    bash -c "$runSofa_cmd" 2> /dev/null && status=$? || status=$?
    echo -e "\n[ End of output for $scene_file ]\n\n" >> $OUTPUT_LOG_FILE

    echo "$scene_file $status" >> $STATUS_LOG_FILE
    popd > /dev/null
}

must_ignore() {
    if [[ -z ${IGNORE_LIST+x} ]]; then
        return 1
    else
        local scene=$1
        while read pattern; do
            if echo "$scene" | grep -qx "$pattern"; then
                return 0
            fi
        done < $IGNORE_LIST
        return 1
    fi
}

list_scenes() {
    if [[ -f "$SCENES" ]]; then
        cat "$SCENES"
    else
        find "$BUILD_DIR/$SCENES" -name '*.scn' | sed -e "s:$BUILD_DIR/::"
    fi
}

run_all_scenes() {
    rm -f $BUILD_DIR/$STATUS_LOG_FILE
    rm -f $BUILD_DIR/$OUTPUT_LOG_FILE
    list_scenes | while read scene;
    do
        if must_ignore "$scene" ; then
            echo "Ignoring $scene"
        else
            echo "Running $scene"
            run_single_scene "$scene"
        fi
    done
}

run_all_scenes
