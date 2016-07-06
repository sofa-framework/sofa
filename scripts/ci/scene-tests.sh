#! /bin/bash

# This scripts tries to run runSofa in batch mode on each .scn file in the
# repository, and saves the results in a bunch of files.
#
# More precisely, it deals with .scn files under the examples/ at the root of
# the source tree, and the .scn files found in the examples/ directory of each
# plugin that was compiled.
#
# The default behaviour it to run 100 iterations for each scene, with a timeout
# of 30 seconds.  This can be influenced via a .scene-tests put directly in one
# of the searched directories, and that contains directives like those:
#
# ignore "path/to/file.scn"
# add "path/to/file.scn"
# timeout "path/to/file.scn" "number-of-seconds"
# iterations "path/to/file.scn" "number-of-iterations"

set -o errexit

usage() {
    echo "Usage: scene-tests.sh [run|count-warnings|count-errors|print-summary] <build-dir> <src-dir>"
}

if [[ "$#" = 3 ]]; then
    command="$1"
    build_dir="$2"
    src_dir="$3"
    output_dir="$build_dir/scene-testing"
else
    usage; exit 1
fi

if [[ ! -d "$build_dir/lib/" ]]; then
    echo "Error: '$build_dir' does not look like a Sofa build."
    usage; exit 1
elif [[ ! -d "$src_dir/applications/plugins" ]]; then
    echo "Error: '$src_dir' does not look like a Sofa source tree."
    usage; exit 1
fi


### Utils

filter-out-comments() {
    sed -e 's/#.*//'
}
remove-leading-blanks() {
    sed -e 's/^[[:blank:]]*//'
}
remove-trailing-blanks() {
    sed -e 's/[[:blank:]]*$//'
}
delete-blank-lines() {
    sed -e '/^$/d'
}
clean-line() {
    filter-out-comments | remove-leading-blanks | remove-trailing-blanks | delete-blank-lines
}
log() {
    # # Send to stderr not to interfere
    tee -a "$output_dir/log.txt" 1>&2
    # cat >> "$output_dir/log.txt"
}

# Well-formed option: 'option "arg 1" "arg 2" "arg 3"'
option-is-well-formed() {
    local cmd='[^[:blank:]]*'
    local arg='"[^"]*"'
    echo "$1" | grep -xqE "^$cmd([[:blank:]]*$arg)+"
}
# $ option-split-args '"a b c" "d" "e f"'
# a b c
# d
# e f
option-split-args() {
    local line="$1"
    local rest="$line"
    while [[ "$rest" != "" ]]; do
        local format='^"\([^"]*\)"[[:blank:]]*\(.*\)'
        local arg=$(echo "$rest" | sed "s/$format/\1/")
        local rest=$(echo "$rest" | sed "s/$format/\2/")
        if [[ "$arg" == "" ]]; then
            # (This should never happen.)
            echo "Warning: error parsing arguments: $line" 1>&2
        fi
        echo "$arg"
    done
}
# get-args 'foo "a" "b" "c"'
# "a" "b" "c"
get-args() {
    echo "$1" | sed -e 's/[^[:blank:]][^[:blank:]]*[[:blank:]][[:blank:]]*//'
}
# get-option 'foo "a" "b" "c"'
# foo
get-option() {
    echo "$1" | sed -e 's/\([^[:blank:]][^[:blank:]]*\).*/\1/'
}
# $ get-arg '"a" "b" "c"' 2
# b
get-arg() {
    echo "$1" | option-split-args "$1" | sed -n "$2p"
}
# $ count-args '"a" "b" "c"'
# 3
count-args() {
    option-split-args "$1" | wc -l | tr -d ' 	'
}

list-scenes() {
    local directory="$1"
    /usr/bin/find "$directory" -name '*.scn' | sed -e "s:$directory/::"
}


get-lib() {
    pushd "$build_dir/lib/" > /dev/null
    ls {lib,}"$1".{dylib,so,lib}* 2> /dev/null | xargs echo
    popd > /dev/null
}

list-plugins() {
    pushd "$src_dir/applications/plugins" > /dev/null
    for plugin in *; do
        if [ -e "$plugin/CMakeLists.txt" ]; then
            echo "$plugin"
        fi
    done
    popd > /dev/null
}

list-scene-directories() {
    # Main directory
    mkdir -p "$output_dir/examples"
    echo examples >> "$output_dir/directories.txt"
    # List directories for compiled plugins only
    list-plugins | while read plugin; do
        local lib="$(get-lib "$plugin")"
        if [ -n "$lib" ]; then
            echo "Plugin $plugin: built (found $lib)" | log
            if [ -d "$src_dir/applications/plugins/$plugin/examples" ]; then
                mkdir -p "$output_dir/applications/plugins/$plugin/examples"
                echo "applications/plugins/$plugin/examples"
            else
                echo "Plugin $plugin: no examples/ directory" | log
            fi
        else
            echo "Plugin $plugin: not built" | log
        fi
    done >> "$output_dir/directories.txt"
}

create-directories() {
    # List directories where scenes will be tested
    list-scene-directories

    # echo "Creating directory structure."
    # List all scenes
    while read path; do
        rm -f "$output_dir/$path/ignore-patterns.txt"
        touch "$output_dir/$path/ignore-patterns.txt"
        rm -f "$output_dir/$path/add-patterns.txt"
        touch "$output_dir/$path/add-patterns.txt"
        list-scenes "$src_dir/$path" > "$output_dir/$path/scenes.txt"
        while read scene; do
            mkdir -p "$output_dir/$path/$scene"
            echo 30 > "$output_dir/$path/$scene/timeout.txt" # Default timeout, in seconds
            echo 100 > "$output_dir/$path/$scene/iterations.txt" # Default number of iterations
            echo "$path/$scene" >> "$output_dir/all-scenes.txt"
        done < "$output_dir/$path/scenes.txt"
    done < "$output_dir/directories.txt"
}

parse-options-files() {
    # echo "Parsing option files."
    while read path; do
        if [[ -e "$src_dir/$path/.scene-tests" ]]; then
            clean-line < "$src_dir/$path/.scene-tests" | while read line; do
                if option-is-well-formed "$line"; then
                    local option=$(get-option "$line")
                    local args=$(get-args "$line")
                    case "$option" in
                        ignore)
                            if [[ "$(count-args "$args")" = 1 ]]; then
                                get-arg "$args" 1 >> "$output_dir/$path/ignore-patterns.txt"
                            else
                                echo "$path/.scene-tests: warning: 'ignore' expects one argument: ignore <pattern>" | log
                            fi
                            ;;
			add)
                            if [[ "$(count-args "$args")" = 1 ]]; then
                                scene="$(get-arg "$args" 1)"
                                echo $scene >> "$output_dir/$path/add-patterns.txt"
                                mkdir -p "$output_dir/$path/$scene"
                                echo 30 > "$output_dir/$path/$scene/timeout.txt" # Default timeout, in seconds
                                echo 100 > "$output_dir/$path/$scene/iterations.txt" # Default number of iterations
                            else
                                echo "$path/.scene-tests: warning: 'add' expects one argument: add <pattern>" | log
                            fi
                            ;;
                        timeout)
                            if [[ "$(count-args "$args")" = 2 ]]; then
                                scene="$(get-arg "$args" 1)"
                                if [[ -e "$src_dir/$path/$scene" ]]; then
                                    get-arg "$args" 2 > "$output_dir/$path/$scene/timeout.txt"
                                else
                                    echo "$path/.scene-tests: warning: no such file: $scene" | log
                                fi
                            else
                                echo "$path/.scene-tests: warning: 'timeout' expects two arguments: timeout <file> <timeout>" | log
                            fi
                            ;;
                        iterations)
                            if [[ "$(count-args "$args")" = 2 ]]; then
                                scene="$(get-arg "$args" 1)"
                                if [[ -e "$src_dir/$path/$scene" ]]; then
                                    get-arg "$args" 2 > "$output_dir/$path/$scene/iterations.txt"
                                else
                                    echo "$path/.scene-tests: warning: no such file: $scene" | log
                                fi
                            else
                                echo "$path/.scene-tests: warning: 'iterations' expects two arguments: iterations <file> <number>" | log
                            fi
                            ;;
                        *)
                            echo "$path/.scene-tests: warning: unknown option: $option" | log
                            ;;
                    esac
                else
                    echo "$path/.scene-tests: warning: ill-formed line: $line" | log
                fi
            done
        fi
    done < "$output_dir/directories.txt"

    # echo "Listing ignored and added scenes."
    while read path; do
        grep -xf "$output_dir/$path/ignore-patterns.txt" \
            "$output_dir/$path/scenes.txt" \
            > "$output_dir/$path/ignored-scenes.txt" || true
        if [ -s "$output_dir/$path/ignore-patterns.txt" ]; then
            grep -xvf "$output_dir/$path/ignore-patterns.txt" \
                "$output_dir/$path/scenes.txt" \
                > "$output_dir/$path/tested-scenes.txt" || true
        else
            cp  "$output_dir/$path/scenes.txt" "$output_dir/$path/tested-scenes.txt"
        fi

        sed -e "s:^:$path/:" "$output_dir/$path/ignored-scenes.txt" >> "$output_dir/all-ignored-scenes.txt"

        # Add scenes
        cp "$output_dir/$path/add-patterns.txt" "$output_dir/$path/added-scenes.txt"
        if [ -s "$output_dir/$path/add-patterns.txt" ]; then
            cat "$output_dir/$path/add-patterns.txt" \
                >> "$output_dir/$path/tested-scenes.txt" || true
            cat "$output_dir/$path/add-patterns.txt" \
                >> "$output_dir/$path/scenes.txt" || true
        fi

        sed -e "s:^:$path/:" "$output_dir/$path/added-scenes.txt" >> "$output_dir/all-added-scenes.txt"
        sed -e "s:^:$path/:" "$output_dir/$path/tested-scenes.txt" >> "$output_dir/all-tested-scenes.txt"
    done < "$output_dir/directories.txt"
}

initialize-scene-testing() {
    echo "Initializing scene testing."
    rm -rf "$output_dir"
    mkdir -p "$output_dir"

    runSofa="$(ls "$build_dir/bin/runSofa"{,d} 2> /dev/null || true)"
    if [[ -x "$runSofa" ]]; then
        echo "Found runSofa: $runSofa" | log
    else
        echo "Error: could not find runSofa."
        exit 1
    fi

    touch "$output_dir/warnings.txt"
    touch "$output_dir/errors.txt"

    create-directories
    parse-options-files
}

test-all-scenes() {
    echo "Scene testing in progress..."
    while read scene; do
        echo "- $scene"
        local iterations=$(cat "$output_dir/$scene/iterations.txt")
        local options="-g batch -s dag -n $iterations"
        local runSofa_cmd="$runSofa $options $src_dir/$scene >> $output_dir/$scene/output.txt 2>&1"
        local timeout=$(cat "$output_dir/$scene/timeout.txt")
        echo "$runSofa_cmd" > "$output_dir/$scene/command.txt"
        "$src_dir/scripts/ci/timeout.sh" runSofa "$runSofa_cmd" $timeout
        local status=-1
        if [[ -e runSofa.timeout ]]; then
            echo 'Timeout!'
            echo timeout > "$output_dir/$scene/status.txt"
            rm -f runSofa.timeout
        else
            cat runSofa.exit_code > "$output_dir/$scene/status.txt"
        fi
        rm -f runSofa.exit_code
    done < "$output_dir/all-tested-scenes.txt"
    echo "Done."
}

extract-warnings() {
    while read scene; do
        if [[ -e "$output_dir/$scene/output.txt" ]]; then
            sed -ne "/^\[WARNING\] [^]]*/s:\([^]]*\):$scene\: \1:p \
                " "$output_dir/$scene/output.txt"
        fi
    done < "$output_dir/all-tested-scenes.txt" > "$output_dir/warnings.txt"
}

extract-errors() {
    while read scene; do
        if [[ -e "$output_dir/$scene/output.txt" ]]; then
            sed -ne "/^\[ERROR\] [^]]*/s:\([^]]*\):$scene\: \1:p \
                " "$output_dir/$scene/output.txt"
        fi
    done < "$output_dir/all-tested-scenes.txt" > "$output_dir/errors.txt"
}

extract-crashes() {
    while read scene; do
        if [[ -e "$output_dir/$scene/status.txt" ]]; then
            local status="$(cat "$output_dir/$scene/status.txt")"
            if [[ "$status" != 0 ]]; then
                echo "$scene: error: $status"
            fi
        fi
    done < "$output_dir/all-tested-scenes.txt" > "$output_dir/crashes.txt"
}

count-tested-scenes() {
    wc -l < "$output_dir/all-tested-scenes.txt" #| tr -d '   '
}

count-warnings() {
    sort "$output_dir/warnings.txt" | uniq | wc -l | tr -d ' 	'
}

count-errors() {
    wc -l < "$output_dir/errors.txt" | tr -d ' 	'
}

count-crashes() {
    wc -l < "$output_dir/crashes.txt" #| tr -d '   '
}

print-summary() {
    echo "Scene testing summary:"
    echo "- $(count-tested-scenes) scene(s) tested"
    echo "- $(count-warnings) warning(s)"
    
    local errors='$(count-errors)'
    echo "- $(count-errors) error(s):"
    if [[ "$errors" != 0 ]]; then
        while read error; do
			echo "  - $error"
        done < "$output_dir/errors.txt"
    fi
    
    local crashes='$(count-crashes)'
    echo "- $(count-crashes) crash(es):"
    if [[ "$crashes" != 0 ]]; then
        while read scene; do
            if [[ -e "$output_dir/$scene/status.txt" ]]; then
                local status="$(cat "$output_dir/$scene/status.txt")"
                    case "$status" in
                    "timeout")
                        echo "  - Timeout: $scene"
                        ;;
                    [0-9]*)
                        if [[ "$status" -gt 128 && ( $(uname) = Darwin || $(uname) = Linux ) ]]; then
                            echo "  - Exit with status $status ($(kill -l $status)): $scene"
                        elif [[ "$status" != 0 ]]; then
                            echo "  - Exit with status $status: $scene"
                        fi
                        ;;
                    *)
                        echo "Error: unexpected value in $output_dir/$scene/status.txt: $status"
                        ;;
                esac
            fi
        done < "$output_dir/all-tested-scenes.txt"
    fi
}

if [[ "$command" = run ]]; then
    initialize-scene-testing
    test-all-scenes
    extract-warnings
    extract-errors
    extract-crashes
elif [[ "$command" = print-summary ]]; then
    print-summary
elif [[ "$command" = count-warnings ]]; then
    count-warnings
elif [[ "$command" = count-errors ]]; then
    count-errors
elif [[ "$command" = count-crashes ]]; then
    count-crashes
else
    echo "Unknown command: $command"
fi
