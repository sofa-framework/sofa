#!/bin/bash

usage() {
    echo "Usage: linux-postinstall-fixup.sh <build-dir> <install-dir>"
}

if [ "$#" -ge 2 ]; then
    BUILD_DIR="$(cd $1 && pwd)"
    INSTALL_DIR="$(cd $2 && pwd)"
    
    OUTPUT_TMP="all_deps.tmp"
else
    usage; exit 1
fi

echo "Fixing up libs..."

# Why are these folders installed in plugins?
rm -rf "$INSTALL_DIR/plugins/iconengines"
rm -rf "$INSTALL_DIR/plugins/imageformats"
rm -rf "$INSTALL_DIR/plugins/platforms"
rm -rf "$INSTALL_DIR/plugins/styles"

rm -f "$OUTPUT_TMP"

echo_debug() {
    if [ -n "$DEBUG" ] && [ "$DEBUG" -gt 0 ]; then
        echo $*
    fi
}

get-lib-deps-assoc() {
	local base_build_dir="$1"
	local base_install_dir="$2"
    local output="$3"
	local build_libs="$(find "$base_build_dir" -type f -name "*.so*" -path "$base_build_dir/lib/*" ; find "$base_build_dir" -type f -name "*.so*" -path "$base_build_dir/bin/*")"
	local install_libs="$(find "$base_install_dir" -type f -name "*.so*" -path "$base_install_dir/lib/*" ; find "$base_install_dir" -type f -name "*.so*" -path "$base_install_dir/bin/*")"

	ldd $build_libs $install_libs | # get all deps from libs in build_dir/[bin,lib] and install_dir/[bin,lib]
        grep " => [^ \.].* " | # remove unneeded results
        grep -v "$base_build_dir" | # remove deps already satisfied locally (in build_dir)
        grep -v "$base_install_dir" | # remove deps already satisfied locally (in install_dir)
        cut -c2- | # remove tabulation at beggining of each line
        sed -e 's/ (.*//g' | # keep only "libname => libpath"
        sort | uniq > "$output"
}


# Write dependencies to OUTPUT_TMP as "<lib-name> => <lib-path>" (from ldd output)
get-lib-deps-assoc "$BUILD_DIR" "$INSTALL_DIR" "$OUTPUT_TMP"

# Copy libs
groups="libQt libpng libicu libmng libxcb"
for group in $groups; do
    echo_debug "group = $group"
    # read all dep lib names matching the group
    lib_names="$(cat $OUTPUT_TMP | grep "${group}.* =>" | sed -e 's/ => .*//g' | sort | uniq)"
    echo_debug "lib_names = $lib_names"
    group_dirname=""
    for lib_name in $lib_names; do
        echo_debug "lib_name = $lib_name"
        # take first path found for the dep lib (paths are sorted so "/a/b/c" comes before "not found")
        lib_path="$(cat $OUTPUT_TMP | grep "${lib_name} =>" | sed -e 's/.* => //g' | sort | uniq | head -n 1)"
        echo_debug "lib_path = $lib_path"
        lib_path_to_copy=""
        if [[ -e "$lib_path" ]]; then
            lib_basename="$(basename $lib_path)"
            group_dirname="$(dirname $lib_path)"
            echo_debug "group_dirname = $group_dirname"
            lib_path_to_copy="$lib_path"
        elif [[ -n "$group_dirname" ]] && [[ -e "$group_dirname/$lib_name" ]]; then
            lib_basename="$lib_name"
            lib_path_to_copy="$group_dirname/$lib_name"
        fi
        if [[ -e "$lib_path_to_copy" ]]; then
            echo "  Copying $lib_basename from $lib_path_to_copy"
            cp -Rf "$lib_path_to_copy"* "$INSTALL_DIR/lib"
        fi
    done
done

echo "Done."
exit 0
