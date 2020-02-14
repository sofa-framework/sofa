#!/bin/bash

usage() {
    echo "Usage: linux-postinstall-fixup.sh <build-dir> <install-dir> [qt-dir]"
}

if [ "$#" -ge 2 ]; then
    BUILD_DIR="$(cd $1 && pwd)"
    INSTALL_DIR="$(cd $2 && pwd)"
    QT_DIR="$3"
    
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
rm -rf "$INSTALL_DIR/plugins/xcbglintegrations"

if [ -d "$QT_DIR" ]; then
    if [ -d "$QT_DIR/plugins/iconengines" ]; then
        cp -R "$QT_DIR/plugins/iconengines" "$INSTALL_DIR/bin"
    fi
    if [ -d "$QT_DIR/plugins/imageformats" ]; then
        cp -R "$QT_DIR/plugins/imageformats" "$INSTALL_DIR/bin"
    fi
    if [ -d "$QT_DIR/plugins/platforms" ]; then
        cp -R "$QT_DIR/plugins/platforms" "$INSTALL_DIR/bin"
    fi
    if [ -d "$QT_DIR/plugins/styles" ]; then
        cp -R "$QT_DIR/plugins/styles" "$INSTALL_DIR/bin"
    fi
    if [ -d "$QT_DIR/plugins/xcbglintegrations" ]; then
        cp -R "$QT_DIR/plugins/xcbglintegrations" "$INSTALL_DIR/bin"
    fi
fi

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
		if [[ "$group" == "libQt" ]] && [ -e "$QT_DIR/lib/$lib_name" ]; then
			lib_path="$QT_DIR/lib/$lib_name"
		else
			lib_path="$(cat $OUTPUT_TMP | grep "${lib_name} =>" | sed -e 's/.* => //g' | sort | uniq | head -n 1)"
		fi
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

# Add QtWebEngine dependencies
if [ -e "$INSTALL_DIR/lib/libQt5WebEngineCore.so.5" ] && [ -d "$QT_DIR" ]; then
	cp "$QT_DIR/libexec/QtWebEngineProcess" "$INSTALL_DIR/bin" # not in INSTALL_DIR/libexec ; see our custom bin/qt.conf
	mkdir "$INSTALL_DIR/translations"
	cp -R "$QT_DIR/translations/qtwebengine_locales" "$INSTALL_DIR/translations"
	cp -R "$QT_DIR/resources" "$INSTALL_DIR"
fi

echo "Done."
rm -f "$OUTPUT_TMP"
exit 0
