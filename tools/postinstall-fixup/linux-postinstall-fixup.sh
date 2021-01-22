#!/bin/bash

usage() {
    echo "Usage: linux-postinstall-fixup.sh <build-dir> <install-dir> [qt-lib-dir] [qt-data-dir]"
}

if [ "$#" -ge 2 ]; then
    BUILD_DIR="$(cd $1 && pwd)"
    INSTALL_DIR="$(cd $2 && pwd)"
    QT_LIB_DIR="$3"
    QT_DATA_DIR="$4"

    OUTPUT_TMP="all_deps.tmp"
else
    usage; exit 1
fi

# Keep plugin_list as short as possible
echo "" > "$INSTALL_DIR/lib/plugin_list.conf"
disabled_plugins='plugins_ignored_by_default'
for plugin in \
        SofaEulerianFluid     \
        SofaDistanceGrid      \
        SofaImplicitField     \
        MultiThreading        \
        DiffusionSolver       \
        image                 \
        Compliant             \
        SofaPython            \
        Flexible              \
        Registration          \
        ExternalBehaviorModel \
        ManifoldTopologies    \
        ManualMapping         \
        THMPGSpatialHashing   \
        SofaCarving           \
        RigidScale            \
    ; do
    disabled_plugins=$disabled_plugins'\|'$plugin
done
grep -v $disabled_plugins "$INSTALL_DIR/lib/plugin_list.conf.default" >> "$INSTALL_DIR/lib/plugin_list.conf"

echo "Fixing up libs..."

# Why are these folders installed in plugins?
rm -rf "$INSTALL_DIR/plugins/iconengines"
rm -rf "$INSTALL_DIR/plugins/imageformats"
rm -rf "$INSTALL_DIR/plugins/platforms"
rm -rf "$INSTALL_DIR/plugins/styles"
rm -rf "$INSTALL_DIR/plugins/xcbglintegrations"

QT_PLUGINS_DIR="$QT_DATA_DIR/plugins"
QT_LIBEXEC_DIR="$QT_DATA_DIR/libexec"
if [[ "$QT_LIB_DIR" == "/usr/lib"* ]]; then
    QT_WEBENGINE_DATA_DIR="/usr/share/qt5/resources"
else
    QT_WEBENGINE_DATA_DIR="$QT_DATA_DIR"
fi

if [ -d "$QT_DIR" ]; then
    if [ -d "$QT_PLUGINS_DIR/iconengines" ]; then
        cp -R "$QT_PLUGINS_DIR/iconengines" "$INSTALL_DIR/bin"
    fi
    if [ -d "$QT_PLUGINS_DIR/imageformats" ]; then
        cp -R "$QT_PLUGINS_DIR/imageformats" "$INSTALL_DIR/bin"
    fi
    if [ -d "$QT_PLUGINS_DIR/platforms" ]; then
        cp -R "$QT_PLUGINS_DIR/platforms" "$INSTALL_DIR/bin"
    fi
    if [ -d "$QT_PLUGINS_DIR/styles" ]; then
        cp -R "$QT_PLUGINS_DIR/styles" "$INSTALL_DIR/bin"
    fi
    if [ -d "$QT_PLUGINS_DIR/xcbglintegrations" ]; then
        cp -R "$QT_PLUGINS_DIR/xcbglintegrations" "$INSTALL_DIR/bin"
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
groups="libQt libpng libicu libmng libxcb libxkb libpcre2 libjbig libwebp libjpeg libsnappy"
for group in $groups; do
    echo_debug "group = $group"
    # read all dep lib names matching the group
    lib_names="$(cat $OUTPUT_TMP | grep "${group}.* =>" | sed -e 's/ => .*//g' | sort | uniq)"
    echo_debug "lib_names = $lib_names"
    group_dirname=""
    for lib_name in $lib_names; do
        echo_debug "lib_name = $lib_name"
        # take first path found for the dep lib (paths are sorted so "/a/b/c" comes before "not found")
		if [[ "$group" == "libQt" ]] && [ -e "$QT_LIB_DIR/$lib_name" ]; then
			lib_path="$QT_LIB_DIR/$lib_name"
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
if [ -e "$INSTALL_DIR/lib/libQt5WebEngineCore.so.5" ] && [ -d "$QT_LIBEXEC_DIR" ] && [ -d "$QT_WEBENGINE_DATA_DIR" ]; then
	cp "$QT_LIBEXEC_DIR/QtWebEngineProcess" "$INSTALL_DIR/bin" # not in INSTALL_DIR/libexec ; see our custom bin/qt.conf
	mkdir "$INSTALL_DIR/translations"
	cp -R "$QT_WEBENGINE_DATA_DIR/translations/qtwebengine_locales" "$INSTALL_DIR/translations"
	cp -R "$QT_WEBENGINE_DATA_DIR/resources" "$INSTALL_DIR"
fi

# Fixup RPATH/RUNPATH
echo "  Fixing RPATH..."
if [ -x "$(command -v patchelf)" ]; then
    defaultRPATH='$ORIGIN/../lib:$$ORIGIN/../lib'
    (
    find "$INSTALL_DIR" -type f -name "*.so.*"
    find "$INSTALL_DIR" -type f -name "runSofa" -path "*/bin/*"
    ) | while read lib; do
        if [[ "$(patchelf --print-rpath $lib)" == "" ]]; then
            echo "    $lib: RPATH = $defaultRPATH"
            patchelf --set-rpath $defaultRPATH $lib
        fi
        patchelf --shrink-rpath $lib
    done
else
    echo "    WARNING: patchelf command not found, RPATH fixing skipped."
fi
echo "  Done."

echo "Done."
rm -f "$OUTPUT_TMP"
exit 0
