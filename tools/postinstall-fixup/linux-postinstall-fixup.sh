#!/bin/bash

usage() {
    echo "Usage: linux-postinstall-fixup.sh <script-dir> <build-dir> <install-dir> [qt-lib-dir] [qt-data-dir]"
}

if [ "$#" -ge 2 ]; then
    SCRIPT_DIR="$(cd $1 && pwd)"
    BUILD_DIR="$(cd $2 && pwd)"
    INSTALL_DIR="$(cd $3 && pwd)"
    QT_LIB_DIR="$4"
    QT_DATA_DIR="$5"
else
    usage; exit 1
fi


echo "SCRIPT_DIR = $SCRIPT_DIR"
echo "BUILD_DIR = $BUILD_DIR"
echo "INSTALL_DIR = $INSTALL_DIR"
echo "QT_LIB_DIR = $QT_LIB_DIR"
echo "QT_DATA_DIR = $QT_DATA_DIR"


# Adapt INSTALL_DIR to IFW install
if [ -d "$INSTALL_DIR/packages/Runtime/data" ]; then
    INSTALL_DIR="$INSTALL_DIR/packages/Runtime/data"
fi

source $SCRIPT_DIR/common.sh
clean_default_plugins "$INSTALL_DIR/lib"

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

echo "QT_LIB_DIR = $QT_LIB_DIR"
echo "QT_DATA_DIR = $QT_DATA_DIR"
echo "QT_PLUGINS_DIR = $QT_PLUGINS_DIR"
echo "QT_LIBEXEC_DIR = $QT_LIBEXEC_DIR"
echo "QT_WEBENGINE_DATA_DIR = $QT_WEBENGINE_DATA_DIR"

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

echo_debug() {
    if [ -n "$DEBUG" ] && [ "$DEBUG" -gt 0 ]; then
        echo $*
    fi
}

get-lib-deps-assoc() {
    local base_dir="$1"
    local output="$2"
    local build_deps_file="$3"
    local libs="$(
        find "$base_dir" -type f -name "*.so*" -path "$base_dir/lib/*";
        find "$base_dir" -type f -name "*.so*" -path "$base_dir/bin/*";
        find "$base_dir" -type f -name "runSofa*" -path "$base_dir/bin/*";
        )"
    libs="$(echo "$libs" | tr -s '\n' ' ')"

    printf "" > "$output" # create empty output

    ldd $libs | # get all deps from libs in build_dir/[bin,lib] and install_dir/[bin,lib]
        grep " => [^ \.].* " | # remove unneeded results
        grep -v "$base_dir" | # remove deps already satisfied locally
        cut -c2- | # remove tabulation at beggining of each line
        sed -e 's/ (.*//g' | # keep only "libname => libpath"
        sort | uniq > "$output"

    # Try to fix "not found" dependencies
    grep "not found" "$output" | while read line; do
        libname="$(echo $line | sed -e 's/ => .*//g')"
        found="false"
        # try with ldd
        for lib in $libs; do
            #echo "RUN CMD: ldd $lib | grep \"$libname\" | head -n 1 | cut -c2- | sed -e 's/ (.*//g' | sed -e 's/.* => //g'"
            libpath="$( ldd $lib | grep "$libname" | sed -e 's/ (.*//g' | sort | head -n 1 | cut -c2- | sed -e 's/.* => //g' )"
            #echo "[ldd] libpath = $libpath"
            if [ -e "$libpath" ]; then
                echo_debug "      $libname found by ldd at $libpath"
                sed -i 's:'"$libname"'.* not found.*:'"$libname"' => '"$libpath"':g' "$output"
                found="true"
                break
            fi
        done
        if [[ "$found" == "false" ]] && [[ -n "$build_deps_file" ]]; then
            # try in build_deps_file
            libpath="$(grep "$libname" "$build_deps_file" | head -n 1 | sed -e 's/.* => //g')"
            #echo "[build_deps_file] libpath = $libpath"
            if [ -e "$libpath" ]; then
                echo_debug "      $libname found by build_deps_file at $libpath"
                sed -i 's:'"$libname"'.* not found.*:'"$libname"' => '"$libpath"':g' "$output"
                found="true"
            fi
        fi
        if [[ "$found" == "false" ]]; then
            echo "      $libname was not found"
        fi
    done
}


# Write dependencies to OUTPUT_TMP as "<lib-name> => <lib-path>" (from ldd output)
echo "  Listing dependencies of:"
echo "    - SOFA"
get-lib-deps-assoc "$BUILD_DIR" "postinstall_deps_SOFA-build.tmp"
get-lib-deps-assoc "$INSTALL_DIR" "postinstall_deps_SOFA.tmp" "postinstall_deps_SOFA-build.tmp"
for plugin in $INSTALL_DIR/plugins/*; do
    if [ -d "$plugin" ]; then
        plugin_name="$(basename "$plugin")"
        echo "    - $plugin_name"
        get-lib-deps-assoc "$plugin" "postinstall_deps_plugin_${plugin_name}.tmp" "postinstall_deps_SOFA-build.tmp"
    fi
done
echo "  Done."


# Copy libs
for deps_file in postinstall_deps_SOFA.tmp postinstall_deps_plugin_*.tmp; do
    target="lib"
    if [[ "$deps_file" == "postinstall_deps_plugin_"* ]]; then
        plugin_name="$(echo $deps_file | sed -e 's/postinstall_deps_plugin_\(.*\).tmp/\1/g')"
        target="plugins/$plugin_name/lib"
    fi
    echo_debug "-------------------------------"
    echo_debug "target = $target"

    groups="libQt libpng libicu libmng libxcb libxkb libpcre2 libjbig libwebp libjpeg libsnappy libtiff"
    for group in $groups; do
        echo_debug "    group = $group"

        # read all dep lib names matching the group
        lib_names="$(cat $deps_file | grep "${group}.* =>" | sed -e 's/ => .*//g' | sort | uniq)"
        echo_debug "    lib_names = $lib_names"
        group_dirname=""
        for lib_name in $lib_names; do
            echo_debug "    lib_name = $lib_name"
            if [[ "$target" != "lib" ]] && [[ -e "$INSTALL_DIR/lib/$lib_name" ]]; then
                # do not copy into plugins the libs that are already in SOFA/lib
                echo_debug "    $lib_name is already in $INSTALL_DIR/lib"
                continue
            fi
            # take first path found for the dep lib (paths are sorted so "/a/b/c" comes before "not found")
            if [[ "$group" == "libQt" ]] && [ -e "$QT_LIB_DIR/$lib_name" ]; then
                lib_path="$QT_LIB_DIR/$lib_name"
            else
                lib_path="$(cat $deps_file | grep "${lib_name} =>" | sed -e 's/.* => //g' | sort | uniq | head -n 1)"
            fi
            echo_debug "    lib_path = $lib_path"
            lib_path_to_copy=""
            if [[ -e "$lib_path" ]]; then
                lib_basename="$(basename $lib_path)"
                group_dirname="$(dirname $lib_path)"
                echo_debug "    group_dirname = $group_dirname"
                lib_path_to_copy="$lib_path"
            elif [[ -n "$group_dirname" ]] && [[ -e "$group_dirname/$lib_name" ]]; then
                lib_basename="$lib_name"
                lib_path_to_copy="$group_dirname/$lib_name"
            fi
            if [[ -e "$lib_path_to_copy" ]]; then
                echo "  Copying $lib_basename from $lib_path_to_copy to $target"
                cp -Rf "$lib_path_to_copy"* "$INSTALL_DIR/$target"
            fi
        done
    done
done

move_metis "$INSTALL_DIR"

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
echo "  Fixing RPATH: done."

echo "Fixing up libs: done."
rm -f postinstall_deps_*
exit 0
