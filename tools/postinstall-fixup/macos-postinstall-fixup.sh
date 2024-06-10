#!/bin/bash
# set -o errexit # Exit on error

usage() {
    echo "Usage: macos-postinstall-fixup.sh <script-dir> <install-dir> [qt-lib-dir] [qt-data-dir] [macdeployqt]"
}

if [ "$#" -ge 1 ]; then
    SCRIPT_DIR="$(cd $1 && pwd)"
    INSTALL_DIR="$(cd $2 && pwd)"
    QT_LIB_DIR="$3"
    QT_DATA_DIR="$4"
    MACDEPLOYQT_EXE="$5"
else
    usage; exit 1
fi

if [[ $INSTALL_DIR == *".app" ]]; then
    BUNDLE_DIR=$INSTALL_DIR
    INSTALL_DIR=$INSTALL_DIR/Contents/MacOS
fi

echo "SCRIPT_DIR = $SCRIPT_DIR"
echo "INSTALL_DIR = $INSTALL_DIR"
echo "BUNDLE_DIR = $BUNDLE_DIR"
echo "QT_LIB_DIR = $QT_LIB_DIR"
echo "QT_DATA_DIR = $QT_DATA_DIR"
echo "MACDEPLOYQT_EXE = $MACDEPLOYQT_EXE"

source $SCRIPT_DIR/common.sh
clean_default_plugins "$INSTALL_DIR/lib"

# Make sure the bin folder exists and contains runSofa
if [ ! -d "$INSTALL_DIR/bin" ]; then
    mkdir -p $INSTALL_DIR/bin
fi
if [ -e "$INSTALL_DIR/runSofa" ]; then
    echo "Moving executable to bin ..."
    mv -f $INSTALL_DIR/runSofa* $INSTALL_DIR/bin/
fi

if [ -d "$BUNDLE_DIR" ] && [ -e "$MACDEPLOYQT_EXE" ]; then
    echo "Fixing up libs with MacDeployQt ..."
    $MACDEPLOYQT_EXE $BUNDLE_DIR -always-overwrite

    cp -R $BUNDLE_DIR/Contents/PlugIns/* $BUNDLE_DIR/Contents/MacOS/bin && rm -rf $BUNDLE_DIR/Contents/PlugIns

    printf "[Paths] \n    Plugins = MacOS/bin \n" > $BUNDLE_DIR/Contents/Resources/qt.conf
elif [ -d "$QT_DATA_DIR" ]; then
    cp -Rf $QT_DATA_DIR/plugins/iconengines $INSTALL_DIR/bin
    cp -Rf $QT_DATA_DIR/plugins/imageformats $INSTALL_DIR/bin
    cp -Rf $QT_DATA_DIR/plugins/platforms $INSTALL_DIR/bin
    cp -Rf $QT_DATA_DIR/plugins/styles $INSTALL_DIR/bin
fi

move_metis "$INSTALL_DIR"

echo "Fixing up libs manually ..."

check-all-deps() {
    mode="$1"
    pass="$2"

    (
    find "$INSTALL_DIR" -type f -name "Qt*" -path "*/Qt*.framework/Versions/*/Qt*" | grep -v "Headers"
    find "$INSTALL_DIR" -type f -name "*.dylib"
    find "$INSTALL_DIR" -type f -name "*.so"
    find "$INSTALL_DIR" -type f -name "runSofa*" -path "*/bin/*"
    ) | while read lib; do
        echo "  Checking (pass $pass) $lib"

        libqt=""
        libboost=""
        libicu=""
        libglew=""
        libjpeg=""
        libpng=""
        libtinyxml2=""
        libtiff=""
        libzstd=""
        liblzma=""
        dependencies="$( otool -L $lib | tail -n +2 | perl -p -e 's/^[\t ]+(.*) \(.*$/\1/g' )"

        is_fixup_needed="false"
        if echo "$dependencies" | grep --quiet "/Qt"       ||
           echo "$dependencies" | grep --quiet "/libboost" ||
           echo "$dependencies" | grep --quiet "/libicu"   ||
           echo "$dependencies" | grep --quiet "/libGLEW"  ||
           echo "$dependencies" | grep --quiet "/libjpeg"  ||
           echo "$dependencies" | grep --quiet "/libpng"   ||
           echo "$dependencies" | grep --quiet "/libtinyxml2"  ||
           echo "$dependencies" | grep --quiet "/libtiff"  ||
           echo "$dependencies" | grep --quiet "/libzstd"  ||
           echo "$dependencies" | grep --quiet "/liblzma"  ; then
            is_fixup_needed="true"
        fi
        if [[ "$is_fixup_needed" == "false" ]]; then
            continue # skip this lib
        fi

        (echo "$dependencies") | while read dep; do
            if libqt="$(echo $dep | egrep -o "/Qt[A-Za-z]*$" | cut -c2-)" && [ -n "$libqt" ]; then
                libname="$libqt"
            elif libboost="$(echo $dep | egrep -o "/libboost_[^\/]*?\.dylib" | cut -c2-)" && [ -n "$libboost" ]; then
                libname="$libboost"
            elif libicu="$(echo $dep | egrep -o "/libicu[^\/]*?\.dylib$" | cut -c2-)" && [ -n "$libicu" ]; then
                libname="$libicu"
            elif libglew="$(echo $dep | egrep -o "/libGLEW[^\/]*?\.dylib$" | cut -c2-)" && [ -n "$libglew" ]; then
                libname="$libglew"
            elif libjpeg="$(echo $dep | egrep -o "/libjpeg[^\/]*?\.dylib$" | cut -c2-)" && [ -n "$libjpeg" ]; then
                libname="$libjpeg"
            elif libpng="$(echo $dep | egrep -o "/libpng[^\/]*?\.dylib$" | cut -c2-)" && [ -n "$libpng" ]; then
                libname="$libpng"
            elif libtinyxml2="$(echo $dep | egrep -o "/libtinyxml2[^\/]*?\.dylib$" | cut -c2-)" && [ -n "$libtinyxml2" ]; then
                libname="$libtinyxml2"
            elif libtiff="$(echo $dep | egrep -o "/libtiff[^\/]*?\.dylib$" | cut -c2-)" && [ -n "$libtiff" ]; then
                libname="$libtiff"
            elif libzstd="$(echo $dep | egrep -o "/libzstd[^\/]*?\.dylib$" | cut -c2-)" && [ -n "$libzstd" ]; then
                libname="$libzstd"
            elif liblzma="$(echo $dep | egrep -o "/liblzma[^\/]*?\.dylib$" | cut -c2-)" && [ -n "$liblzma" ]; then
                libname="$liblzma"
            else
                if [[ "$dep" == "/usr/local/"* ]]; then
                    echo "WARNING: no fixup rule set for: $dep"
                fi
                # this dep is not a lib to fixup
                continue
            fi

            if [[ "$mode" == "copy" ]]; then
                if [ -n "$libqt" ]; then
                    originlib="$QT_LIB_DIR/$libqt.framework"
                    destlib="$INSTALL_DIR/lib/$libqt.framework"
                else
                    originlib="$dep"
                    destlib="$INSTALL_DIR/lib/$libname"
                fi
                if [ -e $originlib ] && [ ! -e $destlib ]; then
                    echo "    cp -Rf $dep $INSTALL_DIR/lib"
                    cp -Rf $originlib $INSTALL_DIR/lib
                fi
            elif [[ "$mode" == "fixup" ]]; then
                if [ -n "$libqt" ]; then
                    rpathlib="$libqt.framework/$libqt"
                else
                    rpathlib="$libname"
                fi
                libbasename="$(basename $lib)"
                echo "install_name_tool -change $dep @rpath/$rpathlib $libbasename"
                install_name_tool -change $dep @rpath/$rpathlib $lib
            fi
        done
    done
}

if [ -d "$BUNDLE_DIR" ]; then
    chmod -R 755 $BUNDLE_DIR/Contents/Frameworks
    chmod -R 755 $BUNDLE_DIR/Contents/MacOS/lib

    INSTALL_DIR=$BUNDLE_DIR

    # remove duplicated libs
    find "$BUNDLE_DIR/Contents/MacOS" -type f -name "*.dylib" | while read lib; do
        libname="$(basename $lib)"
        # libs in all dirs should not be in Frameworks/*
        rm -rf $BUNDLE_DIR/Contents/Frameworks/$libname*
        if [[ "$lib" == *"Contents/MacOS/plugins/"* ]]; then
            # libs in plugins/* should not be in lib/*
            rm -rf $BUNDLE_DIR/Contents/Contents/MacOS/lib/$libname*
        fi
    done

    check-all-deps "fixup" "1/1"
else
    check-all-deps "copy" "1/4"
    check-all-deps "copy" "2/4"
    check-all-deps "copy" "3/4"
    chmod -R 755 $INSTALL_DIR/lib
    check-all-deps "fixup" "4/4"
fi

if [ -d "$BUNDLE_DIR" ]; then
    # Adding default RPATH to all libs and to runSofa
    rm -f install_name_tool.errors.log
    (
    find "$BUNDLE_DIR/Contents/MacOS" -type f -name "*.dylib"
    find "$BUNDLE_DIR/Contents/MacOS" -type f -name "*.so"
    find "$BUNDLE_DIR" -type f -name "runSofa*" -path "*/bin/*"
    ) | while read lib; do
        install_name_tool -add_rpath "@loader_path/../lib" $lib 2>> install_name_tool.errors.log
        install_name_tool -add_rpath "@executable_path/../lib" $lib 2>> install_name_tool.errors.log
        if [[ "$lib" == *"Contents/MacOS/plugins/"* ]]; then
            install_name_tool -add_rpath "@loader_path/../../../../Frameworks" $lib 2>> install_name_tool.errors.log
            install_name_tool -add_rpath "@executable_path/../../../../Frameworks" $lib 2>> install_name_tool.errors.log
        else
            install_name_tool -add_rpath "@loader_path/../../Frameworks" $lib 2>> install_name_tool.errors.log
            install_name_tool -add_rpath "@executable_path/../../Frameworks" $lib 2>> install_name_tool.errors.log
        fi
    done
    ls -d $BUNDLE_DIR/Contents/MacOS/plugins/*/ | while read plugin; do
        pluginname="$(basename $plugin)"
        install_name_tool -add_rpath "@loader_path/../plugins/$pluginname/lib" "$BUNDLE_DIR/Contents/MacOS/bin/runSofa" 2>> install_name_tool.errors.log
        install_name_tool -add_rpath "@executable_path/../plugins/$pluginname/lib" "$BUNDLE_DIR/Contents/MacOS/bin/runSofa" 2>> install_name_tool.errors.log
    done
    cat install_name_tool.errors.log | grep -v 'file already has LC_RPATH for' >&2
fi

echo "Done."
