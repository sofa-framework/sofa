#!/bin/bash
# set -o errexit # Exit on error

usage() {
    echo "Usage: macos-postinstall-fixup.sh <install-dir> [qt-dir] [macdeployqt]"
}

if [ "$#" -ge 1 ]; then
    INSTALL_DIR="$(cd $1 && pwd)"
    QT_DIR="$2"
    MACDEPLOYQT_EXE="$3"
else
    usage; exit 1
fi

if [[ $INSTALL_DIR == *".app" ]]; then
    BUNDLE_DIR=$INSTALL_DIR
    INSTALL_DIR=$INSTALL_DIR/Contents/MacOS
fi

echo "INSTALL_DIR = $INSTALL_DIR"
echo "BUNDLE_DIR = $BUNDLE_DIR"
echo "QT_DIR = $QT_DIR"
echo "MACDEPLOYQT_EXE = $MACDEPLOYQT_EXE"

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
elif [ -d "$QT_DIR" ]; then
    cp -Rf $QT_DIR/plugins/iconengines $INSTALL_DIR/bin
    cp -Rf $QT_DIR/plugins/imageformats $INSTALL_DIR/bin
    cp -Rf $QT_DIR/plugins/platforms $INSTALL_DIR/bin
    cp -Rf $QT_DIR/plugins/styles $INSTALL_DIR/bin
fi

echo "Fixing up libs manually ..."

check-all-deps() {
    mode="$1"
    pass="$2"

    (
    find "$INSTALL_DIR" -type f -name "Qt*" -path "*/Qt*.framework/Versions/*/Qt*" | grep -v "Headers"
    find "$INSTALL_DIR" -type f -name "*.dylib"
    find "$INSTALL_DIR" -type f -name "runSofa" -path "*/bin/*"
    ) | while read lib; do
        echo "  Checking (pass $pass) $lib"

        libqt=""
        libboost=""
        libicu=""
        libglew=""
        libjpeg=""
        libpng=""
        dependencies="$( otool -L $lib | tail -n +2 | perl -p -e 's/^[\t ]+(.*) \(.*$/\1/g' )"

        if echo "$dependencies" | grep --quiet --invert-match "/Qt" &&
           echo "$dependencies" | grep --quiet --invert-match "/libboost" &&
           echo "$dependencies" | grep --quiet --invert-match "/libicu" &&
           echo "$dependencies" | grep --quiet --invert-match "/libGLEW" &&
           echo "$dependencies" | grep --quiet --invert-match "/libjpeg" &&
           echo "$dependencies" | grep --quiet --invert-match "/libpng"; then
            # no lib to fixup in dependencies
            continue
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
            else
                # this dep is not a lib to fixup
                continue
            fi

            if [[ "$mode" == "copy" ]]; then
                if [ -n "$libqt" ]; then
                    originlib="$QT_DIR/lib/$libqt.framework"
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
                #echo "install_name_tool -change $dep @rpath/$rpathlib $lib"
                install_name_tool -change $dep @rpath/$rpathlib $lib
            fi
        done
    done
}

if [ -d "$BUNDLE_DIR" ]; then
    chmod -R 755 $BUNDLE_DIR/Contents/Frameworks
    chmod -R 755 $BUNDLE_DIR/Contents/MacOS/lib

    INSTALL_DIR=$BUNDLE_DIR
    check-all-deps "fixup" "1/1"

    # remove duplicated libs
    ls $BUNDLE_DIR/Contents/Frameworks | while read libname; do
        rm -rf $BUNDLE_DIR/Contents/MacOS/lib/$libname
    done
else
    check-all-deps "copy" "1/4"
    check-all-deps "copy" "2/4"
    check-all-deps "copy" "3/4"
    chmod -R 755 $INSTALL_DIR/lib
    check-all-deps "fixup" "4/4"
fi

echo "Done."