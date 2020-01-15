#!/bin/bash
# set -o errexit # Exit on error

usage() {
    echo "Usage: macos-postinstall-fixup.sh <install-dir> [qt-dir] [macdeployqt]"
}

if [ "$#" -ge 1 ]; then
    INSTALL_DIR="$1"
    QT_DIR="$2"
    MACDEPLOYQT_EXE="$3"
else
    usage; exit 1
fi

echo "INSTALL_DIR = $INSTALL_DIR"
echo "QT_DIR = $QT_DIR"
echo "MACDEPLOYQT_EXE = $MACDEPLOYQT_EXE"

# Make sure the bin folder exists
if [ ! -d "$INSTALL_DIR/bin" ]; then
    mkdir -p $INSTALL_DIR/bin
fi

if [ -e "$INSTALL_DIR/../MacOS/runSofa" ]; then
    echo "Moving executable to bin ..."
    mv -f $INSTALL_DIR/../MacOS/runSofa* $INSTALL_DIR/../MacOS/bin/
fi

if [ -e "$MACDEPLOYQT_EXE" ]; then
    echo "Fixing up libs with MacDeployQt ..."
    $MACDEPLOYQT_EXE $INSTALL_DIR/../../../runSofa.app -always-overwrite

    cp -R $INSTALL_DIR/../PlugIns/* $INSTALL_DIR/../MacOS/bin && rm -rf $INSTALL_DIR/../PlugIns

    printf "[Paths] \n    Plugins = MacOS/bin \n" > $INSTALL_DIR/../Resources/qt.conf
elif [ -d "$QT_DIR" ]; then
    cp -Rf $QT_DIR/plugins/iconengines $INSTALL_DIR/bin
    cp -Rf $QT_DIR/plugins/imageformats $INSTALL_DIR/bin
    cp -Rf $QT_DIR/plugins/platforms $INSTALL_DIR/bin
    cp -Rf $QT_DIR/plugins/styles $INSTALL_DIR/bin
fi

echo "Fixing up libs manually ..."

(
find "$INSTALL_DIR" -type f -name "Qt*" -path "*/Qt*.framework/Versions/*/Qt*" | grep -v "Headers"
find "$INSTALL_DIR" -type f -name "*.dylib"
find "$INSTALL_DIR" -type f -name "runSofa" -path "*/bin/*"
) | while read lib; do

    libboost=""
    libqt=""
    libicu=""
    libglew=""
    libjpeg=""
    libpng=""

    echo -n "  Fixing $lib"

    (otool -L $lib | tail -n +2 | perl -p -e 's/^[\t ]+(.*) \(.*$/\1/g') | while read dep; do
        libboost="$(echo $dep | egrep -o "/libboost_[^\/]*?\.dylib" | cut -c2-)"
        libqt="$(echo $dep | egrep -o "/Qt[A-Za-z]*$" | cut -c2-)"
        libicu="$(echo $dep | egrep -o "/libicu[^\/]*?\.dylib$" | cut -c2-)"
        libglew="$(echo $dep | egrep -o "/libGLEW[^\/]*?\.dylib$" | cut -c2-)"
        libjpeg="$(echo $dep | egrep -o "/libjpeg[^\/]*?\.dylib$" | cut -c2-)"
        libpng="$(echo $dep | egrep -o "/libpng[^\/]*?\.dylib$" | cut -c2-)"
        
        if [ -n "$libboost" ]; then
            #echo "install_name_tool -change $dep @rpath/$libboost $lib"
            install_name_tool -change $dep @rpath/$libboost $lib
        elif [ -n "$libqt" ]; then
            if [ ! -e "$MACDEPLOYQT_EXE" ] && [ -d "$QT_DIR" ] && [ ! -e $INSTALL_DIR/lib/$libqt.framework ] ; then
                cp -Rf $QT_DIR/lib/$libqt.framework $INSTALL_DIR/lib
            fi
            #echo "install_name_tool -change $dep @rpath/$libqt.framework/$libqt $lib"
            install_name_tool -change $dep @rpath/$libqt.framework/$libqt $lib
        elif [ -n "$libicu" ]; then
            #echo "install_name_tool -change $dep @rpath/$libicu $lib"
            install_name_tool -change $dep @rpath/$libicu $lib
        elif [ -n "$libglew" ]; then
            #echo "install_name_tool -change $dep @rpath/$libglew $lib"
            install_name_tool -change $dep @rpath/$libglew $lib
        elif [ -n "$libjpeg" ]; then
            #echo "install_name_tool -change $dep @rpath/$libjpeg $lib"
            install_name_tool -change $dep @rpath/$libjpeg $lib
        elif [ -n "$libpng" ]; then
            #echo "install_name_tool -change $dep @rpath/$libpng $lib"
            install_name_tool -change $dep @rpath/$libpng $lib
        fi
    done

    echo ": done."
done

echo "Done."