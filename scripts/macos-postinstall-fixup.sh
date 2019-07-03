#!/bin/bash
# set -o errexit # Exit on error

usage() {
    echo "Usage: macos-postinstall-fixup.sh <install-dir>"
}

if [ "$#" -ge 1 ]; then
    INSTALL_DIR="$1"
else
    usage; exit 1
fi

if [ -e "$INSTALL_DIR/../MacOS/runSofa" ]; then
    echo "Moving executable to bin ..."
    mv -f "$INSTALL_DIR/runSofa" "$INSTALL_DIR/bin/runSofa"
fi

echo "Fixing up libs..."

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