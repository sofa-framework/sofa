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

        echo "  Checking $lib"

        (otool -L $lib | tail -n +2 | perl -p -e 's/^[\t ]+(.*) \(.*$/\1/g') | while read dep; do
            libboost="$(echo $dep | egrep -o "/libboost_[^\/]*?\.dylib" | cut -c2-)"
            libqt="$(echo $dep | egrep -o "/Qt[A-Za-z]*$" | cut -c2-)"
            libicu="$(echo $dep | egrep -o "/libicu[^\/]*?\.dylib$" | cut -c2-)"
            libglew="$(echo $dep | egrep -o "/libGLEW[^\/]*?\.dylib$" | cut -c2-)"
            libjpeg="$(echo $dep | egrep -o "/libjpeg[^\/]*?\.dylib$" | cut -c2-)"
            libpng="$(echo $dep | egrep -o "/libpng[^\/]*?\.dylib$" | cut -c2-)"
            
            if [ -n "$libboost" ]; then
                if [[ "$mode" == "copy" ]]; then
                    if [ -e $dep ] && [ ! -e $INSTALL_DIR/lib/$libboost ]; then
                        echo "    cp -Rf $dep $INSTALL_DIR/lib"
                        cp -Rf $dep $INSTALL_DIR/lib
                    fi
                elif [[ "$mode" == "fixup" ]]; then
                    #echo "install_name_tool -change $dep @rpath/$libboost $lib"
                    install_name_tool -change $dep @rpath/$libboost $lib
                fi
            elif [ -n "$libqt" ]; then
                if [[ "$mode" == "copy" ]]; then
                    if [ -e $QT_DIR/lib/$libqt.framework ] && [ ! -e $INSTALL_DIR/lib/$libqt.framework ]; then
                        echo "    cp -Rf $QT_DIR/lib/$libqt.framework $INSTALL_DIR/lib"
                        cp -Rf $QT_DIR/lib/$libqt.framework $INSTALL_DIR/lib
                    fi
                elif [[ "$mode" == "fixup" ]]; then
                    #echo "install_name_tool -change $dep @rpath/$libqt.framework/$libqt $lib"
                    install_name_tool -change $dep @rpath/$libqt.framework/$libqt $lib
                fi
            elif [ -n "$libicu" ]; then
                if [[ "$mode" == "copy" ]]; then
                    if [ -e $dep ] && [ ! -e $INSTALL_DIR/lib/$libicu ]; then
                        echo "    cp -Rf $dep $INSTALL_DIR/lib"
                        cp -Rf $dep $INSTALL_DIR/lib
                    fi
                elif [[ "$mode" == "fixup" ]]; then
                    #echo "install_name_tool -change $dep @rpath/$libicu $lib"
                    install_name_tool -change $dep @rpath/$libicu $lib
                fi
            elif [ -n "$libglew" ]; then
                if [[ "$mode" == "copy" ]]; then
                    if [ -e $dep ] && [ ! -e $INSTALL_DIR/lib/$libglew ]; then
                        echo "    cp -Rf $dep $INSTALL_DIR/lib"
                        cp -Rf $dep $INSTALL_DIR/lib
                    fi
                elif [[ "$mode" == "fixup" ]]; then
                    #echo "install_name_tool -change $dep @rpath/$libglew $lib"
                    install_name_tool -change $dep @rpath/$libglew $lib
                fi
            elif [ -n "$libjpeg" ]; then
                if [[ "$mode" == "copy" ]]; then
                    if [ -e $dep ] && [ ! -e $INSTALL_DIR/lib/$libjpeg ]; then
                        echo "    cp -Rf $dep $INSTALL_DIR/lib"
                        cp -Rf $dep $INSTALL_DIR/lib
                    fi
                elif [[ "$mode" == "fixup" ]]; then
                    #echo "install_name_tool -change $dep @rpath/$libjpeg $lib"
                    install_name_tool -change $dep @rpath/$libjpeg $lib
                fi
            elif [ -n "$libpng" ]; then
                if [[ "$mode" == "copy" ]]; then
                    if [ -e $dep ] && [ ! -e $INSTALL_DIR/lib/$libpng ]; then
                        echo "    cp -Rf $dep $INSTALL_DIR/lib"
                        cp -Rf $dep $INSTALL_DIR/lib
                    fi
                elif [[ "$mode" == "fixup" ]]; then
                    #echo "install_name_tool -change $dep @rpath/$libpng $lib"
                    install_name_tool -change $dep @rpath/$libpng $lib
                fi
            fi
        done
    done
}

if [ -d "$BUNDLE_DIR" ]; then
    chmod -R 755 $BUNDLE_DIR/Contents/Frameworks
    chmod -R 755 $BUNDLE_DIR/Contents/MacOS/lib

    INSTALL_DIR=$BUNDLE_DIR
    check-all-deps "fixup"

    # remove duplicated libs
    ls $BUNDLE_DIR/Contents/Frameworks | while read lib_name; do
        rm -rf $BUNDLE_DIR/Contents/MacOS/lib/$lib_name
    done
else
    check-all-deps "copy"
    check-all-deps "copy"
    check-all-deps "copy"
    chmod -R 755 $INSTALL_DIR/lib
    check-all-deps "fixup"
fi

echo "Done."