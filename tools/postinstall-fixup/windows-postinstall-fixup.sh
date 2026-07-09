#!bash

usage() {
    echo "Usage: windows-postinstall-fixup.sh <script-dir> <build-dir> <install-dir>"
}

if [ "$#" -ge 2 ]; then
    SCRIPT_DIR="$(cd $1 && pwd)"
    BUILD_DIR="$(cd $2 && pwd)"
    INSTALL_DIR="$(cd $3 && pwd)"
else
    usage; exit 1
fi

# Adapt INSTALL_DIR to NSIS install
INSTALL_DIR_BIN="$INSTALL_DIR/bin"
if [[ "$INSTALL_DIR" == *"/NSIS/"* ]] && [[ -e "$INSTALL_DIR/../applications/bin" ]]; then
    INSTALL_DIR="$(cd $INSTALL_DIR/.. && pwd)"
    INSTALL_DIR_BIN="$INSTALL_DIR/applications/bin"
fi

echo "SCRIPT_DIR = $SCRIPT_DIR"
echo "BUILD_DIR = $BUILD_DIR"
echo "INSTALL_DIR = $INSTALL_DIR"
echo "INSTALL_DIR_BIN = $INSTALL_DIR_BIN"

source $SCRIPT_DIR/common.sh
clean_default_plugins "$INSTALL_DIR_BIN"

move_metis "$INSTALL_DIR"

# Copy all plugin libs in install/bin (preserving sub-directory structure) to make them easily findable
curDir="$INSTALL_DIR"

if [[ "$INSTALL_DIR" == *"/NSIS/"* ]]; then
    curDir="$curDir/applications"
fi
cd "$curDir"

for p in plugins/*; do
cd "$curDir/$p" && find . -name "*.dll" | while read lib; do
    echo "Moving $lib to $curDir/$lib"
    cp --parents "$lib" "$curDir"
done
done


# Copy all plugins config files in etc/ini to make them easily findable
INSTALL_DIR_ETC="$INSTALL_DIR/etc"
mkdir -p $INSTALL_DIR_ETC
cd "$INSTALL_DIR" && find -name "*.ini" -path "*/plugins/*/etc/*" | while read f; do
    cp  "$f" "$INSTALL_DIR_ETC"
done

