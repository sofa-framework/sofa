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

# Copy all plugin libs in install/bin to make them easily findable
cd "$INSTALL_DIR" && find -name "*.dll" -path "*/plugins/*" | while read lib; do
    cp "$lib" "$INSTALL_DIR_BIN"
done

# Copy all collection libs in install/bin to make them easily findable
cd "$INSTALL_DIR" && find -name "*.dll" -path "*/collections/*" | while read lib; do
    cp "$lib" "$INSTALL_DIR_BIN"
done
