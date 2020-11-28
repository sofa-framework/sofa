#!bash

usage() {
    echo "Usage: windows-postinstall-fixup.sh <build-dir> <install-dir>"
}

if [ "$#" -ge 2 ]; then
    BUILD_DIR="$(cd $1 && pwd)"
    INSTALL_DIR="$(cd $2 && pwd)"
else
    usage; exit 1
fi

# Keep plugin_list as short as possible
echo "" > "$INSTALL_DIR/bin/plugin_list.conf"
for plugin in \
        SofaExporter       \
        SofaSparseSolver   \
        SofaPreconditioner \
        SofaValidation     \
        SofaDenseSolver    \
        SofaNonUniformFem  \
        SofaOpenglVisual   \
        SofaSphFluid       \
        CImgPlugin         \
        SofaMiscCollision  \
    ; do
    grep "$plugin" "$INSTALL_DIR/bin/plugin_list.conf.default" >> "$INSTALL_DIR/bin/plugin_list.conf"
done

# Link all plugin libs in install/bin to make them easily findable
cd "$INSTALL_DIR" && find "plugins" -name "*.dll" | while read lib; do
    libname="$(basename $lib)"
    ln -s "../$lib" "bin/$libname"
done
