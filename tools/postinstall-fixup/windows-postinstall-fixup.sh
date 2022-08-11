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

# Adapt INSTALL_DIR to NSIS install
INSTALL_DIR_BIN="$INSTALL_DIR/bin"
if [[ "$INSTALL_DIR" == *"/NSIS/"* ]] && [[ -e "$INSTALL_DIR/../applications/bin" ]]; then
    INSTALL_DIR="$(cd $INSTALL_DIR/.. && pwd)"
    INSTALL_DIR_BIN="$INSTALL_DIR/applications/bin"
fi

echo "BUILD_DIR = $BUILD_DIR"
echo "INSTALL_DIR = $INSTALL_DIR"
echo "INSTALL_DIR_BIN = $INSTALL_DIR_BIN"

# Keep plugin_list as short as possible
echo "" > "$INSTALL_DIR_BIN/plugin_list.conf"
disabled_plugins='plugins_ignored_by_default'
for plugin in \
        ArticulatedSystemPlugin   \
        CollisionOBBCapsule       \
        Compliant                 \
        DiffusionSolver           \
        ExternalBehaviorModel     \
        Flexible                  \
        Geomagic                  \
        image                     \
        InvertibleFVM             \
        LMConstraint              \
        ManifoldTopologies        \
        ManualMapping             \
        MultiThreading            \
        OptiTrackNatNet           \
        PluginExample             \
        Registration              \
        RigidScale                \
        SensableEmulation         \
        SofaAssimp                \
        SofaCUDA                  \
        SofaCarving               \
        SofaDistanceGrid          \
        SofaEulerianFluid         \
        SofaImplicitField         \
        SofaPython                \
        SofaSimpleGUI             \
        SofaSphFluid              \
        THMPGSpatialHashing       \
    ; do
    disabled_plugins=$disabled_plugins'\|'$plugin
done
grep -v $disabled_plugins "$INSTALL_DIR_BIN/plugin_list.conf.default" >> "$INSTALL_DIR_BIN/plugin_list.conf"

# Copy all plugin libs in install/bin to make them easily findable
cd "$INSTALL_DIR" && find -name "*.dll" -path "*/plugins/*" | while read lib; do
    cp "$lib" "$INSTALL_DIR_BIN"
done

# Copy all collection libs in install/bin to make them easily findable
cd "$INSTALL_DIR" && find -name "*.dll" -path "*/collections/*" | while read lib; do
    cp "$lib" "$INSTALL_DIR_BIN"
done
