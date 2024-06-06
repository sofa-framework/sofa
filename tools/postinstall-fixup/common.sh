#!/bin/bash

function clean_default_plugins()
{
  # Keep plugin_list as short as possible
  echo "" > "$1/plugin_list.conf"
  disabled_plugins='plugins_ignored_by_default'
  for plugin in \
          ArticulatedSystemPlugin   \
          Geomagic                  \
          BeamAdapter               \
          CGALPlugin                \
          CImgPlugin                \
          CollisionOBBCapsule       \
          CSparseSolvers            \
          DiffusionSolver           \
          image                     \
          InvertibleFVM             \
          ManifoldTopologies        \
          ModelOrderReduction       \
          PluginExample             \
          Registration              \
          SceneCreator              \
          SensableEmulation         \
          ShapeMatchingPlugin       \
          SofaAssimp                \
          SofaCarving               \
          SofaCUDA                  \
          SofaDistanceGrid          \
          SofaEulerianFluid         \
          SofaImplicitField         \
          SofaMatrix                \
          SoftRobots                \
          SofaSimpleGUI             \
          SofaSphFluid              \
          SofaValidation            \
          STLIB                     \
          VolumetricRendering       \
      ; do
      disabled_plugins=$disabled_plugins'\|'$plugin
  done
  grep -v $disabled_plugins "$1/plugin_list.conf.default" >> "$1/plugin_list.conf"
}

function move_metis()
{
  INSTALL_DIR=$1

  cd $INSTALL_DIR
  echo "Starting metis relocation..."
  if [[ "$(uname)" != "Darwin" && "$(uname)" != "Linux" ]]; then
    echo " - moving $(find ~+ -type f -name "metis.dll" | head -n 1) into $INSTALL_DIR/bin/"
    mv $(find ~+ -type f -name "metis.dll" | head -n 1) $INSTALL_DIR/bin/ || true
    echo " - moving $(find ~+ -type f -name "metis.lib" | head -n 1) into $INSTALL_DIR/lib/"
    mv $(find ~+ -type f -name "metis.lib" | head -n 1) $INSTALL_DIR/lib/ || true
  elif [[ "$(uname)" == "Darwin" ]]; then
    echo " - moving $( find ~+ -type d -name "metis.framework" | head -n 1) into $INSTALL_DIR/lib/"
      mv $( find ~+ -type d -name "metis.framework" | head -n 1 ) $INSTALL_DIR/lib/
  else
    echo " - moving $( find ~+ -type f -name "libmetis*" | head -n 1) into $INSTALL_DIR/lib/"
    mv $( find ~+ -type f -name "libmetis*" | head -n 1) $INSTALL_DIR/lib/
  fi
  echo " - moving $(find ~+ -type d -name "metis" | grep lib/cmake/metis | head -n 1) into $INSTALL_DIR/lib/cmake/"
  mv $(find ~+ -type d -name "metis" | grep lib/cmake/metis | head -n 1) $INSTALL_DIR/lib/cmake/ || true
  echo " - moving $(find ~+ -type f -name "metis.h" | head -n 1) into $INSTALL_DIR/include/"
  mv $(find ~+ -type f -name "metis.h" | head -n 1) $INSTALL_DIR/include/ || true
}
