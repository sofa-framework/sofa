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
          SofaDistanceGrid          \
          SofaDistanceGrid.CUDA     \
          SofaEulerianFluid         \
          SofaImplicitField         \
          SofaMatrix                \
          SoftRobots                \
          SofaSimpleGUI             \
          SofaSphFluid              \
          SofaValidation            \
          STLIB                     \
          VolumetricRendering       \
          CUDA                      \
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

function generate_stubfiles()
{
    if [ "$#" -eq 0 ]; then
        VM_IS_WINDOWS=0
    else
        VM_IS_WINDOWS=$1
    fi

    echo "Generate stubfiles..."
    if [ -e "$VM_PYTHON3_EXECUTABLE" ]; then
        export SOFA_ROOT="$INSTALL_DIR"

        if $VM_IS_WINDOWS; then

            pythonroot="$(dirname $VM_PYTHON3_EXECUTABLE)"
            pythonroot="$(cd "$pythonroot" && pwd)"
            export PATH="$pythonroot:$pythonroot/DLLs:$pythonroot/Lib:$PATH"
            PYTHON_SCRIPT=$(cd "$SRC_DIR/applications/plugins/SofaPython3/scripts" && pwd -W )\generate_stubs.py
            PYTHON_SITE_PACKAGE_DIR=$(cd "$INSTALL_DIR/plugins/SofaPython3/lib/python3/site-packages" && pwd -W )
            export PYTHONPATH="$PYTHON_SITE_PACKAGE_DIR:$PYTHONPATH"

        else

            PYTHON_SCRIPT=$(cd "$SRC_DIR/applications/plugins/SofaPython3/scripts" && pwd )/generate_stubs.py
            PYTHON_SITE_PACKAGE_DIR=$(cd "$INSTALL_DIR/plugins/SofaPython3/lib/python3/site-packages" && pwd )
            export PYTHONPATH="$PYTHON_SITE_PACKAGE_DIR:$PYTHONPATH"

        fi

        python_exe="$VM_PYTHON3_EXECUTABLE"
        if [ -n "$python_exe" ]; then
            echo "Launching the stub generation with '$python_exe ${PYTHON_SCRIPT} -d $PYTHON_SITE_PACKAGE_DIR -m Sofa --use_pybind11'"
            $python_exe "${PYTHON_SCRIPT}" -d "$PYTHON_SITE_PACKAGE_DIR" -m Sofa --use_pybind11
        fi
    else
        echo "VM_PYTHON3_EXECUTABLE doe not point to an existing file. To generate stubfiles you should point this env var to the Python3.XX executable."
    fi
    echo "Generate stubfiles: done."
}