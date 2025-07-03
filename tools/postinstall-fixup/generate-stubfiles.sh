echo "Generate stubfiles..."

if [ "$#" -ge 3 ]; then
    BUILD_DIR="$(cd $1 && pwd)"
    INSTALL_DIR="$(cd $2 && pwd)"
    SYSTEM_NAME=$3


  echo "Inputs are"
  echo "- BUILD_DIR     :${BUILD_DIR}"
  echo "- INSTALL_DIR   :${INSTALL_DIR}"
  echo "- SYSTEM_NAME   :${SYSTEM_NAME}"
else
    echo "Usage: generate_stubfiles <BUILD_DIR> <INSTALL_DIR> [SYSTEM_NAME = 0]"; exit 1
fi

if [ -e "$VM_PYTHON3_EXECUTABLE" ]; then
    if [[ $SYSTEM_NAME = "Windows"* ]]; then
        pythonroot="$(dirname $VM_PYTHON3_EXECUTABLE)"
        pythonroot="$(cd "$pythonroot" && pwd)"
        export PATH="$pythonroot:$pythonroot/DLLs:$pythonroot/Lib:$PATH"
        PYTHON_SCRIPT=$(cd "${BUILD_DIR}/external_directories/fetched/SofaPython3/scripts" && pwd )/generate_stubs.py

        if [[ $SYSTEM_NAME = "Windows:NSIS" ]]; then
          PYTHON_INSTALL_SITE_PACKAGE_DIR=$(cd "$INSTALL_DIR/applications/plugins/SofaPython3/lib/python3/site-packages" && pwd )
          PYTHON_INSTALL_SITE_PACKAGE_DIR_LIB=$(cd "$INSTALL_DIR/libraries/plugins/SofaPython3/lib/python3/site-packages" && pwd )
          export PYTHONPATH="$PYTHON_INSTALL_SITE_PACKAGE_DIR:$PYTHON_INSTALL_SITE_PACKAGE_DIR_LIB:$PYTHONPATH"

          export PATH="$INSTALL_DIR/applications/lib:$INSTALL_DIR/applications/bin:$INSTALL_DIR/applications/plugins/SofaPython3/lib:$INSTALL_DIR/applications/plugins/SofaPython3/bin:$pythonroot/Lib:$PATH"
          export PATH="$INSTALL_DIR/libraries/lib:$INSTALL_DIR/libraries/bin:$INSTALL_DIR/libraries/plugins/SofaPython3/lib:$INSTALL_DIR/libraries/plugins/SofaPython3/bin:$pythonroot/Lib:$PATH"

          export SOFA_ROOT="$INSTALL_DIR/applications"

        else
          PYTHON_INSTALL_SITE_PACKAGE_DIR=$(cd "$INSTALL_DIR/plugins/SofaPython3/lib/python3/site-packages" && pwd )
          export PYTHONPATH="$PYTHON_INSTALL_SITE_PACKAGE_DIR:$PYTHONPATH"
          export PATH="$INSTALL_DIR/lib:$INSTALL_DIR/bin:$INSTALL_DIR/plugins/SofaPython3/lib:$INSTALL_DIR/plugins/SofaPython3/bin:$pythonroot/Lib:$PATH"
          export SOFA_ROOT="$INSTALL_DIR"

        fi
        echo "PATH=$PATH"
    else
        PYTHON_SCRIPT=$(cd "${BUILD_DIR}/external_directories/fetched/SofaPython3/scripts" && pwd )/generate_stubs.py
        if [[ $SYSTEM_NAME = *"IFW" ]]; then
          INSTALL_DIR="$INSTALL_DIR/packages/Runtime/data/"
        fi
        PYTHON_INSTALL_SITE_PACKAGE_DIR=$(cd "$INSTALL_DIR/plugins/SofaPython3/lib/python3/site-packages" && pwd )
        export PYTHONPATH="$PYTHON_INSTALL_SITE_PACKAGE_DIR:$PYTHONPATH"
        export SOFA_ROOT="$INSTALL_DIR"

    fi

    echo "SOFA_ROOT=$SOFA_ROOT"
    echo "PYTHONPATH=$PYTHONPATH"

    #Create folder if not already created

    python_exe="$VM_PYTHON3_EXECUTABLE"
    if [ -n "$python_exe" ]; then
        echo "Launching the stub generation with '$python_exe ${PYTHON_SCRIPT} -d $PYTHON_INSTALL_SITE_PACKAGE_DIR -m Sofa --use_pybind11'"
        $python_exe "${PYTHON_SCRIPT}" -d "$PYTHON_INSTALL_SITE_PACKAGE_DIR" -m Sofa --use_pybind11
    fi
else
    echo "VM_PYTHON3_EXECUTABLE doe not point to an existing file. To generate stubfiles you should point this env var to the Python3.XX executable."
fi
echo "Generate stubfiles: done."
