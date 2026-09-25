#! /bin/bash
# Activation script

# Create compile_commands.json for language server
export CMAKE_EXPORT_COMPILE_COMMANDS=1

# Activate color output with Ninja
export CMAKE_COLOR_DIAGNOSTICS=1

# Set default build value only if not previously set
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
# Each environment have its dedicated build dir
if [[ -z "$SOFA_BUILD_DIR" ]]; then
    export SOFA_BUILD_DIR=$CONDA_PREFIX/sofa-build
fi
if [[ -z "$SOFA_INSTALL_PREFIX" ]]; then
    export SOFA_INSTALL_PREFIX=$CONDA_PREFIX/
fi

export SOFA_BUILD_TYPE=${SOFA_BUILD_TYPE:=Release}
export SOFA_PYTHON_EXECUTABLE=${SOFA_PYTHON_EXECUTABLE:=$CONDA_PREFIX/bin/python}
export SOFA_BUILD_RELEASE_PACKAGE_WITH_PIXI=0
