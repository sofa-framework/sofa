#! /bin/bash
# Activation script

# Create compile_commands.json for language server
export CMAKE_EXPORT_COMPILE_COMMANDS=1

# Activate color output with Ninja
export CMAKE_COLOR_DIAGNOSTICS=1

# Set default build value only if not previously set
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
# Each environment have its dedicated build dir
export SOFA_BUILD_DIR=$CONDA_PREFIX/sofa-build
export SOFA_BUILD_TYPE=${SOFA_BUILD_TYPE:=Release}
export SOFA_PYTHON_EXECUTABLE=${SOFA_BUILD_TYPE:=$CONDA_PREFIX/bin/python}
export SOFA_PLUGIN_SOFACUDA=${SOFA_PLUGIN_SOFACUDA:=ON}
