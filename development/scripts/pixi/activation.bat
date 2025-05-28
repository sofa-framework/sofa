:: Activation script

:: Create compile_commands.json for language server
set CMAKE_EXPORT_COMPILE_COMMANDS=1

:: Activate color output with Ninja
set CMAKE_COLOR_DIAGNOSTICS=1

:: Each environment have its dedicated build dir
set VISP_BUILD_DIR=%CONDA_PREFIX%\visp-build

:: Set default build value only if not previously set
if not defined SOFA_CMAKE_PREFIX_PATH (set SOFA_CMAKE_PREFIX_PATH=%CONDA_PREFIX%)
if not defined SOFA_BUILD_TYPE (set SOFA_BUILD_TYPE=Release)
if not defined SOFA_PYTHON_EXECUTABLE (set SOFA_PYTHON_EXECUTABLE="%CONDA_PREFIX%\python.exe")
