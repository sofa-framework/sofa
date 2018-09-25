cmake_minimum_required(VERSION 3.1)
project(PythonModule_SofaTypes)

####################################################################################################
### Module dependencies
####################################################################################################
#find_package(PythonLibs 2.7 QUIET)
#if(NOT PythonLibs_FOUND)
#    message("-- Cannot build the python module 'SofaPython3', missing the required PythonLibs 2.7 cmake package.")
#    return()
#endif()

# This must be done before doing the find_package(pybind11)
#set(PYBIND11_PYTHON_VERSION 3.6m)
#set(PYTHON_EXECUTABLE python3)

#include(src/cython/CMakeBuilds.cmake)
include(src/pybind/CMakeBuilds.cmake)

set(EXAMPLES_FILES
    # ${CMAKE_CURRENT_SOURCE_DIR}/examples/example1.py
    )


## Python configuration file (build tree), the lib in the source dir (easier while developping .py files)
file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/SofaTypes" "${CMAKE_CURRENT_SOURCE_DIR}/package/")

## Python configuration file (install tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaTypes-config"
             "modules/SofaTypes/python/")


install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaTypes-config"
         DESTINATION "etc/sofa/python.d"
         RENAME "SofaTypes")

#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/Sofa DESTINATION modules/Sofa/python)

add_subdirectory(tests)
