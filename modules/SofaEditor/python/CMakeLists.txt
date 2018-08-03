cmake_minimum_required(VERSION 3.1)
project(PySofaEditor)

####################################################################################################
# Dependencies
####################################################################################################
find_package(SofaFramework REQUIRED)
find_package(PythonLibs 2.7 QUIET)
if(!PythonLibs_FOUND)
    message("-- PythonSofaEditor is disabled because the PythonLibs 2.7 cmake package is missing.")
    return()
endif()

find_package(SofaEditor QUIET)
if(!SofaEditor_FOUND)
    message("-- PythonSofaEditor is disabled because the 'SofaEditor' module is missing or not activated.")
    return()
endif()

find_package(Cython QUIET)
if(Cython_FOUND)
    message("-- PythonSofaEditor found 'cython' the .cpp will be regenerated from .pxd.")
else()
    message("-- PythonSofaEditor didn't found the 'cython' application, the binding with use default .cpp files.")
endif()

####################################################################################################
# Files
####################################################################################################
set(SRC_FILES
    ## Adds the pure python file
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/__init__.py

    ## Adds the declaration of the real cpp objects so they can be used by cython
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/cpp_SofaEditor.pxd

    ## Adds the definition of the python binding using cython. Each of these files will generates
    ## one .cpp with similar name.
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/SofaEditor.pyx
    )

####################################################################################################
# Targets & Builds
####################################################################################################
sofa_add_python_module(TARGET PythonSofaEditor
                       CYTHONIZE ${Cython_FOUND}
                       DEPENDS SofaEditor
                       SOURCES ${SRC_FILES}
                       OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor)

## Python configuration file (build tree), the lib in the source dir (easier while developping .py files)
file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/SofaEditor" "${CMAKE_CURRENT_SOURCE_DIR}")

## Python configuration file (install tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaEditor-config"
    "modules/SofaEditor/python/")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaEditor-config"
    DESTINATION "etc/sofa/python.d"
    RENAME "SofaEditor")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor DESTINATION modules/SofaEditor/python)

####################################################################################################
# Tests
####################################################################################################
add_subdirectory(tests)
