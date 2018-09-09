cmake_minimum_required(VERSION 3.1)
project(PythonSofaGeometry)

####################################################################################################
### Module dependencies
####################################################################################################
find_package(PythonLibs 2.7 QUIET)
if(NOT PythonLibs_FOUND)
  message("-- Cannot build the python module 'SofaGeometry', missing the required PythonLibs 2.7 cmake package.")
  return()
endif()

find_package(SofaGeometry QUIET)
if(NOT SofaGeometry_FOUND)
  message("-- Cannot build the python module 'SofaGeometry', missing the SofaGeometry c++ module (or it is not activated in the build configuration).")
  return()
endif()

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message("-- Cannot build the python module 'SofaGeometry', missing the pybind software to generate the bindings (http://cython.org/).")
  return()
endif()

####################################################################################################
### Module files
####################################################################################################
set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/pythonSofaGeometry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/pythonSofaGeometry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/py_vec3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/py_ray.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/py_plane.cpp
  )

set(EXAMPLES_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/examples/example1.py
  )

# sofa_add_pybind11_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2  OUTPUT ${CMAKE_CURRENT_BIN_DIR} NAME python_module_name)
sofa_add_pybind11_module(TARGET PythonSofaGeometry
  SOURCES  ${SOURCE_FILES} ${EXAMPLES_FILES}
  DEPENDS SofaGeometry
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry"
  NAME SofaGeometry)

## Python configuration file (build tree), the lib in the source dir (easier while developping .py files)
file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/SofaGeometry" "${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry")

## Python configuration file (install tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaGeometry-config"
  "modules/SofaGeometry/python/")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaGeometry-config"
  DESTINATION "etc/sofa/python.d"
  RENAME "SofaGeometry")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry DESTINATION modules/SofaGeometry/python)

add_subdirectory(tests)
