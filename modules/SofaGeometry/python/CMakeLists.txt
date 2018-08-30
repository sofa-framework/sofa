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

find_package(Cython QUIET)
if(NOT Cython_FOUND)
    message("-- Cannot build the python module 'SofaGeometry', missing the Cython software to generate the bindings (http://cython.org/).")
    return()
endif()

####################################################################################################
### Module files
####################################################################################################
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/Plane.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/Plane.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/Ray.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/Ray.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/vector.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/vector.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/cpp_Plane.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/cpp_Ray.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/cpp_vector.pxd
)

set(EXAMPLES_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/example1.py
    )

# sofa_add_python_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2 CYTHONIZE True OUTPUT ${CMAKE_CURRENT_BIN_DIR})
sofa_add_python_module(TARGET PythonSofaGeometry
                       SOURCES ${SOURCE_FILES} ${EXAMPLES_FILES}
                       DEPENDS SofaGeometry
                       OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry"
                       CYTHONIZE ${Cython_FOUND}
                       )

## Python configuration file (build tree), the lib in the source dir (easier while developping .py files)
file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/SofaGeometry" "${CMAKE_CURRENT_SOURCE_DIR}")

## Python configuration file (install tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaGeometry-config"
     "modules/SofaGeometry/python/")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaGeometry-config"
         DESTINATION "etc/sofa/python.d"
         RENAME "SofaGeometry")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry DESTINATION modules/SofaGeometry/python)

#add_subdirectory(tests)
