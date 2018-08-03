cmake_minimum_required(VERSION 3.1)
project(PythonSofaGeometry)

####################################################################################################
### Module dependencies
####################################################################################################
find_package(PythonLibs 2.7 QUIET)
if(!PythonLibs_FOUND)
    message(FATAL_ERROR "-- The python package 'SofaGeometry' requires the PythonLibs 2.7 cmake package.")
    return()
endif()

find_package(Cython QUIET)
if(!Cython_FOUND)
    message(FATAL_ERROR "-- The python package 'SofaGeometry' requires Cython to be installed to generate the bindings.")
    return()
endif()

find_package(SofaGeometry QUIET)
if(!SofaGeometry_FOUND)
    message(FATAL_ERROR "-- The python package 'SofaGeometry' requires the SofaGeometry c++module to be activated.")
    return()
endif()

####################################################################################################
### Module files
####################################################################################################
set(CYTHON_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/Plane.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/Ray.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/vector.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/Plane.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/Ray.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/vector.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/cpp_Plane.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/cpp_Ray.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry/cpp_vector.pxd
)

set(EXAMPLES_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/example1.py
    )

####################################################################################################
### Build the set of includes path to pass that to the pytonn cmake command.
### For each target (eg: SofaGeometry) we need to retrieve it INTERFACE_INCLUDE_DIRECTORIES.
### The list of directories is then transformed into a : separated string using the $<JOIN: >
### expression-generator.
### These informations are needed when building the binding using cython as it need to access
### the includes.
####################################################################################################
set(INCLUDES_PATH "$<JOIN:$<TARGET_PROPERTY:SofaGeometry,INTERFACE_INCLUDE_DIRECTORIES>,:>")
set(LIB_PATH "$<TARGET_FILE_DIR:SofaGeometry>")

### Defines a new custom command that compile the binding.
### The pyx are first converted into .cpp. Those cpp are then compiled into pyton modules.
add_custom_command(COMMAND python setup.py build_ext --inplace -I ${INCLUDES_PATH} -L ${LIB_PATH}   # Execute this command,
                   COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/SofaGeometry.timestamp
                   DEPENDS ${CYTHON_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py SofaGeometry        # The target depens on these files...
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                                    # In this working directory
                   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/SofaGeometry.timestamp)

add_custom_target(PythonSofaGeometry
                  ALL
                  SOURCES ${CYTHON_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/SofaGeometry.timestamp)

## Python configuration file (build tree), the lib in the source dir (easier while developping .py files)
file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/SofaGeometry" "${CMAKE_CURRENT_SOURCE_DIR}")

## Python configuration file (install tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaGeometry-config"
     "modules/SofaGeometry/python/")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaGeometry-config"
         DESTINATION "etc/sofa/python.d"
         RENAME "SofaGeometry")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/SofaGeometry DESTINATION modules/SofaGeometry/python)

add_subdirectory(tests)
