cmake_minimum_required(VERSION 3.1)
project(PythonModule_Sofa)

####################################################################################################
### Module dependencies
####################################################################################################
#find_package(PythonLibs 2.7 QUIET)
#if(NOT PythonLibs_FOUND)
#    message("-- Cannot build the python module 'SofaPython3', missing the required PythonLibs 2.7 cmake package.")
#    return()
#endif()


include(src/cython/CMakeBuilds.cmake)
include(src/pybind/CMakeBuilds.cmake)

set(EXAMPLES_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/example1.py
    )


## Python configuration file (build tree), the lib in the source dir (easier while developping .py files)
file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/Sofa" "${CMAKE_CURRENT_SOURCE_DIR}")

## Python configuration file (install tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-Sofa-config"
             "modules/Sofa/python/")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-Sofa-config"
         DESTINATION "etc/sofa/python.d"
         RENAME "Sofa")

#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/Sofa DESTINATION modules/Sofa/python)

#add_subdirectory(tests)
