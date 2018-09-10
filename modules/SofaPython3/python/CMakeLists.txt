cmake_minimum_required(VERSION 3.1)
project(PythonModule_SofaPython3)

####################################################################################################
### Module dependencies
####################################################################################################
#find_package(PythonLibs 2.7 QUIET)
#if(NOT PythonLibs_FOUND)
#    message("-- Cannot build the python module 'SofaPython3', missing the required PythonLibs 2.7 cmake package.")
#    return()
#endif()

find_package(Cython QUIET)
if(NOT Cython_FOUND)
    message("-- Cannot build the python module for 'SofaPython3', missing the Cython software to generate the bindings (http://cython.org/).")
    return()
endif()

####################################################################################################
### Module files
####################################################################################################
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/RunTime.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/Base.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/Base.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/BaseNode.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/BaseObject.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/BaseObject.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/BaseData.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/BaseData.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/Node.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/SceneLoaderPY3.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/cpp/sofa/helper/vector_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/cpp/sofa/core/objectmodel/Base_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/cpp/sofa/core/objectmodel/BaseData_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/cpp/sofa/core/objectmodel/BaseObject_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/cpp/sofa/core/objectmodel/BaseNode_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3/cpp/sofa/simulation/Node_wrap.pxd
)

set(EXAMPLES_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/example1.py
    )

# sofa_add_python_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2 CYTHONIZE True OUTPUT ${CMAKE_CURRENT_BIN_DIR})
sofa_add_python_module(TARGET PythonModule_SofaPython3
                       SOURCES ${SOURCE_FILES} ${EXAMPLES_FILES}
                       DEPENDS SofaPython3
                       OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3"
                       CYTHONIZE ${Cython_FOUND}
                       DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/SofaPython3"
                       )

## Python configuration file (build tree), the lib in the source dir (easier while developping .py files)
file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/SofaPython3" "${CMAKE_CURRENT_SOURCE_DIR}")

## Python configuration file (install tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaPython3-config"
             "modules/SofaPython3/python/")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaPython3-config"
         DESTINATION "etc/sofa/python.d"
         RENAME "SofaPython3")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/SofaPython3 DESTINATION modules/SofaPython3/python)

#add_subdirectory(tests)
