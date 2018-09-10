find_package(Cython QUIET)
if(NOT Cython_FOUND)
    message("-- Cannot build the python module for 'Sofa', missing the Cython software to generate the bindings (http://cython.org/).")
    return()
endif()

####################################################################################################
### Module files
####################################################################################################
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/Base.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/Base.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/BaseNode.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/BaseNode.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/BaseObject.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/BaseObject.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/BaseData.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/BaseData.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/Node.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/Node.pxd    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/SceneLoaderPY3.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/cpp/sofa/helper/vector_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/cpp/sofa/core/objectmodel/Base_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/cpp/sofa/core/objectmodel/BaseData_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/cpp/sofa/core/objectmodel/BaseObject_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/cpp/sofa/core/objectmodel/BaseNode_wrap.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/cpp/sofa/simulation/Node_wrap.pxd
)


# sofa_add_python_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2 CYTHONIZE True OUTPUT ${CMAKE_CURRENT_BIN_DIR})
sofa_add_python_module(TARGET PythonModule_Sofa
                       SOURCES ${SOURCE_FILES} ${EXAMPLES_FILES}
                       DEPENDS SofaPython3
                       OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa/"
                       CYTHONIZE ${Cython_FOUND}
                       DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/cython/Sofa"
                       )

