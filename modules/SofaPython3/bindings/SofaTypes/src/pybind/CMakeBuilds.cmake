find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message("-- Cannot build the python module 'SofaTypes', missing the pybind software to generate the bindings.")
  return()
endif()

####################################################################################################
### Module files
####################################################################################################
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Module_SofaTypes.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_BoundingBox.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Color.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Frame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Mat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Quat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Vec.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_BoundingBox.h
#    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Color.h
#    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Frame.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Mat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Quat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Vec.h
)


# sofa_add_pybind11_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2  OUTPUT ${CMAKE_CURRENT_BIN_DIR} NAME python_module_name)
sofa_add_pybind11_module(
        TARGET PythonModule_SofaTypes
        SOURCES  ${SOURCE_FILES} 
        DEPENDS SofaPython3 SofaDefaultType
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/package/"
        NAME SofaTypes
)
