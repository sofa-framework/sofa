find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message("-- Cannot build the python module 'Sofa', missing the pybind software to generate the bindings.")
  return()
endif()

####################################################################################################
### Module files
####################################################################################################
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_BaseData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_BaseData.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_BaseObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_BaseObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Binding_Node.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Module_Sofa.cpp
)


# sofa_add_pybind11_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2  OUTPUT ${CMAKE_CURRENT_BIN_DIR} NAME python_module_name)
sofa_add_pybind11_module(
        TARGET PythonModule_Sofa
        SOURCES  ${SOURCE_FILES} 
        DEPENDS SofaPython3 SofaSimulationGraph
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/package/"
        NAME Sofa
)

target_include_directories(PythonModule_Sofa PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(PythonModule_Sofa PUBLIC "$<INSTALL_INTERFACE:include>")
