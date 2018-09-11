find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message("-- Cannot build the python module 'Sofa', missing the pybind software to generate the bindings.")
  return()
endif()

####################################################################################################
### Module files
####################################################################################################
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/Module_SofaRuntime.cpp
)


# sofa_add_pybind11_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2  OUTPUT ${CMAKE_CURRENT_BIN_DIR} NAME python_module_name)
sofa_add_pybind11_module(
        TARGET PythonModule_SofaRuntime
        SOURCES  ${SOURCE_FILES} 
        DEPENDS SofaPython3 SofaSimulationGraph PythonModule_Sofa
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/package/"
        NAME SofaRuntime
)
