find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)#
  message("-- Cannot build the python module 'Sofa', missing the pybind software to generate the bindings.")
  return()
endif()

add_subdirectory(Sofa)

#add_subdirectory(SofaRuntime)
#add_subdirectory(SofaTypes)
