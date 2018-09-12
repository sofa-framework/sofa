cmake_minimum_required(VERSION 3.1)
project(SofaPython3 VERSION 1.0)

find_package(SofaFramework REQUIRED)
find_package(SofaSimulation REQUIRED)

set(PYTHON3_LIBRARY "python3.6m")
set(PYTHON3_INCLUDE_DIR "/usr/include/python3.6/")
set(PYTHON_LIBRARIES ${PYTHON3_LIBRARY})
set(PYTHON_INCLUDE_DIRS ${PYTHON3_INCLUDE_DIR})

set(PYTHON_EXECUTABLE python3)
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message("-- Cannot build the python module 'Sofa', missing the pybind software to generate the bindings.")
  return()
endif()

set(HEADER_FILES
    src/SofaPython3/config.h
    src/SofaPython3/initModule.h
    src/SofaPython3/PythonEnvironment.h   
    src/SofaPython3/PythonTest.h
)

set(SOURCE_FILES
    src/SofaPython3/initModule.cpp
    src/SofaPython3/PythonEnvironment.cpp
    src/SofaPython3/PythonTest.cpp
)

set(EXTRA_FILES
    SofaPython3Config.cmake.in
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAPYTHON3")

target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationCore pybind11::embed)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

target_link_libraries(${PROJECT_NAME} PRIVATE "${PYTHON3_LIBRARY}" )

target_include_directories(${PROJECT_NAME} PRIVATE "${PYTHON3_INCLUDE_DIR}")


## When installing, copy the content of the example directory into module/SofaPython3
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples DESTINATION modules/SofaPython3 COMPONENT resources)

## When installing, copy the target into module/SofaPython3
install(TARGETS ${PROJECT_NAME} DESTINATION modules/SofaPython3)

# Config files and install rules for pythons scripts
sofa_set_python_directory(${PROJECT_NAME} "python")

# The signature of sofa_create_package is
#     sofa_create_package(package_name, version, the_targets, <include_subdir>)
# This function assumes that there is a FooConfig.cmake.in file template in the source directory.
# This function is an all in one replacement for install(TARGET), install(EXPORT), and also add
# the example directory in the INSTALL set.
sofa_create_package(${PROJECT_NAME} ${PROJECT_VERSION} SofaPython3 "SofaPython3")

### Tests
#add_subdirectory(tests)

### Python binding
add_subdirectory(bindings)
