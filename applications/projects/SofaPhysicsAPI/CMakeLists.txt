cmake_minimum_required(VERSION 3.12)
project(SofaPhysicsAPI LANGUAGES CXX)

set(SOFAPHYSICSAPI_SRC_DIR "src/SofaPhysicsAPI")
set(HEADER_FILES
    ${SOFAPHYSICSAPI_SRC_DIR}/config.h.in
    ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsAPI.h
    ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsOutputMesh_impl.h
    ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsSimulation.h
    ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsBindings.h
    ${SOFAPHYSICSAPI_SRC_DIR}/fakegui.h
)

set(SOURCE_FILES
    ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsOutputMesh.cpp
    ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsSimulation.cpp
    ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsBindings.cpp
    ${SOFAPHYSICSAPI_SRC_DIR}/fakegui.cpp
)

find_package(Sofa.Simulation.Graph REQUIRED)
sofa_find_package(Sofa.Component REQUIRED)
sofa_find_package(Sofa.GUI.Common REQUIRED)
sofa_find_package(SofaValidation QUIET)

if(SofaValidation_FOUND)
    message("SofaPhysicsAPI: SofaValidation has been found; enabling DataController and DataMonitor support.")
    list(APPEND HEADER_FILES 
        ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsDataController_impl.h
        ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsDataMonitor_impl.h
    )
    list(APPEND HEADER_FILES 
        ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsDataController.cpp
        ${SOFAPHYSICSAPI_SRC_DIR}/SofaPhysicsDataMonitor.cpp
    )
else()
    message("SofaPhysicsAPI: SofaValidation has not been found; DataController and DataMonitor will not be supported.")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Graph Sofa.GUI.Common)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component)

if(SofaValidation_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaValidation)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${Sofa_VERSION})


## Install rules and CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME  ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )
