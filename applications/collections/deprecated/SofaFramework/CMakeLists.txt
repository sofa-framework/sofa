cmake_minimum_required(VERSION 3.12)
project(SofaFramework)

# Version
set(SOFAFRAMEWORK_VERSION "${Sofa_VERSION}")

# Modules
set(SOFAFRAMEWORK_MODULES
    Sofa.Type
    Sofa.Topology
    Sofa.Geometry 
    Sofa.Helper
    Sofa.LinearAlgebra
    Sofa.DefaultType
    Sofa.Core
    Sofa.Simulation.Core
    )

set(SOFAFRAMEWORK_LEGACYMODULES
    SofaHelper 
    SofaDefaultType
    SofaCore 
    Sofa.SimulationCore
)

# add subprojects into a IDE folder called SofaFramework
foreach(module ${SOFAFRAMEWORK_MODULES})
    sofa_find_package(${module})
endforeach()

foreach(module ${SOFAFRAMEWORK_LEGACYMODULES})
    add_subdirectory(../modules/${module} ${CMAKE_CURRENT_BINARY_DIR}/${module})
endforeach()

set(SOFAFRAMEWORK_SRC src/SofaFramework)
set(HEADER_FILES
    ${SOFAFRAMEWORK_SRC}/config.h.in
    ${SOFAFRAMEWORK_SRC}/initSofaFramework.h
    )
set(SOURCE_FILES
    ${SOFAFRAMEWORK_SRC}/initSofaFramework.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAFRAMEWORK_MODULES} ${SOFAFRAMEWORK_LEGACYMODULES})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER SofaFramework)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
#    RELOCATABLE "collections" # disabled because we need SofaFramework to be directly available
    )
