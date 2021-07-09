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
    Sofa.DefaultType
    Sofa.Core
    Sofa.SimulationCore
    )

# Temporary: to remove once the folders have been renamed
set(SOFAFRAMEWORK_LEGACYMODULES 
    Sofa.Type 
    Sofa.Topology 
    Sofa.Geometry 
    SofaHelper 
    SofaDefaultType 
    SofaCore 
    SofaSimulationCore
)

foreach(module ${SOFAFRAMEWORK_LEGACYMODULES})
    add_subdirectory(../modules/${module} ${CMAKE_CURRENT_BINARY_DIR}/${module})
endforeach()

# add subprojects into a IDE folder called SofaFramework
foreach(module ${SOFAFRAMEWORK_MODULES})
    set_target_properties(${module} PROPERTIES FOLDER SofaFramework) # IDE folder
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
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAFRAMEWORK_MODULES})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
