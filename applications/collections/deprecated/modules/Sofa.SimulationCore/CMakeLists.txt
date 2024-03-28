cmake_minimum_required(VERSION 3.22)
project(Sofa.SimulationCore)

set(SOURCE_FILES
    empty.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Simulation.Core instead.")

sofa_find_package(Sofa.Simulation.Core REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER SofaFramework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
