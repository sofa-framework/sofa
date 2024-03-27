cmake_minimum_required(VERSION 3.22)
project(SofaMiscSolver LANGUAGES CXX)

set(SOFAMISCSOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCSOLVER_SRC}/config.h.in
    ${SOFAMISCSOLVER_SRC}/initSofaMiscSolver.h
)

set(SOURCE_FILES
    ${SOFAMISCSOLVER_SRC}/initSofaMiscSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Core Sofa.Simulation.Core)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.ODESolver.Backward and Sofa.Component.ODESolver.Forward instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.ODESolver.Backward REQUIRED)
sofa_find_package(Sofa.Component.ODESolver.Forward REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.ODESolver.Backward Sofa.Component.ODESolver.Forward)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
    )
