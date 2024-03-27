cmake_minimum_required(VERSION 3.22)
project(SofaExplicitOdeSolver LANGUAGES CXX)

set(SOFAEXPLICITEODESOLVER_SRC "src/${PROJECT_NAME}")

sofa_find_package(SofaFramework REQUIRED)

set(HEADER_FILES
    ${SOFAEXPLICITEODESOLVER_SRC}/config.h.in
    ${SOFAEXPLICITEODESOLVER_SRC}/initSofaExplicitOdeSolver.h
)

set(SOURCE_FILES
    ${SOFAEXPLICITEODESOLVER_SRC}/initSofaExplicitOdeSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.ODESolver.Forward instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.ODESolver.Forward REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.ODESolver.Forward)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
