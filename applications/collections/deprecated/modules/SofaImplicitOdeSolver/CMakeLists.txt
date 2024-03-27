cmake_minimum_required(VERSION 3.22)
project(SofaImplicitOdeSolver LANGUAGES CXX)

set(SOFAIMPLICITEODESOLVER_SRC "src/${PROJECT_NAME}")

sofa_find_package(SofaFramework REQUIRED)

set(HEADER_FILES
    ${SOFAIMPLICITEODESOLVER_SRC}/config.h.in
    ${SOFAIMPLICITEODESOLVER_SRC}/initSofaImplicitOdeSolver.h
)

set(SOURCE_FILES
    ${SOFAIMPLICITEODESOLVER_SRC}/initSofaImplicitOdeSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.ODESolver.Backward instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.ODESolver.Backward REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.ODESolver.Backward)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
