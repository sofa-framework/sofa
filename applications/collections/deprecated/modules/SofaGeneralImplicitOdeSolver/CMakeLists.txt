cmake_minimum_required(VERSION 3.22)
project(SofaGeneralImplicitOdeSolver LANGUAGES CXX)

set(SOFAGENERALIMPLICITODESOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/config.h.in
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/initSofaGeneralImplicitOdeSolver.h
    )

set(SOURCE_FILES
    ${SOFAGENERALIMPLICITODESOLVER_SRC}/initSofaGeneralImplicitOdeSolver.cpp
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
