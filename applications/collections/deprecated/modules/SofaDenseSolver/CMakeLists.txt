cmake_minimum_required(VERSION 3.22)
project(SofaDenseSolver LANGUAGES CXX)

set(SOFADENSESOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFADENSESOLVER_SRC}/config.h.in
    ${SOFADENSESOLVER_SRC}/initSofaDenseSolver.h
    )

set(SOURCE_FILES
    ${SOFADENSESOLVER_SRC}/initSofaDenseSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.LinearSolver.Direct instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED) # SVDLinearSolver
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.LinearSolver.Direct)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaDenseSolver"
    RELOCATABLE "collections"
)
