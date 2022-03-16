cmake_minimum_required(VERSION 3.12)
project(SofaDenseSolver LANGUAGES CXX)

sofa_find_package(SofaBaseLinearSolver REQUIRED)

set(SOFADENSESOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFADENSESOLVER_SRC}/config.h.in
    ${SOFADENSESOLVER_SRC}/initSofaDenseSolver.h
    )

set(SOURCE_FILES
    ${SOFADENSESOLVER_SRC}/initSofaDenseSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaDenseSolver"
    RELOCATABLE "plugins"
)
