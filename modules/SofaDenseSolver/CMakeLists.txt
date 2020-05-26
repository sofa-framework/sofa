cmake_minimum_required(VERSION 3.1)
project(SofaDenseSolver VERSION 1.0 LANGUAGES CXX)

sofa_find_package(Newmat REQUIRED BOTH_SCOPES)
sofa_find_package(SofaBase REQUIRED)
sofa_find_package(SofaGeneral REQUIRED)

set(SOFADENSESOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFADENSESOLVER_SRC}/config.h.in
    ${SOFADENSESOLVER_SRC}/initDenseSolver.h
    )

set(SOURCE_FILES
    ${SOFADENSESOLVER_SRC}/initDenseSolver.cpp
    )

list(APPEND HEADER_FILES
    ${SOFADENSESOLVER_SRC}/LULinearSolver.h
    ${SOFADENSESOLVER_SRC}/NewMatMatrix.h
    ${SOFADENSESOLVER_SRC}/NewMatVector.h
    )
list(APPEND SOURCE_FILES
    ${SOFADENSESOLVER_SRC}/LULinearSolver.cpp
    ${SOFADENSESOLVER_SRC}/NewMatCGLinearSolver.cpp
    ${SOFADENSESOLVER_SRC}/NewMatCholeskySolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver SofaGeneralLinearSolver)
target_link_libraries(${PROJECT_NAME} PUBLIC newmat)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaDenseSolver"
    RELOCATABLE "plugins"
)
