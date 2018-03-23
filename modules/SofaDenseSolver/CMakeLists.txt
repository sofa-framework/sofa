cmake_minimum_required(VERSION 3.1)
project(SofaDenseSolver)

set(HEADER_FILES
    config.h
    initDenseSolver.h
    )

set(SOURCE_FILES
    initDenseSolver.cpp
    )

list(APPEND HEADER_FILES
    LULinearSolver.h
    NewMatMatrix.h
    NewMatVector.h
    )
list(APPEND SOURCE_FILES
    LULinearSolver.cpp
    NewMatCGLinearSolver.cpp
    NewMatCholeskySolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver)
target_link_libraries(${PROJECT_NAME} PUBLIC newmat)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_DENSE_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
