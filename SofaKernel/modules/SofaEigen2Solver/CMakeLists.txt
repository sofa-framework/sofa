cmake_minimum_required(VERSION 3.1)
project(SofaEigen2Solver)

set(HEADER_FILES
    EigenBaseSparseMatrix.h
    EigenMatrixManipulator.h
    EigenSparseMatrix.h
    EigenVector.h
    EigenVectorWrapper.h
    SVDLinearSolver.h
    config.h
    initEigen2Solver.h
)

set(SOURCE_FILES
    EigenMatrixManipulator.cpp
    EigenVector.cpp
    SVDLinearSolver.cpp
    initEigen2Solver.cpp
)

find_package(Eigen3 REQUIRED)

if (SOFA_OPENMP AND "${Eigen3_VERSION}" VERSION_LESS 3.2.9)
    set(SOFAEIGEN2SOLVER_WITH_OPENMP 1 PARENT_SCOPE)
endif()

if (SOFAEIGEN2SOLVER_WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    list(APPEND HEADER_FILES EigenBaseSparseMatrix_MT.h)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_EIGEN2_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if (SOFAEIGEN2SOLVER_WITH_OPENMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

sofa_install_targets(SofaCommon ${PROJECT_NAME} "SofaCommon/${PROJECT_NAME}")
