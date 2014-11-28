cmake_minimum_required(VERSION 2.8.11)
project(SofaBaseLinearSolver)

set(HEADER_FILES

    initBaseLinearSolver.h
    CGLinearSolver.h
    CGLinearSolver.inl
    CholeskySolver.h
    CholeskySolver.inl
    BTDLinearSolver.h
    BTDLinearSolver.inl
    FullVector.h
    FullMatrix.h
    DiagonalMatrix.h
    SingleMatrixAccessor.h
    SparseMatrix.h
    CompressedRowSparseMatrix.h
    CompressedRowSparseMatrix.inl
    GraphScatteredTypes.h
    DefaultMultiMatrixAccessor.h
    MatrixLinearSolver.h
    MatrixLinearSolver.inl
    MatrixExpr.h
    matrix_bloc_traits.h
    BlocMatrixWriter.h
    MinResLinearSolver.h
    MinResLinearSolver.inl

    )

set(SOURCE_FILES

    initBaseLinearSolver.cpp
    CGLinearSolver.cpp
    CholeskySolver.cpp
    BTDLinearSolver.cpp
    FullVector.cpp
    GraphScatteredTypes.cpp
    DefaultMultiMatrixAccessor.cpp
    MatrixLinearSolver.cpp
    SingleMatrixAccessor.cpp
    MinResLinearSolver.cpp

    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaSimulationCommon ) 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_DEPENDENCY_INCLUDE_DIR}>")

set( COMPILER_FLAGS "-DSOFA_BUILD_BASE_LINEAR_SOLVER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFABASE_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaBase_libraries
        EXPORT  SofaBaseTargets
        RUNTIME DESTINATION "${SOFABASE_INSTALL_BIN_DIR}" COMPONENT SofaBase_libraries
        LIBRARY DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        ARCHIVE DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        PUBLIC_HEADER DESTINATION "${SOFABASE_INSTALL_INCLUDE_DIR}/SofaBaseLinearSolver" COMPONENT SofaBase_headers
        )
