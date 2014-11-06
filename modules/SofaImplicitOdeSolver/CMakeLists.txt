cmake_minimum_required(VERSION 2.8)
project(SofaImplicitOdeSolver)

set(HEADER_FILES

    initImplicitODESolver.h
    EulerImplicitSolver.h
    StaticSolver.h

    )

set(SOURCE_FILES

    initImplicitODESolver.cpp
    EulerImplicitSolver.cpp
    StaticSolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} SofaSimulationTree)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAIMPLICITODESOLVER_COMPILER_FLAGS "-DSOFA_BUILD_IMPLICIT_ODE_SOLVER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAIMPLICITODESOLVER_COMPILER_FLAGS}"  )


install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaCommon_libraries
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION "${SOFACOMMON_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}"
        )

install(DIRECTORY ${SofaImplicitOdeSolver_SOURCE_DIR}/ DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/SofaImplicitOdeSolver"
        COMPONENT SofaCommon_headers
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )
