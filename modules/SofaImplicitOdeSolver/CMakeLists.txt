cmake_minimum_required(VERSION 2.8.12)
project(SofaImplicitOdeSolver)

set(HEADER_FILES

    config.h
    initImplicitODESolver.h
    EulerImplicitSolver.h
    StaticSolver.h
    VariationalSymplecticSolver.h
    )

set(SOURCE_FILES

    initImplicitODESolver.cpp
    EulerImplicitSolver.cpp
    StaticSolver.cpp
    VariationalSymplecticSolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} PUBLIC SofaSimulationTree)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/SofaCommon>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAIMPLICITODESOLVER_COMPILER_FLAGS "-DSOFA_BUILD_IMPLICIT_ODE_SOLVER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAIMPLICITODESOLVER_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION bin COMPONENT SofaCommon_libraries
        LIBRARY DESTINATION lib COMPONENT SofaCommon_libraries
        ARCHIVE DESTINATION lib COMPONENT SofaCommon_libraries
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}" COMPONENT SofaCommon_headers
        )
