cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.ODESolver.Forward LANGUAGES CXX)

set(SOFACOMPONENTODESOLVERFORWARD_SOURCE_DIR "src/sofa/component/odesolver/forward")

set(SOFAMODULES_SOURCE_DIR "${SOFA_SOURCE_DIR}/modules")
set(SOFAKERNELMODULES_SOURCE_DIR "${SOFA_KERNEL_SOURCE_DIR}/modules")

set(SOFAEXPLICITODESOLVER_DIR "${SOFAKERNELMODULES_SOURCE_DIR}/SofaExplicitOdeSolver/")
set(SOFAEXPLICITODESOLVER_SRC "${SOFAEXPLICITODESOLVER_DIR}/src/SofaExplicitOdeSolver")

set(SOFAGENERALEXPLICITODESOLVER_DIR "${SOFAMODULES_SOURCE_DIR}/SofaGeneralExplicitOdeSolver/")
set(SOFAGENERALEXPLICITODESOLVER_SRC "${SOFAGENERALEXPLICITODESOLVER_DIR}/src/SofaGeneralExplicitOdeSolver")
set(SOFAMISCSOLVER_DIR "${SOFAMODULES_SOURCE_DIR}/SofaMiscSolver/")
set(SOFAMISCSOLVER_SRC "${SOFAMISCSOLVER_DIR}/src/SofaMiscSolver")

set(HEADER_FILES
    ${SOFACOMPONENTODESOLVERFORWARD_SOURCE_DIR}/config.h.in
    ${SOFAEXPLICITODESOLVER_SRC}/EulerSolver.h
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/CentralDifferenceSolver.h
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/RungeKutta2Solver.h
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/RungeKutta4Solver.h
    ${SOFAMISCSOLVER_SRC}/DampVelocitySolver.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTODESOLVERFORWARD_SOURCE_DIR}/init.cpp
    ${SOFAEXPLICITODESOLVER_SRC}/EulerSolver.cpp
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/CentralDifferenceSolver.cpp
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/RungeKutta2Solver.cpp
    ${SOFAGENERALEXPLICITODESOLVER_SRC}/RungeKutta4Solver.cpp
    ${SOFAMISCSOLVER_SRC}/DampVelocitySolver.cpp
)
set(WRAPPER_FILES
    wrapper/SofaExplicitOdeSolver/config.h
    wrapper/SofaGeneralExplicitOdeSolver/config.h
    wrapper/SofaMiscSolver/config.h
)
source_group("wrapper" FILES ${WRAPPER_FILES})

set(TARGET_NAME ${PROJECT_NAME})
sofa_find_package(SofaFramework REQUIRED)

configure_file(${SOFACOMPONENTODESOLVERFORWARD_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/include/sofa/component/odesolver/forward/config.h)

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${WRAPPER_FILES})
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${TARGET_NAME} PUBLIC SofaSimulationCore)

target_compile_definitions(${TARGET_NAME} PRIVATE BUILD_SOFACOMPONENTODESOLVEREXPLICIT)
target_compile_definitions(${TARGET_NAME} PRIVATE SOFA_BUILD_SOFAEXPLICITODESOLVER)
target_compile_definitions(${TARGET_NAME} PRIVATE SOFA_BUILD_SOFAGENERALEXPLICITODESOLVER)
target_compile_definitions(${TARGET_NAME} PRIVATE SOFA_BUILD_SOFAMISCSOLVER)
target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/")
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/wrapper/")
target_include_directories(${TARGET_NAME} PUBLIC "${SOFAEXPLICITODESOLVER_DIR}/src")
target_include_directories(${TARGET_NAME} PUBLIC "${SOFAGENERALEXPLICITODESOLVER_DIR}/src")
target_include_directories(${TARGET_NAME} PUBLIC "${SOFAMISCSOLVER_DIR}/src")
