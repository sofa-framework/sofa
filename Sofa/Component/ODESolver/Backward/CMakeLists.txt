cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.ODESolver.Backward LANGUAGES CXX)

set(SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR "src/sofa/component/odesolver/backward")

set(HEADER_FILES
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/init.h
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/EulerImplicitSolver.h
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/StaticSolver.h
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/VariationalSymplecticSolver.h
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/NewmarkImplicitSolver.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/EulerImplicitSolver.cpp
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/StaticSolver.cpp
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/VariationalSymplecticSolver.cpp
    ${SOFACOMPONENTODESOLVERBACKWARD_SOURCE_DIR}/NewmarkImplicitSolver.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

cmake_dependent_option(SOFA_COMPONENT_ODESOLVER_BACKWARD_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_ODESOLVER_BACKWARD_BUILD_TESTS)
    add_subdirectory(tests)
endif()
