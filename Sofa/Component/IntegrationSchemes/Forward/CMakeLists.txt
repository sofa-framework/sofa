cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.IntegrationSchemes.Forward LANGUAGES CXX)

set(SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR "src/sofa/component/integrationschemes/forward")

set(HEADER_FILES
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/init.h
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/RungeKutta2IntegrationScheme.h
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/RungeKutta4IntegrationScheme.h
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/EulerExplicitIntegrationScheme.h
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/DampVelocityIntegrationScheme.h
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/CentralDifferenceIntegrationScheme.h
)

set(DEPRECATED_HEADER_FILES
    compat/sofa/component/odesolver/forward/EulerExplicitSolver.h
    compat/sofa/component/odesolver/forward/CentralDifferenceSolver.h
    compat/sofa/component/odesolver/forward/DampVelocitySolver.h
    compat/sofa/component/odesolver/forward/RungeKutta2Solver.h
    compat/sofa/component/odesolver/forward/RungeKutta4Solver.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/RungeKutta2IntegrationScheme.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/RungeKutta4IntegrationScheme.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/EulerExplicitIntegrationScheme.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/DampVelocityIntegrationScheme.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMESFORWARD_SOURCE_DIR}/CentralDifferenceIntegrationScheme.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${DEPRECATED_HEADER_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compat>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}_compat>
)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

install(DIRECTORY compat/ DESTINATION include/${PROJECT_NAME}_compat COMPONENT headers)

# CMake-level compat: redirect find_package(Sofa.Component.ODESolver.Forward) with an error
configure_package_config_file(
    Sofa.Component.ODESolver.ForwardConfig.cmake.in
    "${CMAKE_BINARY_DIR}/lib/cmake/Sofa.Component.ODESolver.Forward/Sofa.Component.ODESolver.ForwardConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/Sofa.Component.ODESolver.Forward"
)
install(FILES
    "${CMAKE_BINARY_DIR}/lib/cmake/Sofa.Component.ODESolver.Forward/Sofa.Component.ODESolver.ForwardConfig.cmake"
    DESTINATION "lib/cmake/Sofa.Component.ODESolver.Forward"
    COMPONENT headers
)

