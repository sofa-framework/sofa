cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.IntegrationScheme.Backward LANGUAGES CXX)

set(SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR "src/sofa/component/integrationscheme/backward")

set(HEADER_FILES
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/init.h
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/EulerImplicitIntegrationScheme.h
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/NewmarkIntegrationScheme.h
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/LinearMultistepIntegrationScheme.h
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/BDFIntegrationScheme.h
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/StaticEquilibriumIntegrationScheme.h
)

set(DEPRECATED_HEADER_FILES
    compat/sofa/component/odesolver/backward/EulerImplicitSolver.h
    compat/sofa/component/odesolver/backward/NewmarkImplicitSolver.h
    compat/sofa/component/odesolver/backward/StaticSolver.h
    compat/sofa/component/odesolver/backward/BDFOdeSolver.h
    compat/sofa/component/odesolver/backward/BaseLinearMultiStepMethod.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/EulerImplicitIntegrationScheme.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/NewmarkIntegrationScheme.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/LinearMultistepIntegrationScheme.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/BDFIntegrationScheme.cpp
    ${SOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_SOURCE_DIR}/StaticEquilibriumIntegrationScheme.cpp
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

# CMake-level compat: redirect find_package(Sofa.Component.ODESolver.Backward) with an error
configure_package_config_file(
    compat/Sofa.Component.ODESolver.BackwardConfig.cmake.in
    "${CMAKE_BINARY_DIR}/lib/cmake/Sofa.Component.ODESolver.BackwardConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/Sofa.Component.ODESolver.Backward"
)
install(FILES
    "${CMAKE_BINARY_DIR}/lib/cmake/Sofa.Component.ODESolver.BackwardConfig.cmake"
    DESTINATION "lib/cmake/Sofa.Component.ODESolver.Backward"
    COMPONENT headers
)


