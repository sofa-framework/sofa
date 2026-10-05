cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.IntegrationScheme.Backward_test)

set(SOURCE_FILES
    BDFIntegrationScheme_test.cpp
    EulerImplicitIntegrationSchemeDynamic_test.cpp
    EulerImplicitIntegrationSchemeStatic_test.cpp
    EulerImplicitIntegrationScheme_withDamping_test.cpp
    NewmarkIntegrationSchemeDynamic_test.cpp
    StaticEquilibriumIntegrationScheme_test.cpp
    SpringIntegrationSchemeDynamic_test.cpp
)

add_definitions("-DSOFACOMPONENTINTEGRATIONSCHEMEBACKWARD_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# dependencies are managed directly in the target_link_libraries pass
target_link_libraries(${PROJECT_NAME} Sofa.Testing
    Sofa.Component.IntegrationScheme.Testing
    Sofa.Component.IntegrationScheme.Backward
    Sofa.Component.LinearSolver.Direct
    Sofa.Component.StateContainer)
