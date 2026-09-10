cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.IntegrationScheme.Forward_test)

set(SOURCE_FILES
    CentralDifferenceIntegrationSchemeDynamic_test.cpp
    EulerExplicitIntegrationSchemeDynamic_test.cpp
    RungeKutta2IntegrationSchemeDynamic_test.cpp
    RungeKutta4IntegrationSchemeDynamic_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# dependencies are managed directly in the target_link_libraries pass
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.IntegrationScheme.Testing Sofa.Component.IntegrationScheme.Forward Sofa.Component.StateContainer)
