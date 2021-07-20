cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.LinearSystem_test)

set(SOURCE_FILES
    MatrixLinearSystem_test.cpp
    MappingGraph_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# dependencies are managed directly in the target_link_libraries pass
target_link_libraries(${PROJECT_NAME} Sofa.Testing
    Sofa.Component.LinearSystem
    Sofa.Component.StateContainer
    Sofa.Component.Mapping.Linear
    Sofa.Component.SolidMechanics.Spring
)
