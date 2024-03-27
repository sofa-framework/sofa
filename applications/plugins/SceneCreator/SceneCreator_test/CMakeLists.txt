cmake_minimum_required(VERSION 3.22)

project(SceneCreator_test)

set(HEADER_FILES

)

set(SOURCE_FILES
    SceneCreator_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SceneCreator Sofa.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.SolidMechanics.FEM.Elastic Sofa.Component.StateContainer Sofa.Component.Topology.Container.Grid )


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
