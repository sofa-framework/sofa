cmake_minimum_required(VERSION 3.22)

project(sceneCreatorExamples)

sofa_find_package(Sofa.GUI.Common REQUIRED)
sofa_find_package(SceneCreator REQUIRED)

set(HEADERS
    )
set(SOURCES
    SceneCreatorBenchmarks.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} SceneCreator Sofa.GUI.Common)
