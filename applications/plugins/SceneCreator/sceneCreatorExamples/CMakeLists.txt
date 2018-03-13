cmake_minimum_required(VERSION 3.1)

project(sceneCreatorExamples)

find_package(SofaGui)
find_package(SceneCreator)
find_package(SofaComponentBase)
find_package(SofaComponentCommon)

SET( HEADERS

    )

set( SOURCES
    SceneCreatorBenchmarks.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} SofaGuiMain SceneCreator SofaComponentBase SofaComponentCommon)
