cmake_minimum_required(VERSION 3.1)
project(SceneCreator VERSION 0.1)

set(HEADER_FILES
    src/SceneCreator/GetAssembledSizeVisitor.h
    src/SceneCreator/GetVectorVisitor.h
    src/SceneCreator/SceneCreator.h
    src/SceneCreator/SceneUtils.h
)

set(SOURCE_FILES
    src/SceneCreator/GetAssembledSizeVisitor.cpp
    src/SceneCreator/GetVectorVisitor.cpp
    src/SceneCreator/SceneCreator.cpp
    src/SceneCreator/SceneUtils.cpp
    src/SceneCreator/initSceneCreator.cpp
)

find_package(SofaSimulation REQUIRED)
find_package(Metis QUIET)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph)
target_compile_definitions(${PROJECT_NAME} PRIVATE  "-DSOFA_BUILD_SCENECREATOR")
target_compile_definitions(${PROJECT_NAME} PUBLIC  "-DSOFA_HAVE_SCENECREATOR")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

if(Metis_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_HAVE_METIS")
endif()

## Install rules for the library; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    INCLUDE_SOURCE_DIR "src/SceneCreator"
    INCLUDE_INSTALL_DIR "SceneCreator"
    )

## Add example project to use sceneCreator
add_subdirectory(sceneCreatorExamples)

## Add test project
if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    add_subdirectory(SceneCreator_test)
endif()
