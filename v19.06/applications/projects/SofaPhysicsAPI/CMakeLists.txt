cmake_minimum_required(VERSION 3.1)
project(SofaPhysicsAPI)

set(SOFAPHYSICSAPI_VERSION 0.1)

find_package(GLUT REQUIRED)

set(HEADER_FILES
    SofaPhysicsAPI.h
    SofaPhysicsDataController_impl.h
    SofaPhysicsDataMonitor_impl.h
    SofaPhysicsOutputMesh_impl.h
    SofaPhysicsSimulation.h
    fakegui.h
)

set(SOURCE_FILES
    SofaPhysicsDataController.cpp
    SofaPhysicsDataMonitor.cpp
    SofaPhysicsOutputMesh.cpp
    SofaPhysicsSimulation.cpp
    fakegui.cpp
    # main.cpp
)

find_package(SofaGui REQUIRED)
find_package(SofaGeneral REQUIRED)
find_package(SceneCreator REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiMain SofaComponentGeneral SceneCreator)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAPHYSICSAPI_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAPHYSICSAPI")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

## Install rules and CMake package configurations files
sofa_create_package(SofaPhysicsAPI ${SOFAPHYSICSAPI_VERSION} ${PROJECT_NAME} SofaPhysicsAPI)
