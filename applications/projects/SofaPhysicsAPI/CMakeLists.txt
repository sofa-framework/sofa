cmake_minimum_required(VERSION 3.12)
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
find_package(SofaValidation REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiMain SofaGeneral SceneCreator SofaValidation)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAPHYSICSAPI_VERSION})

## Install rules and CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME SofaPhysicsAPI
    PACKAGE_VERSION ${SOFAPHYSICSAPI_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR SofaPhysicsAPI
    )
