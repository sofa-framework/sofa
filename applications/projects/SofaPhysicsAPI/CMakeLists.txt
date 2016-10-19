cmake_minimum_required(VERSION 3.1)
project(SofaPhysicsAPI)

set(SOFAPHYSICSAPI_VERSION 0.1)

# \todo {@epernod remove that define when understand how to replace this}
SET(USE_OGL_TETRA_MODEL false)


set(HEADER_FILES
    SofaPhysicsAPI.h
    SofaPhysicsDataController_impl.h
    SofaPhysicsDataMonitor_impl.h
    SofaPhysicsOutputMesh_impl.h
    SofaPhysicsSimulation_impl.h
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

IF(USE_OGL_TETRA_MODEL)
add_definitions(-DUSE_OGL_TETRA_MODEL)

set(HEADER_FILES ${HEADER_FILES}
    SofaPhysicsOutputMesh_Tetrahedron_impl.h
)

set(SOURCE_FILES ${SOURCE_FILES}
    SofaPhysicsOutputMesh_Tetrahedron.cpp
)
ENDIF()

find_package(SofaGui REQUIRED)
find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiMain SofaComponentGeneral)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAPHYSICSAPI_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAPHYSICSAPI")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

## Install rules and CMake package configurations files
sofa_create_package(SofaPhysicsAPI ${SOFAPHYSICSAPI_VERSION} ${PROJECT_NAME} include/SofaPhysicsAPI)
