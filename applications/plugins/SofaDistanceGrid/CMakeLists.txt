cmake_minimum_required(VERSION 3.1)
project(SofaDistanceGrid)

set(SOFADISTANCEGRID_VERSION 1.0)
set(SOFA_HAVE_SOFADISTANCEGRID 1)

find_package(SofaMiscCollision REQUIRED)

set(HEADER_FILES
    config.h
    initSofaDistanceGrid.h
    DistanceGrid.h
    components/collision/FFDDistanceGridDiscreteIntersection.h
    components/collision/FFDDistanceGridDiscreteIntersection.inl
    components/collision/RigidDistanceGridDiscreteIntersection.h
    components/collision/RigidDistanceGridDiscreteIntersection.inl
    components/collision/DistanceGridCollisionModel.h
    components/forcefield/DistanceGridForceField.h
    components/forcefield/DistanceGridForceField.inl
)

set(SOURCE_FILES
    initSofaDistanceGrid.cpp
    DistanceGrid.cpp
    RegisterModelToCollisionFactory.cpp
    components/forcefield/DistanceGridForceField.cpp
    components/collision/FFDDistanceGridDiscreteIntersection.cpp
    components/collision/RayDistanceGridContact.cpp
    components/collision/RigidDistanceGridDiscreteIntersection.cpp
    components/collision/DistanceGridCollisionModel.cpp
    components/collision/BarycentricPenalityContact_DistanceGrid.cpp
)

set(EXTRA_FILES
    README.md
    examples/README.md
    examples/DistanceGridCollisionModel.scn
    examples/FFDDistanceGridCollisionModel.scn
    examples/DistanceGridForceField-liver.scn
    )

add_subdirectory(SofaDistanceGrid_test)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaMiscCollision ${SOFAADVANCED_DEPENDENCY_LINK})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_HAVE_SOFADISTANCEGRID")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

find_package(MiniFlowVR QUIET)
if(MiniFlowVR_FOUND)
    add_definitions("-DSOFA_HAVE_MINIFLOWVR")
    target_link_libraries(${PROJECT_NAME} PUBLIC miniFlowVR)
    message(STATUS "SofaDistanceGrid: MiniFlowVR enabled.")
else()
    message(STATUS "SofaDistanceGrid: MiniFlowVR disabled. therefore some features will not be compiled.")
endif()


target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")


## Install rules for the library and headers; CMake package configurations files
sofa_create_package(SofaDistanceGrid ${SOFADISTANCEGRID_VERSION} ${PROJECT_NAME} SofaDistanceGrid)
