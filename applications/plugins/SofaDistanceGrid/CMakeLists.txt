cmake_minimum_required(VERSION 3.1)
project(SofaDistanceGrid VERSION 1.0)

find_package(SofaMiscCollision REQUIRED)
sofa_find_package(MiniFlowVR QUIET)

set(HEADER_FILES
    config.h.in
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

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaMiscCollision)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SOFADISTANCEGRID")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if(MiniFlowVR_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE miniFlowVR) # Private because not exported in API
    message(STATUS "SofaDistanceGrid: MiniFlowVR found.")
else()
    message(STATUS "SofaDistanceGrid: MiniFlowVR not found, some features will not be compiled.")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(SofaDistanceGrid_test)
    endif()
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
