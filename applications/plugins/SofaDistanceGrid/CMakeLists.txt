cmake_minimum_required(VERSION 3.22)
project(SofaDistanceGrid LANGUAGES CXX)

option(SOFADISTANCEGRID_USE_MINIFLOWVR "Build and use the miniFlowVR library with DistanceGrid." OFF)
if(SOFADISTANCEGRID_USE_MINIFLOWVR)
    add_subdirectory(extlibs/miniFlowVR)
    set_target_properties(miniFlowVR PROPERTIES DEBUG_POSTFIX "_d")
    #set_target_properties(miniFlowVR PROPERTIES FOLDER "SofaExtlibs")
endif()

option(SOFADISTANCEGRID_USE_MESHCONV "Build meshconv, mainly used to generate DistanceGrid file. Needs miniFlowVR." OFF)
if(SOFADISTANCEGRID_USE_MESHCONV)
    find_package(MiniFlowVR QUIET)
    if(MiniFlowVR_FOUND)
        sofa_add_subdirectory(application applications/meshconv meshconv OFF)
    else()
        message(WARNING "meshconv needs miniFlowVR; enable SOFADISTANCEGRID_USE_MINIFLOWVR to build it.")
    endif()
endif()

set(SOFADISTANCEGRID_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFADISTANCEGRID_SRC}/config.h.in
    ${SOFADISTANCEGRID_SRC}/initSofaDistanceGrid.h
    ${SOFADISTANCEGRID_SRC}/DistanceGrid.h
    ${SOFADISTANCEGRID_SRC}/components/collision/FFDDistanceGridDiscreteIntersection.h
    ${SOFADISTANCEGRID_SRC}/components/collision/FFDDistanceGridDiscreteIntersection.inl
    ${SOFADISTANCEGRID_SRC}/components/collision/RigidDistanceGridDiscreteIntersection.h
    ${SOFADISTANCEGRID_SRC}/components/collision/RigidDistanceGridDiscreteIntersection.inl
    ${SOFADISTANCEGRID_SRC}/components/collision/DistanceGridCollisionModel.h
    ${SOFADISTANCEGRID_SRC}/components/forcefield/DistanceGridForceField.h
    ${SOFADISTANCEGRID_SRC}/components/forcefield/DistanceGridForceField.inl
)

set(SOURCE_FILES
    ${SOFADISTANCEGRID_SRC}/initSofaDistanceGrid.cpp
    ${SOFADISTANCEGRID_SRC}/DistanceGrid.cpp
    ${SOFADISTANCEGRID_SRC}/RegisterModelToCollisionFactory.cpp
    ${SOFADISTANCEGRID_SRC}/components/forcefield/DistanceGridForceField.cpp
    ${SOFADISTANCEGRID_SRC}/components/collision/FFDDistanceGridDiscreteIntersection.cpp
    ${SOFADISTANCEGRID_SRC}/components/collision/RayDistanceGridContact.cpp
    ${SOFADISTANCEGRID_SRC}/components/collision/RigidDistanceGridDiscreteIntersection.cpp
    ${SOFADISTANCEGRID_SRC}/components/collision/DistanceGridCollisionModel.cpp
    ${SOFADISTANCEGRID_SRC}/components/collision/BarycentricPenalityContact_DistanceGrid.cpp
)

set(EXTRA_FILES
    README.md
    examples/README.md
    examples/DistanceGridCollisionModel.scn
    examples/FFDDistanceGridCollisionModel.scn
    examples/DistanceGridForceField-liver.scn
    )


find_package(MiniFlowVR QUIET)
sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Collision REQUIRED)
sofa_find_package(Sofa.GL QUIET)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision)

if(Sofa.GL_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL)
endif()

if(MiniFlowVR_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE miniFlowVR) # Private because not exported in API
    message(STATUS "SofaDistanceGrid: MiniFlowVR found.")
else()
    message(STATUS "SofaDistanceGrid: MiniFlowVR not found, some features will not be compiled.")
endif()

if(SOFA_BUILD_TESTS)
    add_subdirectory(SofaDistanceGrid_test)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )

sofa_add_subdirectory(plugin extensions/CUDA SofaDistanceGrid.CUDA)
