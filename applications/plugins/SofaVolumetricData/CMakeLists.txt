cmake_minimum_required(VERSION 3.1)
project(SofaVolumetricData)

set(SOFAVOLUMETRICDATA_VERSION 1.0)

find_package(SofaDistanceGrid REQUIRED)
find_package(SofaImplicitField REQUIRED)

set(HEADER_FILES
    config.h
    initVolumetricData.h
    DistanceGrid.h
    FFDDistanceGridDiscreteIntersection.h
    FFDDistanceGridDiscreteIntersection.inl
    RigidDistanceGridDiscreteIntersection.h
    RigidDistanceGridDiscreteIntersection.inl
    DistanceGridCollisionModel.h
    DistanceGridForceField.h
    DistanceGridForceField.inl
    ImplicitSurfaceContainer.h
    ImplicitSurfaceMapping.h
    ImplicitSurfaceMapping.inl
    InterpolatedImplicitSurface.h
    InterpolatedImplicitSurface.inl
)

set(SOURCE_FILES
    initVolumetricData.cpp
)

configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/SofaVolumetricData/config.h")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision ${SOFAADVANCED_DEPENDENCY_LINK})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction)

if(SOFA_HAVE_MINIFLOWVR)
    target_link_libraries(${PROJECT_NAME} PUBLIC miniFlowVR)
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VOLUMETRIC_DATA -DSOFA_HAVE_MINIFLOWVR")
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VOLUMETRIC_DATA")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(SofaVolumetricData ${SOFAVOLUMETRICDATA_VERSION} ${PROJECT_NAME} SofaVolumetricData)
