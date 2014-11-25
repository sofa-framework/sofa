cmake_minimum_required(VERSION 2.8.11)
project(SofaVolumetricData)

set(HEADER_FILES

    initVolumetricData.h
    ImplicitSurfaceContainer.h
    InterpolatedImplicitSurface.h
    InterpolatedImplicitSurface.inl
    ImplicitSurfaceMapping.h
    ImplicitSurfaceMapping.inl

    )

set(SOURCE_FILES

    initVolumetricData.cpp
    ImplicitSurfaceContainer.cpp
    InterpolatedImplicitSurface.cpp
    ImplicitSurfaceMapping.cpp

    )

if(SOFA_HAVE_MINIFLOWVR)

    list(APPEND HEADER_FILES

        DistanceGridForceField.h
        DistanceGridForceField.inl
        DistanceGrid.h
        DistanceGridCollisionModel.h
        RigidDistanceGridDiscreteIntersection.h
        RigidDistanceGridDiscreteIntersection.inl
        FFDDistanceGridDiscreteIntersection.h
        FFDDistanceGridDiscreteIntersection.inl

        )

    list(APPEND SOURCE_FILES

        DistanceGridForceField.cpp
        DistanceGrid.cpp
        DistanceGridCollisionModel.cpp
        RayDistanceGridContact.cpp
        RigidDistanceGridDiscreteIntersection.cpp
        FFDDistanceGridDiscreteIntersection.cpp
        BarycentricPenalityContact_DistanceGrid.cpp

        )
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} SofaMeshCollision ${SOFAADVANCED_DEPENDENCY_LINK} ) 

if( SOFA_HAVE_MINIFLOWVR )
    target_link_libraries( ${PROJECT_NAME} SofaUserInteraction )
endif() 

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAADVANCED_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAADVANCED_DEPENDENCY_DIR}>")

set( COMPILER_FLAGS "-DSOFA_BUILD_VOLUMETRIC_DATA")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAADVANCED_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaAdvanced_libraries
        EXPORT  SofaAdvancedTargets
        RUNTIME DESTINATION "${SOFAADVANCED_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAADVANCED_INSTALL_INCLUDE_DIR}/SofaVolumetricData"
        )
