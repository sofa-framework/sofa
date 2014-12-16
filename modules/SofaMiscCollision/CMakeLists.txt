cmake_minimum_required(VERSION 2.8)
project(SofaMiscCollision)

set(HEADER_FILES

    initMiscCollision.h
    TriangleModelInRegularGrid.h
    RuleBasedContactManager.h
    DefaultCollisionGroupManager.h
    SolverMerger.h
    TetrahedronDiscreteIntersection.h
    SpatialGridPointModel.h
    TetrahedronModel.h
    BarycentricStickContact.h
    BarycentricStickContact.inl
    RayTriangleVisitor.h

    )

set(SOURCE_FILES

    initMiscCollision.cpp
    TriangleModelInRegularGrid.cpp
    RuleBasedContactManager.cpp
    DefaultCollisionGroupManager.cpp
    SolverMerger.cpp
	TetrahedronDiscreteIntersection.cpp
    SpatialGridPointModel.cpp
    TetrahedronModel.cpp
    TetrahedronBarycentricPenalityContact.cpp
    TetrahedronRayContact.cpp
    TetrahedronFrictionContact.cpp
    BarycentricStickContact.cpp
    RayTriangleVisitor.cpp

    )

if(SOFA_HAVE_SMP)
	list(APPEND HEADER_FILES "ParallelCollisionPipeline.h")
	list(APPEND SOURCE_FILES "ParallelCollisionPipeline.cpp")
endif()

if(SOFA_HAVE_MINIFLOWVR)
    list(APPEND SOURCE_FILES "FrictionContact_DistanceGrid.cpp")
endif()

if(SOFA_HAVE_EIGEN2)
	list(APPEND SOURCE_FILES "TetrahedronBarycentricDistanceLMConstraintContact.cpp")
if(SOFA_HAVE_MINIFLOWVR)
	list(APPEND SOURCE_FILES "BarycentricDistanceLMConstraintContact_DistanceGrid.cpp")
endif()
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseTopology SofaMeshCollision SofaConstraint SofaVolumetricData SofaExplicitOdeSolver SofaImplicitOdeSolver SofaSphFluid SofaOpenglVisual)
#if(SOFA_HAVE_EIGEN2)
#    target_link_libraries(${PROJECT_NAME} SofaEigen2Solver)
#endif()
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_DEPENDENCY_DIR}>")

set( SOFAMISCCOLLISION_COMPILER_FLAGS "-DSOFA_BUILD_MISC_COLLISION")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAMISC_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAMISCCOLLISION_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaMiscCollision_libraries
        EXPORT    SofaMiscTargets
        RUNTIME DESTINATION "${SOFAMISC_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAMISC_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
)
