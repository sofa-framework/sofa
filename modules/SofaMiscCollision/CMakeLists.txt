cmake_minimum_required(VERSION 2.8.12)
project(SofaMiscCollision)

set(HEADER_FILES
    BarycentricStickContact.h
    BarycentricStickContact.inl
    DefaultCollisionGroupManager.h
    RayTriangleVisitor.h
    RuleBasedContactManager.h
    SolverMerger.h
    SpatialGridPointModel.h
    TetrahedronDiscreteIntersection.h
    TetrahedronModel.h
    TriangleModelInRegularGrid.h
    config.h
    initMiscCollision.h
)

set(SOURCE_FILES
    BarycentricStickContact.cpp
    DefaultCollisionGroupManager.cpp
    RayTriangleVisitor.cpp
    RuleBasedContactManager.cpp
    SolverMerger.cpp
    SpatialGridPointModel.cpp
    TetrahedronBarycentricPenalityContact.cpp
    TetrahedronDiscreteIntersection.cpp
    TetrahedronFrictionContact.cpp
    TetrahedronModel.cpp
    TetrahedronRayContact.cpp
    TriangleModelInRegularGrid.cpp
    initMiscCollision.cpp
)

if(SOFA_HAVE_SMP)
    list(APPEND HEADER_FILES ParallelCollisionPipeline.h)
    list(APPEND SOURCE_FILES ParallelCollisionPipeline.cpp)
endif()

if(SOFA_HAVE_MINIFLOWVR)
    list(APPEND SOURCE_FILES FrictionContact_DistanceGrid.cpp)
endif()

list(APPEND SOURCE_FILES TetrahedronBarycentricDistanceLMConstraintContact.cpp)
if(SOFA_HAVE_MINIFLOWVR)
    list(APPEND SOURCE_FILES BarycentricDistanceLMConstraintContact_DistanceGrid.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaMeshCollision SofaConstraint SofaVolumetricData SofaExplicitOdeSolver SofaImplicitOdeSolver SofaSphFluid SofaOpenglVisual)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_COLLISION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
