cmake_minimum_required(VERSION 2.8.12)
project(SofaMiscCollision)

set(HEADER_FILES
    BarycentricStickContact.h
    BarycentricStickContact.inl
    DefaultCollisionGroupManager.h
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

if(NOT SOFA_NO_OPENGL)
    list(APPEND HEADER_FILES RayTriangleVisitor.h)
    list(APPEND SOURCE_FILES RayTriangleVisitor.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaMeshCollision SofaConstraint SofaVolumetricData SofaExplicitOdeSolver SofaImplicitOdeSolver SofaSphFluid)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)
if(NOT SOFA_NO_OPENGL)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaOpenglVisual)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_COLLISION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
