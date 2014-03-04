include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
    BarycentricStickContact.inl)

set(SOURCE_FILES
    initMiscCollision.cpp
    FrictionContact_DistanceGrid.cpp
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
    # eigen2solver
    TetrahedronBarycentricDistanceLMConstraintContact.cpp
    BarycentricDistanceLMConstraintContact_DistanceGrid.cpp)

if(SOFA-MISC_SMP)
    list(APPEND HEADER_FILES "ParallelCollisionPipeline.h")
    list(APPEND SOURCE_FILES "ParallelCollisionPipeline.cpp")
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_COLLISION")
AddSourceDependencies(eigen)
AddLinkerDependencies(MeshCollision Constraint VolumetricData ExplicitOdeSolver ImplicitOdeSolver Eigen2Solver)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

