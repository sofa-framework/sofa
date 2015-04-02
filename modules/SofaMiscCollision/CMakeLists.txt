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
    BarycentricStickContact.inl
    RayTriangleVisitor.h

    )

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
    RayTriangleVisitor.cpp

	TetrahedronBarycentricDistanceLMConstraintContact.cpp # eigen2solver
	BarycentricDistanceLMConstraintContact_DistanceGrid.cpp # eigen2solver
    )

if(SOFA-MISC_SMP)
	list(APPEND HEADER_FILES "ParallelCollisionPipeline.h")
	list(APPEND SOURCE_FILES "ParallelCollisionPipeline.cpp")
endif()


#if(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)
#	list(APPEND SOURCE_FILES "TetrahedronBarycentricDistanceLMConstraintContact.cpp")
#	list(APPEND SOURCE_FILES "BarycentricDistanceLMConstraintContact_DistanceGrid.cpp")
#endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_COLLISION" )
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaMeshCollision SofaConstraint SofaVolumetricData SofaExplicitOdeSolver SofaImplicitOdeSolver SofaEigen2Solver)

#if(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)
#    AddLinkerDependencies(SofaEigen2Solver)
#endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
