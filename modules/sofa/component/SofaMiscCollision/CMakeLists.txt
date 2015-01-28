include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMiscCollision.h 
    # ../collision/TriangleModelInRegularGrid.h
    # ../collision/RuleBasedContactManager.h 
#    ../collision/DefaultCollisionGroupManager.h
#     ../collision/SolverMerger.h
    # ../collision/TetrahedronDiscreteIntersection.h 
    # ../collision/SpatialGridPointModel.h 
    # ../collision/TetrahedronModel.h 
    # ../collision/BarycentricStickContact.h 
    # ../collision/BarycentricStickContact.inl 

    )
    
set(SOURCE_FILES

    ../initMiscCollision.cpp 
	# ../collision/FrictionContact_DistanceGrid.cpp 
 #    ../collision/TriangleModelInRegularGrid.cpp
 #    ../collision/RuleBasedContactManager.cpp 
#    ../collision/DefaultCollisionGroupManager.cpp
#     ../collision/SolverMerger.cpp
	# ../collision/TetrahedronDiscreteIntersection.cpp 
 #    ../collision/SpatialGridPointModel.cpp 
 #    ../collision/TetrahedronModel.cpp 
 #    ../collision/TetrahedronBarycentricPenalityContact.cpp 
 #    ../collision/TetrahedronRayContact.cpp 
 #    ../collision/TetrahedronFrictionContact.cpp 
 #    ../collision/BarycentricStickContact.cpp 
 
#	../collision/TetrahedronBarycentricDistanceLMConstraintContact.cpp # eigen2solver
#	../collision/BarycentricDistanceLMConstraintContact_DistanceGrid.cpp # eigen2solver
    )

if(SOFA-MISC_SMP)
	list(APPEND HEADER_FILES "../collision/ParallelCollisionPipeline.h")
	list(APPEND SOURCE_FILES "../collision/ParallelCollisionPipeline.cpp")
endif()
    

#if(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)
#	list(APPEND SOURCE_FILES "../collision/TetrahedronBarycentricDistanceLMConstraintContact.cpp")
#	list(APPEND SOURCE_FILES "../collision/BarycentricDistanceLMConstraintContact_DistanceGrid.cpp")
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
