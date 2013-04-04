cmake_minimum_required(VERSION 2.8)

project("SofaMiscCollision")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initMiscCollision.h 
    collision/TriangleModelInRegularGrid.h 
    collision/TreeCollisionGroupManager.h 
    collision/RuleBasedContactManager.h 
    collision/DefaultCollisionGroupManager.h 
    collision/SolverMerger.h 
    collision/TetrahedronDiscreteIntersection.h 
    collision/SpatialGridPointModel.h 
    collision/TetrahedronModel.h 
    collision/BarycentricStickContact.h 
    collision/BarycentricStickContact.inl 

    )
    
set(SOURCE_FILES

    initMiscCollision.cpp 
	collision/FrictionContact_DistanceGrid.cpp 
    collision/TriangleModelInRegularGrid.cpp 
    collision/TreeCollisionGroupManager.cpp 
    collision/RuleBasedContactManager.cpp 
    collision/DefaultCollisionGroupManager.cpp 
    collision/SolverMerger.cpp 
	collision/TetrahedronDiscreteIntersection.cpp 
    collision/SpatialGridPointModel.cpp 
    collision/TetrahedronModel.cpp 
    collision/TetrahedronBarycentricPenalityContact.cpp 
    collision/TetrahedronRayContact.cpp 
    collision/TetrahedronFrictionContact.cpp 
    collision/BarycentricStickContact.cpp 
 
    )

if(SOFA_SMP)
	list(APPEND HEADER_FILES "collision/ParallelCollisionPipeline.h")
	list(APPEND SOURCE_FILES "collision/ParallelCollisionPipeline.cpp")
endif()
    
if(SIMULATION_GRAPH_BGL)
	list(APPEND HEADER_FILES "collision/BglCollisionGroupManager.h")
	list(APPEND SOURCE_FILES "collision/BglCollisionGroupManager.cpp")
endif()

if(EXTERNAL_HAVE_EIGEN2)
	list(APPEND SOURCE_FILES "collision/TetrahedronBarycentricDistanceLMConstraintContact.cpp")
	list(APPEND SOURCE_FILES "collision/BarycentricDistanceLMConstraintContact_DistanceGrid.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_MISC_COLLISION" )
set(LINKER_DEPENDENCIES SofaMeshCollision SofaConstraint SofaVolumetricData SofaExplicitOdeSolver SofaImplicitOdeSolver )

if(SIMULATION_GRAPH_BGL)
    list(APPEND LINKER_DEPENDENCIES SofaSimulationBGL)
endif()

#if(EXTERNAL_HAVE_EIGEN2)
#    list(APPEND LINKER_DEPENDENCIES eigen)
#endif()

include(${SOFA_CMAKE_DIR}/post.cmake)
