include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initVolumetricData.h 
    # ../container/ImplicitSurfaceContainer.h 
    # ../container/InterpolatedImplicitSurface.h 
    # ../container/InterpolatedImplicitSurface.inl 
    # ../forcefield/DistanceGridForceField.h 
    # ../forcefield/DistanceGridForceField.inl 
    # ../mapping/ImplicitSurfaceMapping.h 
    # ../mapping/ImplicitSurfaceMapping.inl 
    # ../container/DistanceGrid.h 
    # ../collision/DistanceGridCollisionModel.h 
    # ../collision/RigidDistanceGridDiscreteIntersection.h 
    # ../collision/RigidDistanceGridDiscreteIntersection.inl 
    # ../collision/FFDDistanceGridDiscreteIntersection.h 
    # ../collision/FFDDistanceGridDiscreteIntersection.inl 

    )
    
set(SOURCE_FILES

    ../initVolumetricData.cpp 
    # ../container/ImplicitSurfaceContainer.cpp 
    # ../container/InterpolatedImplicitSurface.cpp 
    # ../forcefield/DistanceGridForceField.cpp 
    # ../mapping/ImplicitSurfaceMapping.cpp 
    # ../container/DistanceGrid.cpp 
    # ../collision/DistanceGridCollisionModel.cpp 
    # ../collision/RayDistanceGridContact.cpp 
    # ../collision/RigidDistanceGridDiscreteIntersection.cpp 
    # ../collision/FFDDistanceGridDiscreteIntersection.cpp 
    # ../collision/BarycentricPenalityContact_DistanceGrid.cpp 
 
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_VOLUMETRIC_DATA" )
AddLinkerDependencies(SofaBaseCollision SofaMeshCollision SofaUserInteraction FlowVR)
        
include(${SOFA_CMAKE_DIR}/postProject.cmake)
