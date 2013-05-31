include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initBaseCollision.h 
    ../collision/BaseContactMapper.h 
    ../collision/DefaultPipeline.h 
    ../collision/Sphere.h 
    ../collision/SphereModel.h 
    ../collision/SphereModel.inl 
    ../collision/Cube.h 
    ../collision/CubeModel.h 
    ../collision/DiscreteIntersection.h 
    ../collision/DiscreteIntersection.inl 
    ../collision/BruteForceDetection.h 
    ../collision/DefaultContactManager.h 
    ../collision/MinProximityIntersection.h 
    ../collision/NewProximityIntersection.h 
    ../collision/NewProximityIntersection.inl 
    ../collision/CapsuleModel.h 
    ../collision/CapsuleModel.inl 
    ../collision/BaseProximityIntersection.h 
    ../collision/CapsuleIntTool.h 
    ../collision/OBBModel.inl 
    ../collision/OBBModel.h 
    ../collision/OBBIntTool.h 
    ../collision/IntrOBBOBB.h 
    ../collision/IntrOBBOBB.inl 
    ../collision/IntrUtility3.h 
    ../collision/IntrUtility3.inl 
    ../collision/IntrSphereOBB.h 
    ../collision/IntrCapsuleOBB.h 
    ../collision/IntrCapsuleOBB.inl 
    ../collision/IntrSphereOBB.inl 
    ../collision/Intersector.h
    
    )
    
set(SOURCE_FILES

    ../initBaseCollision.cpp 
    ../collision/BaseContactMapper.cpp 
    ../collision/DefaultPipeline.cpp 
    ../collision/SphereModel.cpp 
    ../collision/CubeModel.cpp 
    ../collision/CapsuleModel.cpp 
    ../collision/DiscreteIntersection.cpp 
    ../collision/BruteForceDetection.cpp 
    ../collision/DefaultContactManager.cpp 
    ../collision/MinProximityIntersection.cpp 
    ../collision/NewProximityIntersection.cpp 
    ../collision/BaseProximityIntersection.cpp
    ../collision/CapsuleIntTool.cpp 
    ../collision/OBBModel.cpp 
    ../collision/OBBIntTool.cpp 
    ../collision/IntrOBBOBB.cpp 
    ../collision/IntrUtility3.cpp 
    ../collision/IntrCapsuleOBB.cpp 
    ../collision/IntrSphereOBB.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_COLLISION")
AddLinkerDependencies(SofaBaseMechanics SofaRigid miniFlowVR SofaSphFluid)
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
