cmake_minimum_required(VERSION 2.8)

project("SofaBaseCollision")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initBaseCollision.h 
    collision/BaseContactMapper.h 
    collision/DefaultPipeline.h 
    collision/Sphere.h 
    collision/SphereModel.h 
    collision/SphereModel.inl 
    collision/Cube.h 
    collision/CubeModel.h 
    collision/DiscreteIntersection.h 
    collision/DiscreteIntersection.inl 
    collision/BruteForceDetection.h 
    collision/DefaultContactManager.h 
    collision/MinProximityIntersection.h 
    collision/NewProximityIntersection.h 
    collision/NewProximityIntersection.inl 
    collision/CapsuleModel.h 
    collision/CapsuleModel.inl 
    collision/BaseProximityIntersection.h 
    collision/CapsuleIntTool.h 
    collision/OBBModel.inl 
    collision/OBBModel.h 
    collision/RigidSphereModel.h 
    collision/RigidSphereModel.inl 
    collision/OBBIntTool.h 
    collision/IntrOBBOBB.h 
    collision/IntrOBBOBB.inl 
    collision/IntrUtility3.h 
    collision/IntrUtility3.inl 
    collision/IntrSphereOBB.h 
    collision/IntrCapsuleOBB.h 
    collision/IntrCapsuleOBB.inl 
    collision/IntrSphereOBB.inl 
    collision/Intersector.h
    
    )
    
set(SOURCE_FILES

    initBaseCollision.cpp 
    collision/BaseContactMapper.cpp 
    collision/DefaultPipeline.cpp 
    collision/SphereModel.cpp 
    collision/CubeModel.cpp 
    collision/CapsuleModel.cpp 
    collision/DiscreteIntersection.cpp 
    collision/BruteForceDetection.cpp 
    collision/DefaultContactManager.cpp 
    collision/MinProximityIntersection.cpp 
    collision/NewProximityIntersection.cpp 
    collision/BaseProximityIntersection.cpp
    collision/CapsuleIntTool.cpp 
    collision/RigidSphereModel.cpp 
    collision/OBBModel.cpp 
    collision/OBBIntTool.cpp 
    collision/IntrOBBOBB.cpp 
    collision/IntrUtility3.cpp 
    collision/IntrCapsuleOBB.cpp 
    collision/IntrSphereOBB.cpp

    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaRidge MiniFlowVR SofaSphFluid)
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_BASE_COLLISION")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
