include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initBaseCollision.h
    ../collision/BaseContactMapper.h
    ../collision/DefaultPipeline.h
    ../collision/Cube.h
    ../collision/CubeModel.h
    ../collision/Sphere.h
    ../collision/SphereModel.h
    ../collision/SphereModel.inl
    ../collision/DiscreteIntersection.h
    ../collision/BruteForceDetection.h
    ../collision/DefaultContactManager.h
    ../collision/NewProximityIntersection.h
    ../collision/NewProximityIntersection.inl
    ../collision/BaseProximityIntersection.h
    ../collision/BaseIntTool.h        
    ../collision/IntrUtility3.h
    ../collision/IntrUtility3.inl
    )

set(SOURCE_FILES
    ../initBaseCollision.cpp
    ../collision/CubeModel.cpp
    ../collision/SphereModel.cpp
    ../collision/BaseContactMapper.cpp
    ../collision/DefaultPipeline.cpp
    ../collision/DiscreteIntersection.cpp
    ../collision/BruteForceDetection.cpp
    ../collision/DefaultContactManager.cpp
    ../collision/NewProximityIntersection.cpp
    ../collision/BaseProximityIntersection.cpp
    ../collision/BaseIntTool.cpp
    ../collision/IntrUtility3.cpp
    )


if(SOFA-FLAG_SOFAPRO MATCHES OFF)
    set(HEADER_FILES ${HEADER_FILES}
        ../collision/MinProximityIntersection.h
        ../collision/CapsuleModel.h
        ../collision/CapsuleModel.inl
        ../collision/RigidCapsuleModel.h
        ../collision/RigidCapsuleModel.inl
        ../collision/CylinderModel.h
        ../collision/CylinderModel.inl
        ../collision/CapsuleIntTool.h
        ../collision/CapsuleIntTool.inl
        ../collision/OBBModel.inl
        ../collision/OBBModel.h
        ../collision/OBBIntTool.h
        ../collision/IntrOBBOBB.h
        ../collision/IntrOBBOBB.inl
        ../collision/IntrSphereOBB.h
        ../collision/IntrCapsuleOBB.h
        ../collision/IntrCapsuleOBB.inl
        ../collision/IntrSphereOBB.inl
        ../collision/Intersector.h
        ../collision/ContactListener.h
    )

    set(SOURCE_FILES ${SOURCE_FILES}
        
        ../collision/CapsuleModel.cpp
        ../collision/CapsuleIntTool.cpp
        ../collision/OBBModel.cpp
        ../collision/RigidCapsuleModel.cpp
        ../collision/CylinderModel.cpp
        ../collision/OBBIntTool.cpp
        ../collision/IntrOBBOBB.cpp
        ../collision/IntrCapsuleOBB.cpp
        ../collision/IntrSphereOBB.cpp
        ../collision/ContactListener.cpp
        ../collision/MinProximityIntersection.cpp
    )

endif(SOFA-FLAG_SOFAPRO MATCHES OFF)

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_COLLISION")
AddLinkerDependencies(SofaBaseMechanics SofaRigid FlowVR SofaSphFluid)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
