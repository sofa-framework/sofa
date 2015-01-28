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
    ../collision/BruteForceDetection.h
    ../collision/DefaultContactManager.h
    # ../collision/MinProximityIntersection.h
    ../collision/NewProximityIntersection.h
    ../collision/NewProximityIntersection.inl
    ../collision/CapsuleModel.h
    ../collision/CapsuleModel.inl
    ../collision/BaseProximityIntersection.h
    ../collision/CapsuleIntTool.h
    ../collision/CapsuleIntTool.inl
    ../collision/OBBModel.inl
    ../collision/OBBModel.h
	../collision/RigidCapsuleModel.h
	../collision/RigidCapsuleModel.inl
	../collision/CylinderModel.h
	../collision/CylinderModel.inl
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
    ../collision/BaseIntTool.h
 #    ../collision/ContactListener.h
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
    # ../collision/MinProximityIntersection.cpp
    ../collision/NewProximityIntersection.cpp
    ../collision/BaseProximityIntersection.cpp
    ../collision/CapsuleIntTool.cpp
    ../collision/OBBModel.cpp
	../collision/RigidCapsuleModel.cpp
	../collision/CylinderModel.cpp
    ../collision/OBBIntTool.cpp
    ../collision/IntrOBBOBB.cpp
    ../collision/IntrUtility3.cpp
    ../collision/IntrCapsuleOBB.cpp
    ../collision/IntrSphereOBB.cpp
    ../collision/BaseIntTool.cpp
    ../collision/ContactListener.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_COLLISION")
AddLinkerDependencies(SofaBaseMechanics SofaRigid FlowVR SofaSphFluid)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
