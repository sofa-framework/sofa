include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initBaseCollision.h
    BaseContactMapper.h
    DefaultPipeline.h
    Sphere.h
    SphereModel.h
    SphereModel.inl
    Cube.h
    CubeModel.h
    DiscreteIntersection.h
    BruteForceDetection.h
    DefaultContactManager.h
    MinProximityIntersection.h
    NewProximityIntersection.h
    NewProximityIntersection.inl
    CapsuleModel.h
    CapsuleModel.inl
    BaseProximityIntersection.h
    CapsuleIntTool.h
    CapsuleIntTool.inl
    OBBModel.inl
    OBBModel.h
	RigidCapsuleModel.h
	RigidCapsuleModel.inl
	CylinderModel.h
	CylinderModel.inl
    OBBIntTool.h
    IntrOBBOBB.h
    IntrOBBOBB.inl
    IntrUtility3.h
    IntrUtility3.inl
    IntrSphereOBB.h
    IntrCapsuleOBB.h
    IntrCapsuleOBB.inl
    IntrSphereOBB.inl
    Intersector.h
    BaseIntTool.h
    ContactListener.h
    )

set(SOURCE_FILES

    initBaseCollision.cpp
    BaseContactMapper.cpp
    DefaultPipeline.cpp
    SphereModel.cpp
    CubeModel.cpp
    CapsuleModel.cpp
    DiscreteIntersection.cpp
    BruteForceDetection.cpp
    DefaultContactManager.cpp
    MinProximityIntersection.cpp
    NewProximityIntersection.cpp
    BaseProximityIntersection.cpp
    CapsuleIntTool.cpp
    OBBModel.cpp
	RigidCapsuleModel.cpp
	CylinderModel.cpp
    OBBIntTool.cpp
    IntrOBBOBB.cpp
    IntrUtility3.cpp
    IntrCapsuleOBB.cpp
    IntrSphereOBB.cpp
    BaseIntTool.cpp
    ContactListener.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_COLLISION")
AddLinkerDependencies(SofaBaseMechanics SofaRigid FlowVR SofaSphFluid)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
