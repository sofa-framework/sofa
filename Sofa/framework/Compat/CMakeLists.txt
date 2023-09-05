cmake_minimum_required(VERSION 3.12)
project(Sofa.Compat LANGUAGES CXX)

set(SOFACOMPATSRC_ROOT "src")

set(HEADER_FILES
    # ${SOFACOMPATSRC_ROOT}/config.h.in 
    ${SOFACOMPATSRC_ROOT}/sofa/helper/system/atomic.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Color.h
    ${SOFACOMPATSRC_ROOT}/SofaBaseLinearSolver/BlocMatrixWriter.h
    ${SOFACOMPATSRC_ROOT}/SofaBaseLinearSolver/DefaultMultiMatrixAccessor.h
    ${SOFACOMPATSRC_ROOT}/SofaBaseLinearSolver/CRSMultiMatrixAccessor.h
    ${SOFACOMPATSRC_ROOT}/SofaBaseLinearSolver/SingleMatrixAccessor.h
    ${SOFACOMPATSRC_ROOT}/SofaBaseUtils/RequiredPlugin.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/BaseIntTool.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/CapsuleContactMapper.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/CapsuleIntersection.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/CapsuleIntTool.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/CapsuleIntTool.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/CapsuleModel.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/CapsuleModel.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrCapsuleOBB.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrCapsuleOBB.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrMeshUtility.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrMeshUtility.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrOBBOBB.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrOBBOBB.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrSphereOBB.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrSphereOBB.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrTriangleOBB.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrTriangleOBB.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrUtility3.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/IntrUtility3.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/MeshIntTool.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/MeshIntTool.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/OBBContactMapper.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/OBBIntersection.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/OBBIntTool.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/OBBModel.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/OBBModel.inl
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/RigidCapsuleModel.h
    ${SOFACOMPATSRC_ROOT}/SofaMiscCollision/RigidCapsuleModel.inl
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/FindByTypeVisitor.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/SceneLoaderPHP.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/SceneLoaderXML.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/TransformationVisitor.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/AttributeElement.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/BaseElement.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/BaseMultiMappingElement.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/DataElement.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/Element.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/Element.inl
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/ElementNameHelper.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/NodeElement.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/ObjectElement.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationCommon/xml/XML.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationGraph/DAGNode.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationGraph/DAGSimulation.h
    ${SOFACOMPATSRC_ROOT}/SofaSimulationGraph/SimpleApi.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheck.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheckAPIChange.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheckCollisionResponse.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheckDeprecatedComponents.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheckDuplicatedName.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheckMissingRequiredPlugin.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheckUsingAlias.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheckerListener.h
    ${SOFACOMPATSRC_ROOT}/SofaGraphComponent/SceneCheckerVisitor.h
)

set(SOURCE_FILES
    ${SOFACOMPATSRC_ROOT}/sofa/initSofaCompat.cpp
)

sofa_find_package(Sofa.Config REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    OPTIMIZE_BUILD_DIR 0
)
