cmake_minimum_required(VERSION 3.12)
project(SofaPhysicsAPI)

set(SOFAPHYSICSAPI_VERSION 0.1)

find_package(GLUT REQUIRED)

set(HEADER_FILES
    SofaPhysicsAPI.h
    SofaPhysicsDataController_impl.h
    SofaPhysicsDataMonitor_impl.h
    SofaPhysicsOutputMesh_impl.h
    SofaPhysicsSimulation.h
    fakegui.h
)

set(SOURCE_FILES
    SofaPhysicsDataController.cpp
    SofaPhysicsDataMonitor.cpp
    SofaPhysicsOutputMesh.cpp
    SofaPhysicsSimulation.cpp
    fakegui.cpp
    # main.cpp
)

find_package(SofaGui REQUIRED)

# Are all these modules (formerly SofaGeneral) necessary?
find_package(SofaBoundaryCondition REQUIRED)
find_package(SofaConstraint REQUIRED)
find_package(SofaGeneralAnimationLoop REQUIRED)
find_package(SofaGeneralDeformable REQUIRED)
find_package(SofaGeneralEngine REQUIRED)
find_package(SofaGeneralExplicitOdeSolver REQUIRED)
find_package(SofaGeneralImplicitOdeSolver REQUIRED)
find_package(SofaGeneralLinearSolver REQUIRED)
find_package(SofaGeneralLoader REQUIRED)
find_package(SofaGeneralMeshCollision REQUIRED)
find_package(SofaGeneralObjectInteraction REQUIRED)
find_package(SofaGeneralRigid REQUIRED)
find_package(SofaGeneralSimpleFem REQUIRED)
find_package(SofaGeneralTopology REQUIRED)
find_package(SofaGeneralVisual REQUIRED)
find_package(SofaGraphComponent REQUIRED)
find_package(SofaTopologyMapping REQUIRED)
find_package(SofaUserInteraction REQUIRED)

find_package(SceneCreator REQUIRED)
find_package(SofaValidation REQUIRED)
sofa_find_package(SofaSimulationCommon REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGui SceneCreator SofaValidation SofaSimulationCommon)
target_link_libraries(${PROJECT_NAME} PUBLIC
    SofaBoundaryCondition
    SofaConstraint
    SofaGeneralAnimationLoop
    SofaGeneralDeformable
    SofaGeneralEngine
    SofaGeneralExplicitOdeSolver
    SofaGeneralImplicitOdeSolver
    SofaGeneralLinearSolver
    SofaGeneralLoader
    SofaGeneralMeshCollision
    SofaGeneralObjectInteraction
    SofaGeneralRigid
    SofaGeneralSimpleFem
    SofaGeneralTopology
    SofaGeneralVisual
    SofaGraphComponent
    SofaTopologyMapping
    SofaUserInteraction
    )
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAPHYSICSAPI_VERSION})

## Install rules and CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME SofaPhysicsAPI
    PACKAGE_VERSION ${SOFAPHYSICSAPI_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR SofaPhysicsAPI
    )
