cmake_minimum_required(VERSION 3.12)
project(SofaUserInteraction LANGUAGES CXX)

set(SOFAUSERINTERACTION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAUSERINTERACTION_SRC}/config.h.in
    ${SOFAUSERINTERACTION_SRC}/initSofaUserInteraction.h
    )

set(SOURCE_FILES
    ${SOFAUSERINTERACTION_SRC}/initSofaUserInteraction.cpp
    )

list(APPEND HEADER_FILES
    # AddFramePerformer.h
    # AddFramePerformer.inl
    ${SOFAUSERINTERACTION_SRC}/InteractionPerformer.h
    ${SOFAUSERINTERACTION_SRC}/MouseInteractor.h
    ${SOFAUSERINTERACTION_SRC}/MouseInteractor.inl
    ${SOFAUSERINTERACTION_SRC}/AddRecordedCameraPerformer.h
    ${SOFAUSERINTERACTION_SRC}/AttachBodyPerformer.h
    ${SOFAUSERINTERACTION_SRC}/AttachBodyPerformer.inl
    ${SOFAUSERINTERACTION_SRC}/ComponentMouseInteraction.h
    ${SOFAUSERINTERACTION_SRC}/ComponentMouseInteraction.inl
    ${SOFAUSERINTERACTION_SRC}/Controller.h
    ${SOFAUSERINTERACTION_SRC}/FixParticlePerformer.h
    ${SOFAUSERINTERACTION_SRC}/FixParticlePerformer.inl
    ${SOFAUSERINTERACTION_SRC}/InciseAlongPathPerformer.h
    ${SOFAUSERINTERACTION_SRC}/MechanicalStateController.h
    ${SOFAUSERINTERACTION_SRC}/MechanicalStateController.inl
    ${SOFAUSERINTERACTION_SRC}/Ray.h
    ${SOFAUSERINTERACTION_SRC}/RayContact.h
    ${SOFAUSERINTERACTION_SRC}/RayDiscreteIntersection.h
    ${SOFAUSERINTERACTION_SRC}/RayDiscreteIntersection.inl
    ${SOFAUSERINTERACTION_SRC}/RayModel.h
    ${SOFAUSERINTERACTION_SRC}/RayNewProximityIntersection.h
    ${SOFAUSERINTERACTION_SRC}/RayTraceDetection.h
    ${SOFAUSERINTERACTION_SRC}/SleepController.h
    ${SOFAUSERINTERACTION_SRC}/TopologicalChangeManager.h
    ${SOFAUSERINTERACTION_SRC}/RemovePrimitivePerformer.h
    ${SOFAUSERINTERACTION_SRC}/RemovePrimitivePerformer.inl
    ${SOFAUSERINTERACTION_SRC}/StartNavigationPerformer.h
    ${SOFAUSERINTERACTION_SRC}/SuturePointPerformer.h
    ${SOFAUSERINTERACTION_SRC}/SuturePointPerformer.inl
    )

list(APPEND SOURCE_FILES
    # AddFramePerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/InteractionPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/MouseInteractor.cpp
    ${SOFAUSERINTERACTION_SRC}/AddRecordedCameraPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/AttachBodyPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/ComponentMouseInteraction.cpp
    ${SOFAUSERINTERACTION_SRC}/Controller.cpp
    ${SOFAUSERINTERACTION_SRC}/FixParticlePerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/InciseAlongPathPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/MechanicalStateController.cpp
    ${SOFAUSERINTERACTION_SRC}/RayContact.cpp
    ${SOFAUSERINTERACTION_SRC}/RayDiscreteIntersection.cpp
    ${SOFAUSERINTERACTION_SRC}/RayModel.cpp
    ${SOFAUSERINTERACTION_SRC}/RayNewProximityIntersection.cpp
    ${SOFAUSERINTERACTION_SRC}/RayTraceDetection.cpp
    ${SOFAUSERINTERACTION_SRC}/SleepController.cpp
    ${SOFAUSERINTERACTION_SRC}/TopologicalChangeManager.cpp
    ${SOFAUSERINTERACTION_SRC}/RemovePrimitivePerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/StartNavigationPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/SuturePointPerformer.cpp
    )

find_package(SofaCommon REQUIRED) # SofaMeshCollision SofaDeformable
find_package(SofaGeneralMeshCollision REQUIRED)
find_package(SofaGeneralVisual REQUIRED)
find_package(SofaTopologyMapping REQUIRED)
find_package(SofaBoundaryCondition REQUIRED)
find_package(SofaGraphComponent REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaGeneralMeshCollision SofaGeneralVisual SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
