cmake_minimum_required(VERSION 3.22)
project(Sofa.GUI.Component LANGUAGES CXX)

set(SOFAGUICOMPONENT_SOURCE_DIR src/sofa/gui/component)

set(HEADER_FILES
    ${SOFAGUICOMPONENT_SOURCE_DIR}/config.h.in
    ${SOFAGUICOMPONENT_SOURCE_DIR}/init.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/AddFrameButtonSetting.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/AddRecordedCameraButtonSetting.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/AttachBodyButtonSetting.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/FixPickedParticleButtonSetting.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/ConstraintAttachButtonSetting.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/AddRecordedCameraPerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/AttachBodyPerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/AttachBodyPerformer.inl
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/ComponentMouseInteraction.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/ComponentMouseInteraction.inl
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/ConstraintAttachBodyPerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/ConstraintAttachBodyPerformer.inl
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/FixParticlePerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/FixParticlePerformer.inl
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/InciseAlongPathPerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/InteractionPerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/MouseInteractor.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/MouseInteractor.inl
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/RemovePrimitivePerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/RemovePrimitivePerformer.inl
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/StartNavigationPerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/SuturePointPerformer.h
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/SuturePointPerformer.inl
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/TopologicalChangeManager.h
)

set(SOURCE_FILES
    ${SOFAGUICOMPONENT_SOURCE_DIR}/init.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/AddFrameButtonSetting.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/AddRecordedCameraButtonSetting.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/AttachBodyButtonSetting.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/FixPickedParticleButtonSetting.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/ConstraintAttachButtonSetting.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/AddRecordedCameraPerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/AttachBodyPerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/ComponentMouseInteraction.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/ConstraintAttachBodyPerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/FixParticlePerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/InciseAlongPathPerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/InteractionPerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/MouseInteractor.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/RemovePrimitivePerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/StartNavigationPerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/SuturePointPerformer.cpp
    ${SOFAGUICOMPONENT_SOURCE_DIR}/performer/TopologicalChangeManager.cpp

)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Setting REQUIRED)
sofa_find_package(Sofa.Component.Visual REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.Spring REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Projective REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Model REQUIRED)
sofa_find_package(Sofa.Component.Collision.Geometry REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Mapper REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
sofa_find_package(Sofa.Component.Topology.Mapping REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Setting)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Visual)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.Spring)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Projective)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Model)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Geometry)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Mapper)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.StateContainer)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Dynamic)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Mapping)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
