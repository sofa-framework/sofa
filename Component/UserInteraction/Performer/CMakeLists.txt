cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.UserInteraction.Performer LANGUAGES CXX)

set(SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR "src/sofa/component/userinteraction/performer")

set(HEADER_FILES
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/init.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/AddRecordedCameraPerformer.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/AttachBodyPerformer.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/AttachBodyPerformer.inl
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/ComponentMouseInteraction.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/ComponentMouseInteraction.inl
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/FixParticlePerformer.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/FixParticlePerformer.inl
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/InciseAlongPathPerformer.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/InteractionPerformer.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/MouseInteractor.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/MouseInteractor.inl
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/RemovePrimitivePerformer.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/RemovePrimitivePerformer.inl
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/StartNavigationPerformer.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/SuturePointPerformer.h
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/SuturePointPerformer.inl
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/TopologicalChangeManager.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/init.cpp    
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/AddRecordedCameraPerformer.cpp
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/AttachBodyPerformer.cpp    
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/ComponentMouseInteraction.cpp
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/FixParticlePerformer.cpp
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/InciseAlongPathPerformer.cpp
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/InteractionPerformer.cpp    
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/MouseInteractor.cpp
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/RemovePrimitivePerformer.cpp
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/StartNavigationPerformer.cpp
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/SuturePointPerformer.cpp
    ${SOFACOMPONENTUSERINTERACTIONPERFORMER_SOURCE_DIR}/TopologicalChangeManager.cpp
)

sofa_find_package(Sofa.SimulationCore REQUIRED)
sofa_find_package(Sofa.Component.UserInteraction.ConfigurationSetting REQUIRED)
sofa_find_package(Sofa.Component.Visual REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.Spring REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Projective REQUIRED)
sofa_find_package(Sofa.Component.Collision.Model REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Mapper REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
sofa_find_package(Sofa.Component.Topology.Mapping REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.SimulationCore)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.UserInteraction.ConfigurationSetting)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Visual)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.Spring)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Projective)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Model)
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
