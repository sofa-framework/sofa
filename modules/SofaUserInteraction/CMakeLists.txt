cmake_minimum_required(VERSION 3.12)
project(SofaUserInteraction LANGUAGES CXX)

set(SOFAUSERINTERACTION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAUSERINTERACTION_SRC}/config.h.in
    ${SOFAUSERINTERACTION_SRC}/initSofaUserInteraction.h
    ${SOFAUSERINTERACTION_SRC}/AddRecordedCameraPerformer.h
    ${SOFAUSERINTERACTION_SRC}/AttachBodyPerformer.h
    ${SOFAUSERINTERACTION_SRC}/AttachBodyPerformer.inl
    ${SOFAUSERINTERACTION_SRC}/ComponentMouseInteraction.h
    ${SOFAUSERINTERACTION_SRC}/ComponentMouseInteraction.inl
    ${SOFAUSERINTERACTION_SRC}/FixParticlePerformer.h
    ${SOFAUSERINTERACTION_SRC}/FixParticlePerformer.inl
    ${SOFAUSERINTERACTION_SRC}/InciseAlongPathPerformer.h
    ${SOFAUSERINTERACTION_SRC}/InteractionPerformer.h
    ${SOFAUSERINTERACTION_SRC}/MouseInteractor.h
    ${SOFAUSERINTERACTION_SRC}/MouseInteractor.inl
    ${SOFAUSERINTERACTION_SRC}/RemovePrimitivePerformer.h
    ${SOFAUSERINTERACTION_SRC}/RemovePrimitivePerformer.inl
    ${SOFAUSERINTERACTION_SRC}/StartNavigationPerformer.h
    ${SOFAUSERINTERACTION_SRC}/SuturePointPerformer.h
    ${SOFAUSERINTERACTION_SRC}/SuturePointPerformer.inl
    ${SOFAUSERINTERACTION_SRC}/TopologicalChangeManager.h
    )

set(SOURCE_FILES
    ${SOFAUSERINTERACTION_SRC}/initSofaUserInteraction.cpp
    ${SOFAUSERINTERACTION_SRC}/AddRecordedCameraPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/AttachBodyPerformer.cpp    
    ${SOFAUSERINTERACTION_SRC}/ComponentMouseInteraction.cpp
    ${SOFAUSERINTERACTION_SRC}/FixParticlePerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/InciseAlongPathPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/InteractionPerformer.cpp    
    ${SOFAUSERINTERACTION_SRC}/MouseInteractor.cpp
    ${SOFAUSERINTERACTION_SRC}/RemovePrimitivePerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/StartNavigationPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/SuturePointPerformer.cpp
    ${SOFAUSERINTERACTION_SRC}/TopologicalChangeManager.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAUSERINTERACTION_SRC}/SleepController.h
    )

list(APPEND SOURCE_FILES
    ${SOFAUSERINTERACTION_SRC}/SleepController.cpp
    )

sofa_find_package(SofaMeshCollision REQUIRED)
sofa_find_package(SofaDeformable REQUIRED)
sofa_find_package(SofaGeneralMeshCollision REQUIRED)
sofa_find_package(SofaGeneralVisual REQUIRED)
sofa_find_package(SofaTopologyMapping REQUIRED)
sofa_find_package(SofaBoundaryCondition REQUIRED)
sofa_find_package(SofaGraphComponent REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaGeneralMeshCollision SofaGeneralVisual SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent)

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Collision.Geometry, Sofa.Component.Collision.Detection.Algorithm, Sofa.Component.Collision.Detection.Intersection, Sofa.Component.Collision.Response.Contact and Sofa.Component.UserInteraction.Controller instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Collision.Geometry REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
sofa_find_package(Sofa.Component.UserInteraction.Controller REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Geometry)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Algorithm)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Intersection)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Contact)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.UserInteraction.Controller)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
