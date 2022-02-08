cmake_minimum_required(VERSION 3.12)
project(ArticulatedSystemPlugin VERSION 1.0 LANGUAGES CXX)

set(ARTICULATEDSYSTEMPLUGIN_SRC_DIR src/ArticulatedSystemPlugin)
set(HEADER_FILES
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/config.h.in
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/init.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedHierarchyContainer.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedHierarchyContainer.inl
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedSystemMapping.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedSystemMapping.inl
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedHierarchyController.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedHierarchyBVHController.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/bvh/BVHChannels.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/bvh/BVHJoint.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/bvh/BVHLoader.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/bvh/BVHMotion.h
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/bvh/BVHOffset.h
    )
set(SOURCE_FILES
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/init.cpp
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedHierarchyContainer.cpp
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedSystemMapping.cpp
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedHierarchyController.cpp
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/ArticulatedHierarchyBVHController.cpp
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/bvh/BVHJoint.cpp
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/bvh/BVHLoader.cpp
    ${ARTICULATEDSYSTEMPLUGIN_SRC_DIR}/bvh/BVHMotion.cpp
    )

find_package(SofaBaseMechanics REQUIRED)
find_package(SofaUserInteraction REQUIRED)

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaUserInteraction)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
