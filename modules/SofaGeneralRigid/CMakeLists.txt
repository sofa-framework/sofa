cmake_minimum_required(VERSION 3.12)
project(SofaGeneralRigid LANGUAGES CXX)

set(SOFAGENERALRIGID_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALRIGID_SRC}/config.h.in
    ${SOFAGENERALRIGID_SRC}/initSofaGeneralRigid.h
    )

set(SOURCE_FILES
    ${SOFAGENERALRIGID_SRC}/initSofaGeneralRigid.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALRIGID_SRC}/ArticulatedHierarchyContainer.h
    ${SOFAGENERALRIGID_SRC}/ArticulatedHierarchyContainer.inl
    ${SOFAGENERALRIGID_SRC}/ArticulatedSystemMapping.h
    ${SOFAGENERALRIGID_SRC}/ArticulatedSystemMapping.inl
    ${SOFAGENERALRIGID_SRC}/LineSetSkinningMapping.h
    ${SOFAGENERALRIGID_SRC}/LineSetSkinningMapping.inl
    ${SOFAGENERALRIGID_SRC}/SkinningMapping.h
    ${SOFAGENERALRIGID_SRC}/SkinningMapping.inl
    ${SOFAGENERALRIGID_SRC}/bvh/BVHChannels.h
    ${SOFAGENERALRIGID_SRC}/bvh/BVHJoint.h
    ${SOFAGENERALRIGID_SRC}/bvh/BVHLoader.h
    ${SOFAGENERALRIGID_SRC}/bvh/BVHMotion.h
    ${SOFAGENERALRIGID_SRC}/bvh/BVHOffset.h
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALRIGID_SRC}/ArticulatedHierarchyContainer.cpp
    ${SOFAGENERALRIGID_SRC}/ArticulatedSystemMapping.cpp
    ${SOFAGENERALRIGID_SRC}/LineSetSkinningMapping.cpp
    ${SOFAGENERALRIGID_SRC}/SkinningMapping.cpp
    ${SOFAGENERALRIGID_SRC}/bvh/BVHJoint.cpp
    ${SOFAGENERALRIGID_SRC}/bvh/BVHLoader.cpp
    ${SOFAGENERALRIGID_SRC}/bvh/BVHMotion.cpp
    )

sofa_find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)

sofa_configure_unity_build(TARGET ${PROJECT_NAME}
                           ACTIVATED "${SOFA_BUILD_WITH_UNITY_BUILD_ENABLED}"
                           BATCHSIZE "${SOFA_UNITY_BUILD_BATCH_SIZE}")

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
# cmake_dependent_option(SOFAGENERALRIGID_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
# if(SOFAGENERALRIGID_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(${PROJECT_NAME}_test)
# endif()
