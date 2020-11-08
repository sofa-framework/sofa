cmake_minimum_required(VERSION 3.12)
project(SofaRigid LANGUAGES CXX)

set(SOFARIGID_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFARIGID_SRC}/config.h.in
    ${SOFARIGID_SRC}/initSofaRigid.h
    ${SOFARIGID_SRC}/JointSpring.h
    ${SOFARIGID_SRC}/JointSpring.inl
    ${SOFARIGID_SRC}/JointSpringForceField.h
    ${SOFARIGID_SRC}/JointSpringForceField.inl
    ${SOFARIGID_SRC}/RigidMapping.h
    ${SOFARIGID_SRC}/RigidMapping.inl
    ${SOFARIGID_SRC}/RigidRigidMapping.h
    ${SOFARIGID_SRC}/RigidRigidMapping.inl
)

set(SOURCE_FILES
    ${SOFARIGID_SRC}/initSofaRigid.cpp
    ${SOFARIGID_SRC}/JointSpring.cpp
    ${SOFARIGID_SRC}/JointSpringForceField.cpp
    ${SOFARIGID_SRC}/RigidMapping.cpp
    ${SOFARIGID_SRC}/RigidRigidMapping.cpp
)

find_package(SofaBase REQUIRED) # SofaBaseMechanics
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFARIGID_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFARIGID_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
