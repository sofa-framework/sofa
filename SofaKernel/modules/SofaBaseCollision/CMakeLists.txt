cmake_minimum_required(VERSION 3.12)
project(SofaBaseCollision LANGUAGES CXX)

set(SOFABASECOLLISION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASECOLLISION_SRC}/config.h.in
    ${SOFABASECOLLISION_SRC}/fwd.h
    ${SOFABASECOLLISION_SRC}/initSofaBaseCollision.h
    ${SOFABASECOLLISION_SRC}/BaseContactMapper.h
    ${SOFABASECOLLISION_SRC}/BaseIntTool.h
    ${SOFABASECOLLISION_SRC}/BaseProximityIntersection.h
    ${SOFABASECOLLISION_SRC}/BruteForceDetection.h
    ${SOFABASECOLLISION_SRC}/CapsuleIntTool.h
    ${SOFABASECOLLISION_SRC}/CapsuleIntTool.inl
    ${SOFABASECOLLISION_SRC}/CapsuleModel.h
    ${SOFABASECOLLISION_SRC}/CapsuleModel.inl
    ${SOFABASECOLLISION_SRC}/ContactListener.h
    ${SOFABASECOLLISION_SRC}/Cube.h
    ${SOFABASECOLLISION_SRC}/CubeModel.h
    ${SOFABASECOLLISION_SRC}/CylinderModel.h
    ${SOFABASECOLLISION_SRC}/CylinderModel.inl
    ${SOFABASECOLLISION_SRC}/DefaultContactManager.h
    ${SOFABASECOLLISION_SRC}/DefaultPipeline.h
    ${SOFABASECOLLISION_SRC}/DiscreteIntersection.h
    ${SOFABASECOLLISION_SRC}/Intersector.h
    ${SOFABASECOLLISION_SRC}/IntrCapsuleOBB.h
    ${SOFABASECOLLISION_SRC}/IntrCapsuleOBB.inl
    ${SOFABASECOLLISION_SRC}/IntrOBBOBB.h
    ${SOFABASECOLLISION_SRC}/IntrOBBOBB.inl
    ${SOFABASECOLLISION_SRC}/IntrSphereOBB.h
    ${SOFABASECOLLISION_SRC}/IntrSphereOBB.inl
    ${SOFABASECOLLISION_SRC}/IntrUtility3.h
    ${SOFABASECOLLISION_SRC}/IntrUtility3.inl
    ${SOFABASECOLLISION_SRC}/MinProximityIntersection.h
    ${SOFABASECOLLISION_SRC}/MirrorIntersector.h
    ${SOFABASECOLLISION_SRC}/NewProximityIntersection.h
    ${SOFABASECOLLISION_SRC}/NewProximityIntersection.inl
    ${SOFABASECOLLISION_SRC}/OBBIntTool.h
    ${SOFABASECOLLISION_SRC}/OBBModel.h
    ${SOFABASECOLLISION_SRC}/OBBModel.inl
    ${SOFABASECOLLISION_SRC}/RigidCapsuleModel.h
    ${SOFABASECOLLISION_SRC}/RigidCapsuleModel.inl
    ${SOFABASECOLLISION_SRC}/Sphere.h
    ${SOFABASECOLLISION_SRC}/SphereModel.h
    ${SOFABASECOLLISION_SRC}/SphereModel.inl    
)

set(SOURCE_FILES
    ${SOFABASECOLLISION_SRC}/initSofaBaseCollision.cpp
    ${SOFABASECOLLISION_SRC}/BaseContactMapper.cpp
    ${SOFABASECOLLISION_SRC}/BaseIntTool.cpp
    ${SOFABASECOLLISION_SRC}/BaseProximityIntersection.cpp
    ${SOFABASECOLLISION_SRC}/BruteForceDetection.cpp
    ${SOFABASECOLLISION_SRC}/CapsuleIntTool.cpp
    ${SOFABASECOLLISION_SRC}/CapsuleModel.cpp
    ${SOFABASECOLLISION_SRC}/ContactListener.cpp
    ${SOFABASECOLLISION_SRC}/CubeModel.cpp
    ${SOFABASECOLLISION_SRC}/CylinderModel.cpp
    ${SOFABASECOLLISION_SRC}/DefaultContactManager.cpp
    ${SOFABASECOLLISION_SRC}/DefaultPipeline.cpp
    ${SOFABASECOLLISION_SRC}/DiscreteIntersection.cpp
    ${SOFABASECOLLISION_SRC}/IntrCapsuleOBB.cpp
    ${SOFABASECOLLISION_SRC}/IntrOBBOBB.cpp
    ${SOFABASECOLLISION_SRC}/IntrSphereOBB.cpp
    ${SOFABASECOLLISION_SRC}/IntrUtility3.cpp
    ${SOFABASECOLLISION_SRC}/MinProximityIntersection.cpp
    ${SOFABASECOLLISION_SRC}/NewProximityIntersection.cpp
    ${SOFABASECOLLISION_SRC}/OBBIntTool.cpp
    ${SOFABASECOLLISION_SRC}/OBBModel.cpp
    ${SOFABASECOLLISION_SRC}/RigidCapsuleModel.cpp
    ${SOFABASECOLLISION_SRC}/SphereModel.cpp
)

sofa_find_package(SofaFramework REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationCore)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASECOLLISION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASECOLLISION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaBaseCollision_test)
endif()
