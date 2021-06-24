cmake_minimum_required(VERSION 3.12)
project(SofaMiscCollision VERSION 1.0)

sofa_find_package(SofaBaseCollision REQUIRED)
sofa_find_package(SofaBaseTopology REQUIRED)
sofa_find_package(SofaGeneralMeshCollision REQUIRED)
sofa_find_package(SofaGeneralDeformable REQUIRED)
sofa_find_package(SofaConstraint REQUIRED)
sofa_find_package(SofaExplicitOdeSolver REQUIRED)
sofa_find_package(SofaGeneralExplicitOdeSolver REQUIRED)
sofa_find_package(SofaImplicitOdeSolver REQUIRED)

set(SRC_ROOT src/SofaMiscCollision)

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/initSofaMiscCollision.h
    )

set(SOURCE_FILES
    ${SRC_ROOT}/initSofaMiscCollision.cpp
    )

set(EXTRA_FILES
    ${SRC_ROOT}/README.md
    )

list(APPEND HEADER_FILES
    ${SRC_ROOT}/BarycentricStickContact.h
    ${SRC_ROOT}/BarycentricStickContact.inl
    ${SRC_ROOT}/CapsuleIntersection.h
    ${SRC_ROOT}/DefaultCollisionGroupManager.h
    ${SRC_ROOT}/OBBIntersection.h
    ${SRC_ROOT}/RayTriangleVisitor.h
    ${SRC_ROOT}/RuleBasedContactManager.h
    ${SRC_ROOT}/SolverMerger.h
    ${SRC_ROOT}/TetrahedronDiscreteIntersection.h
    ${SRC_ROOT}/TetrahedronModel.h
    ${SRC_ROOT}/TriangleModelInRegularGrid.h
    ${SRC_ROOT}/CapsuleContactMapper.h
    ${SRC_ROOT}/OBBContactMapper.h
    )

list(APPEND SOURCE_FILES
    ${SRC_ROOT}/BarycentricStickContact.cpp
    ${SRC_ROOT}/CapsuleIntersection.cpp
    ${SRC_ROOT}/CapsuleContact.cpp
    ${SRC_ROOT}/CapsuleContactMapper.h
    ${SRC_ROOT}/CapsuleContactMapper.cpp
    ${SRC_ROOT}/DefaultCollisionGroupManager.cpp
    ${SRC_ROOT}/OBBIntersection.cpp
    ${SRC_ROOT}/OBBContact.cpp
    ${SRC_ROOT}/OBBContactMapper.h
    ${SRC_ROOT}/OBBContactMapper.cpp
    ${SRC_ROOT}/RayTriangleVisitor.cpp
    ${SRC_ROOT}/RuleBasedContactManager.cpp
    ${SRC_ROOT}/SolverMerger.cpp
    ${SRC_ROOT}/TetrahedronBarycentricPenalityContact.cpp
    ${SRC_ROOT}/TetrahedronDiscreteIntersection.cpp
    ${SRC_ROOT}/TetrahedronFrictionContact.cpp
    ${SRC_ROOT}/TetrahedronModel.cpp
    ${SRC_ROOT}/TetrahedronRayContact.cpp
    ${SRC_ROOT}/TriangleModelInRegularGrid.cpp
    )
message(STATUS "SofaMiscModule: build all maintained sofa components.")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaGeneralMeshCollision SofaGeneralDeformable SofaConstraint SofaExplicitOdeSolver SofaGeneralExplicitOdeSolver SofaImplicitOdeSolver SofaBaseCollision)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)
if(SofaSphFluid_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaSphFluid)
endif()
if(SofaDistanceGrid_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDistanceGrid)
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCCOLLISION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCCOLLISION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscCollision_test)
endif()
