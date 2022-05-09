cmake_minimum_required(VERSION 3.12)
project(SofaMiscCollision VERSION 1.0)

sofa_find_package(SofaBaseCollision REQUIRED)
sofa_find_package(SofaBaseMechanics REQUIRED)
sofa_find_package(SofaBaseTopology REQUIRED)
sofa_find_package(SofaGeneralMeshCollision REQUIRED)
sofa_find_package(SofaGeneralDeformable REQUIRED)
sofa_find_package(SofaConstraint REQUIRED)
sofa_find_package(SofaExplicitOdeSolver REQUIRED)
sofa_find_package(SofaGeneralExplicitOdeSolver REQUIRED)
sofa_find_package(SofaImplicitOdeSolver REQUIRED)

set(SOFAMISCCOLLISION_SRC src/SofaMiscCollision)

set(HEADER_FILES
    ${SOFAMISCCOLLISION_SRC}/config.h.in
    ${SOFAMISCCOLLISION_SRC}/initSofaMiscCollision.h
    )

set(SOURCE_FILES
    ${SOFAMISCCOLLISION_SRC}/initSofaMiscCollision.cpp
    )

set(EXTRA_FILES
    ${SOFAMISCCOLLISION_SRC}/README.md
    )

list(APPEND HEADER_FILES
    ${SOFAMISCCOLLISION_SRC}/BarycentricStickContact.h
    ${SOFAMISCCOLLISION_SRC}/BarycentricStickContact.inl
    ${SOFAMISCCOLLISION_SRC}/DefaultCollisionGroupManager.h
    ${SOFAMISCCOLLISION_SRC}/RayTriangleVisitor.h
    ${SOFAMISCCOLLISION_SRC}/RuleBasedContactManager.h
    ${SOFAMISCCOLLISION_SRC}/SolverMerger.h
    ${SOFAMISCCOLLISION_SRC}/TetrahedronDiscreteIntersection.h
    ${SOFAMISCCOLLISION_SRC}/TetrahedronModel.h
    ${SOFAMISCCOLLISION_SRC}/TriangleModelInRegularGrid.h
)

list(APPEND SOURCE_FILES
    ${SOFAMISCCOLLISION_SRC}/BarycentricStickContact.cpp
    ${SOFAMISCCOLLISION_SRC}/DefaultCollisionGroupManager.cpp
    ${SOFAMISCCOLLISION_SRC}/RayTriangleVisitor.cpp
    ${SOFAMISCCOLLISION_SRC}/RuleBasedContactManager.cpp
    ${SOFAMISCCOLLISION_SRC}/DefaultCollisionGroupManager.cpp
    ${SOFAMISCCOLLISION_SRC}/SolverMerger.cpp
    ${SOFAMISCCOLLISION_SRC}/TetrahedronBarycentricPenalityContact.cpp
    ${SOFAMISCCOLLISION_SRC}/TetrahedronDiscreteIntersection.cpp
    ${SOFAMISCCOLLISION_SRC}/TetrahedronFrictionContact.cpp
    ${SOFAMISCCOLLISION_SRC}/TetrahedronModel.cpp
    ${SOFAMISCCOLLISION_SRC}/TetrahedronRayContact.cpp
    ${SOFAMISCCOLLISION_SRC}/TriangleModelInRegularGrid.cpp

)

message(STATUS "SofaMiscModule: build all maintained sofa components.")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaGeneralMeshCollision SofaGeneralDeformable SofaConstraint SofaExplicitOdeSolver SofaGeneralExplicitOdeSolver SofaImplicitOdeSolver SofaBaseCollision SofaBaseMechanics)
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
