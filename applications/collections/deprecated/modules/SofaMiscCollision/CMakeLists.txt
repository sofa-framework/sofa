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
    ${SOFAMISCCOLLISION_SRC}/DefaultCollisionGroupManager.h
    ${SOFAMISCCOLLISION_SRC}/RayTriangleVisitor.h
    ${SOFAMISCCOLLISION_SRC}/SolverMerger.h
)

list(APPEND SOURCE_FILES

    ${SOFAMISCCOLLISION_SRC}/DefaultCollisionGroupManager.cpp
    ${SOFAMISCCOLLISION_SRC}/RayTriangleVisitor.cpp
    ${SOFAMISCCOLLISION_SRC}/SolverMerger.cpp

)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaGeneralMeshCollision SofaGeneralDeformable SofaConstraint SofaExplicitOdeSolver SofaGeneralExplicitOdeSolver SofaImplicitOdeSolver SofaBaseCollision SofaBaseMechanics)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Collision.Geometry, Sofa.Component.Collision.Detection.Intersection and Sofa.Component.Collision.Response.Contact instead.
#    Moreover, OBB and Capsule-related components have been moved into the external plugin CollisionOBBCapsule.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Collision.Geometry REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Geometry)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Intersection)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Contact)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "collections"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCCOLLISION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCCOLLISION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscCollision_test)
endif()
