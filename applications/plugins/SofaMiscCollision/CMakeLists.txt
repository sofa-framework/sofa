cmake_minimum_required(VERSION 3.1)
project(SofaMiscCollision VERSION 1.0)

sofa_find_package(SofaSphFluid QUIET)
sofa_find_package(SofaDistanceGrid QUIET)

set(SRC_ROOT src/SofaMiscCollision)

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/initMiscCollision.h
    )

set(SOURCE_FILES
    ${SRC_ROOT}/initMiscCollision.cpp
    )

set(EXTRA_FILES
    ${SRC_ROOT}/README.md
    )

list(APPEND HEADER_FILES
    ${SRC_ROOT}/BarycentricStickContact.h
    ${SRC_ROOT}/BarycentricStickContact.inl
    ${SRC_ROOT}/DefaultCollisionGroupManager.h
    ${SRC_ROOT}/RayTriangleVisitor.h
    ${SRC_ROOT}/RuleBasedContactManager.h
    ${SRC_ROOT}/SolverMerger.h
    ${SRC_ROOT}/TetrahedronDiscreteIntersection.h
    ${SRC_ROOT}/TetrahedronModel.h
    ${SRC_ROOT}/TriangleModelInRegularGrid.h
    )

list(APPEND SOURCE_FILES
    ${SRC_ROOT}/BarycentricStickContact.cpp
    ${SRC_ROOT}/DefaultCollisionGroupManager.cpp
    ${SRC_ROOT}/RayTriangleVisitor.cpp
    ${SRC_ROOT}/RuleBasedContactManager.cpp
    ${SRC_ROOT}/SolverMerger.cpp
    ${SRC_ROOT}/TetrahedronBarycentricPenalityContact.cpp
    ${SRC_ROOT}/TetrahedronDiscreteIntersection.cpp
    ${SRC_ROOT}/TetrahedronFrictionContact.cpp
    ${SRC_ROOT}/TetrahedronModel.cpp
    ${SRC_ROOT}/TetrahedronRayContact.cpp
    ${SRC_ROOT}/TriangleModelInRegularGrid.cpp
    ${SRC_ROOT}/TetrahedronBarycentricDistanceLMConstraintContact.cpp
    )
message(STATUS "SofaMiscModule: build all maintained sofa components.")

if(SofaSphFluid_FOUND)
    list(APPEND HEADER_FILES SpatialGridPointModel.h)
    list(APPEND SOURCE_FILES SpatialGridPointModel.cpp)
    message(STATUS "SofaMiscCollision: optional dependency SofaSphFluid found activing: <SpatialGridPointModel>")
else()
    message(STATUS "SofaMiscCollision: optional dependency SofaSphFluid not found de-activing: <SpatialGridPointModel>")
endif()

if(SOFA_HAVE_SMP)
    list(APPEND HEADER_FILES ParallelCollisionPipeline.h)
    list(APPEND SOURCE_FILES ParallelCollisionPipeline.cpp)
endif()

if(SofaDistanceGrid_FOUND)
    list(APPEND SOURCE_FILES FrictionContact_DistanceGrid.cpp)
    list(APPEND SOURCE_FILES BarycentricDistanceLMConstraintContact_DistanceGrid.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaGeneralMeshCollision SofaGeneralDeformable SofaConstraint SofaExplicitOdeSolver SofaGeneralExplicitOdeSolver SofaImplicitOdeSolver SofaBaseCollision)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)
if(SofaSphFluid_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaSphFluid)
endif()
if(SofaDistanceGrid_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDistanceGrid)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_COLLISION")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    INCLUDE_SOURCE_DIR "${SRC_ROOT}"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
