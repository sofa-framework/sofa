cmake_minimum_required(VERSION 2.8.11)
project(SofaTopologyMapping)

set(HEADER_FILES

    initTopologyMapping.h
    Mesh2PointMechanicalMapping.h
    Mesh2PointMechanicalMapping.inl
    SimpleTesselatedTetraMechanicalMapping.h
    SimpleTesselatedTetraMechanicalMapping.inl
    CenterPointTopologicalMapping.h
    Edge2QuadTopologicalMapping.h
    Hexa2QuadTopologicalMapping.h
    Hexa2TetraTopologicalMapping.h
    Mesh2PointTopologicalMapping.h
    Mesh2BezierTopologicalMapping.h
    Quad2TriangleTopologicalMapping.h
    SimpleTesselatedHexaTopologicalMapping.h
    SimpleTesselatedTetraTopologicalMapping.h
    Tetra2TriangleTopologicalMapping.h
    Triangle2EdgeTopologicalMapping.h
    IdentityTopologicalMapping.h
    SubsetTopologicalMapping.h

    )

set(SOURCE_FILES

    initTopologyMapping.cpp
    Mesh2PointMechanicalMapping.cpp
    SimpleTesselatedTetraMechanicalMapping.cpp
    CenterPointTopologicalMapping.cpp
    Edge2QuadTopologicalMapping.cpp
    Hexa2QuadTopologicalMapping.cpp
    Hexa2TetraTopologicalMapping.cpp
    Mesh2PointTopologicalMapping.cpp
    Mesh2BezierTopologicalMapping.cpp
    Quad2TriangleTopologicalMapping.cpp
    SimpleTesselatedHexaTopologicalMapping.cpp
    SimpleTesselatedTetraTopologicalMapping.cpp
    Tetra2TriangleTopologicalMapping.cpp
    Triangle2EdgeTopologicalMapping.cpp
    IdentityTopologicalMapping.cpp
    SubsetTopologicalMapping.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaBaseTopology )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFATOPOLOGY_MAPPING_COMPILER_FLAGS "-DSOFA_BUILD_TOPOLOGY_MAPPING")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFATOPOLOGY_MAPPING_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGeneral_libraries
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"

)

