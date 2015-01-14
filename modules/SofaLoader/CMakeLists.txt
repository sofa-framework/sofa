cmake_minimum_required(VERSION 2.8.12)
project(SofaLoader)

set(HEADER_FILES

    initLoader.h
    GridMeshCreator.h
    StringMeshCreator.h
    MeshGmshLoader.h
    MeshObjLoader.h
	GIDMeshLoader.h
    MeshOffLoader.h
    MeshTrianLoader.h
    MeshVTKLoader.h
    MeshSTLLoader.h
    MeshXspLoader.h
    OffSequenceLoader.h
    SphereLoader.h
    VoxelGridLoader.h
    InputEventReader.h
    ReadState.h
    ReadState.inl
    ReadTopology.h
    ReadTopology.inl
    )

set(SOURCE_FILES

    initLoader.cpp
    GridMeshCreator.cpp
    StringMeshCreator.cpp
    MeshGmshLoader.cpp
	MeshObjLoader.cpp
	GIDMeshLoader.cpp
    MeshOffLoader.cpp
    MeshTrianLoader.cpp
    MeshVTKLoader.cpp
    MeshSTLLoader.cpp
    MeshXspLoader.cpp
    OffSequenceLoader.cpp
    SphereLoader.cpp
    VoxelGridLoader.cpp
    InputEventReader.cpp
    ReadState.cpp
    ReadTopology.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} PUBLIC SofaSimulationTree SofaHelper)

set( SOFALOADER_COMPILER_FLAGS "-DSOFA_BUILD_LOADER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFALOADER_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION "${SOFACOMMON_INSTALL_BIN_DIR}" COMPONENT SofaCommon_libraries
        LIBRARY DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}" COMPONENT SofaCommon_libraries
        ARCHIVE DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}" COMPONENT SofaCommon_libraries
        PUBLIC_HEADER DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaCommon_headers
        )