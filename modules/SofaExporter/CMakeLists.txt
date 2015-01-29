cmake_minimum_required(VERSION 2.8.12)
project(SofaExporter)

set(HEADER_FILES

    initExporter.h
    WriteState.h
    WriteState.inl
    WriteTopology.h
    WriteTopology.inl
    VTKExporter.h
    OBJExporter.h
    STLExporter.h
    MeshExporter.h
    BlenderExporter.h
    BlenderExporter.inl

    )

set(SOURCE_FILES

    initExporter.cpp
    WriteState.cpp
    WriteTopology.cpp
    VTKExporter.cpp
    OBJExporter.cpp
    STLExporter.cpp
    MeshExporter.cpp
    BlenderExporter.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAEXPORTER_COMPILER_FLAGS "-DSOFA_BUILD_EXPORTER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAEXPORTER_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )


install(TARGETS ${PROJECT_NAME}
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}" COMPONENT SofaGeneral_libraries
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}" COMPONENT SofaGeneral_libraries
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}" COMPONENT SofaGeneral_libraries
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaGeneral_headers
)

