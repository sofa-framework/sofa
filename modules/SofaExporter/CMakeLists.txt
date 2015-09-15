cmake_minimum_required(VERSION 2.8.12)
project(SofaExporter)

set(HEADER_FILES
    BlenderExporter.h
    BlenderExporter.inl
    MeshExporter.h
    OBJExporter.h
    STLExporter.h
    VTKExporter.h
    WriteState.h
    WriteState.inl
    WriteTopology.h
    WriteTopology.inl
    config.h
    initExporter.h
)

set(SOURCE_FILES
    BlenderExporter.cpp
    MeshExporter.cpp
    OBJExporter.cpp
    STLExporter.cpp
    VTKExporter.cpp
    WriteState.cpp
    WriteTopology.cpp
    initExporter.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_EXPORTER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaGeneral_install(${PROJECT_NAME})
