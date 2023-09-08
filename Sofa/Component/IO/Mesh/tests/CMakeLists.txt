cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.IO.Mesh_test)

set(SOURCE_FILES
    MeshExporter_test.cpp
    MeshGmshLoader_test.cpp
    MeshOBJLoader_test.cpp
    MeshSTLLoader_test.cpp
    MeshVTKLoader_test.cpp
    MeshXspLoader_test.cpp
    OffSequenceLoader_test.cpp
    STLExporter_test.cpp
    VisualModelOBJExporter_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFA_COMPONENT_IO_MESH_TEST_FILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/files/\"")
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.IO.Mesh)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
