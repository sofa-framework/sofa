include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    MeshVTKLoader_test.cpp
    MeshObjLoader_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaLoader
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
 
