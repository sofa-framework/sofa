cmake_minimum_required(VERSION 3.1)

project(SofaHelper_test)

set(SOURCE_FILES
    types/Color_test.cpp
    types/Material_test.cpp
    KdTree_test.cpp
    Utils_test.cpp
    Quater_test.cpp
    SVector_test.cpp
    vector_test.cpp
    io/MeshOBJ_test.cpp
    io/XspLoader_test.cpp
    system/FileMonitor_test.cpp
    system/FileRepository_test.cpp
    system/FileSystem_test.cpp
    system/PluginManager_test.cpp
    system/atomic_test.cpp
    #system/thread/CircularQueue_test.cpp
    logging/logging_test.cpp
    testing/TestMessageHandler_test.cpp
)

if(NOT SOFA_NO_OPENGL)
    list(APPEND SOURCE_FILES
        gl/GLSLShader_test.cpp
    )
endif()

add_subdirectory(system/TestPlugin)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaHelper)

add_test(NAME SofaHelper_test COMMAND SofaHelper_test)
