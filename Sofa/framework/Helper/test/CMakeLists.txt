cmake_minimum_required(VERSION 3.12)

project(Sofa.Helper_test)

set(SOURCE_FILES
    DiffLib_test.cpp
    Factory_test.cpp
    KdTree_test.cpp
    NameDecoder_test.cpp
    OptionsGroup_test.cpp
    StringUtils_test.cpp
    TagFactory_test.cpp
    Utils_test.cpp
    accessor/ReadAccessor.cpp
    accessor/WriteAccessor.cpp
    io/MeshOBJ_test.cpp
    io/STBImage_test.cpp
    io/XspLoader_test.cpp
    logging/logging_test.cpp
    narrow_cast_test.cpp
    system/FileMonitor_test.cpp
    system/FileRepository_test.cpp
    system/FileSystem_test.cpp
    system/PluginManager_test.cpp
    system/thread/CircularQueue_test.cpp
)

add_subdirectory(system/TestPlugin)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Helper)

add_definitions("-DSTBIMAGETEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/io/images\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
