cmake_minimum_required(VERSION 3.1)
project(TestPlugin)


set(HEADER_FILES
    ComponentA.h
    ComponentB.h
    TestPlugin.h
)

set(SOURCE_FILES
    ComponentA.cpp
    ComponentB.cpp
    initTestPlugin.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_TESTPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaCore SofaDefaultType)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT TestPlugin_libraries
        EXPORT TestPluginTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
