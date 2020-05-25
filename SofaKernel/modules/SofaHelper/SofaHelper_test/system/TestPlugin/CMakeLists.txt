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
target_link_libraries(${PROJECT_NAME} SofaCore SofaDefaultType)

install(TARGETS ${PROJECT_NAME}
        COMPONENT TestPlugin_libraries
        EXPORT TestPluginTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
