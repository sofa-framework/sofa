cmake_minimum_required(VERSION 3.1)
project(EmptyCmakePlugin)

set(HEADER_FILES
    config.h
)

set(SOURCE_FILES
    initEmptyCmakePlugin.cpp
)

find_package(SofaFramework REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore)

install(TARGETS ${PROJECT_NAME}
        COMPONENT EmptyCmakePlugin_libraries
        EXPORT EmptyCmakePluginTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
