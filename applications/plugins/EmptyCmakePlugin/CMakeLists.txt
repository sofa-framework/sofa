cmake_minimum_required(VERSION 2.8)
project(EmptyCmakePlugin)


set(HEADER_FILES
    initEmptyCmakePlugin.h
)

set(SOURCE_FILES
    initEmptyCmakePlugin.cpp
)

find_package(SofaFramework REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PLUGINEXAMPLE")
target_link_libraries(${PROJECT_NAME} SofaCore)

install(TARGETS ${PROJECT_NAME}
    COMPONENT EmptyCmakePlugin_libraries
    EXPORT EmptyCmakePluginTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
