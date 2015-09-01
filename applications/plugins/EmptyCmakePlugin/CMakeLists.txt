cmake_minimum_required(VERSION 2.8)
project(EmptyCmakePlugin)

set(EMPTYCMAKEPLUGIN_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(EMPTYCMAKEPLUGIN_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(EMPTYCMAKEPLUGIN_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

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
    RUNTIME DESTINATION "${EMPTYCMAKEPLUGIN_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${EMPTYCMAKEPLUGIN_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${EMPTYCMAKEPLUGIN_INSTALL_LIB_DIR}")
