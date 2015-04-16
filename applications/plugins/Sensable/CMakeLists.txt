cmake_minimum_required(VERSION 2.8)
project(Sensable)

set(SENSABLE_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SENSABLE_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SENSABLE_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    NewOmniDriver.h
    OmniDriver.h
    EnslavementForceFeedback.h)

set(SOURCE_FILES
    initSensable.cpp
    NewOmniDriver.cpp
    OmniDriver.cpp
	EnslavementForceFeedback.cpp)

set(README_FILES PluginSensable.txt)

find_package(SofaGeneral REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(OpenHaptics REQUIRED)
include_directories(${OPENHAPTICS_INCLUDE_DIR})
include_directories(${OPENHAPTICS_UTIL_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SENSABLEPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral ${OPENHAPTICS_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
    COMPONENT Sensable_libraries
    EXPORT SensableTargets
    RUNTIME DESTINATION "${SENSABLE_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${SENSABLE_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${SENSABLE_INSTALL_LIB_DIR}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
