cmake_minimum_required(VERSION 2.8)
project(SensableEmulation)

set(SENSABLEEMULATION_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SENSABLEEMULATION_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SENSABLEEMULATION_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    #NewOmniDriverEmu.h
	initSensableEmulation.h
    OmniDriverEmu.h)

set(SOURCE_FILES
    #NewOmniDriverEmu.cpp
    initSensableEmulation.cpp
    OmniDriverEmu.cpp)

set(README_FILES
	PluginSensableEmulation.txt)

find_package(SofaGeneral REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SENSABLEEMULATIONPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral SofaHaptics ${Boost_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
    COMPONENT SensableEmulation_libraries
    EXPORT SensableEmulationTargets
    RUNTIME DESTINATION "${SENSABLEEMULATION_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${SENSABLEEMULATION_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${SENSABLEEMULATION_INSTALL_LIB_DIR}")
