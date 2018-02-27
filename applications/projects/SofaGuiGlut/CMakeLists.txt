cmake_minimum_required(VERSION 3.1)
project(SofaGuiGlut)

find_package(SofaSimulation REQUIRED)
find_package(GLUT REQUIRED)
find_package(Boost COMPONENTS thread chrono )

set(HEADER_FILES
    SimpleGUI.h)

set(SOURCE_FILES
    SimpleGUI.cpp)


if(Boost_FOUND)
    list(APPEND HEADER_FILES MultithreadGUI.h)
    list(APPEND SOURCE_FILES MultithreadGUI.cpp)
endif()

list(APPEND SOURCE_FILES Main.cpp)

find_package(GLUT REQUIRED)

if(NOT APPLE)
	set(GLUT_ACTUAL_LIBRARIES  ${GLUT_glut_LIBRARY} )
else()
    # Workaround: on OS X, "${GLUT_glut_LIBRARY}" is not a target (it is
    # actually "-framework GLUT"), so we use find_library to find
    # something better (like /System/Library/Frameworks/GLUT.framework.)
    # (Fun fact: FindGLUT.cmake litterally reads "These values for Apple
    # could probably do with improvement.)"
	find_library(ACTUAL_GLUT_LIBRARIES "GLUT")
	set(GLUT_ACTUAL_LIBRARIES  ${ACTUAL_GLUT_LIBRARIES} )
endif()

set(_configPluginFileName plugin_list.conf)
set(_defaultConfigPluginFileName "${_configPluginFileName}.default")
set(_defaultConfigPluginFilePath "${CMAKE_BINARY_DIR}/${_pluginLocation}/${_defaultConfigPluginFileName}")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES}  )

target_link_libraries(${PROJECT_NAME} SofaGuiCommon SofaSimulationGraph ${GLUT_ACTUAL_LIBRARIES} )
if(Boost_FOUND)
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIGLUT -DCONFIG_PLUGIN_FILENAME=${_configPluginFileName} -DDEFAULT_CONFIG_PLUGIN_FILENAME=${_defaultConfigPluginFileName}")

sofa_install_targets(SofaGui SofaGuiGlut "")
