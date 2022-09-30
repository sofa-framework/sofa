cmake_minimum_required(VERSION 3.12)
project(SensableEmulation)


set(HEADER_FILES
    OmniDriverEmu.h
    config.h
)

set(SOURCE_FILES
    OmniDriverEmu.cpp
    initSensableEmulation.cpp
)

set(README_FILES PluginSensableEmulation.txt)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Component.Haptics REQUIRED)
sofa_find_package(Sofa.Component.Controller REQUIRED)
sofa_find_package(Sofa.GL.Component.Rendering3D REQUIRED)
sofa_find_package(Boost COMPONENTS thread REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SENSABLEEMULATIONPLUGIN")
target_link_libraries(${PROJECT_NAME} Sofa.Component.Haptics Sofa.Component.Controller Sofa.GL.Component.Rendering3D Boost::thread)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

# Seems to be required by boost::asio
if(UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

install(TARGETS ${PROJECT_NAME}
        COMPONENT SensableEmulation_libraries
        EXPORT SensableEmulationTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
