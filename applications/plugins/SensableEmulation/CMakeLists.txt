cmake_minimum_required(VERSION 3.1)
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

find_package(SofaHaptics REQUIRED)
find_package(Boost QUIET COMPONENTS thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SENSABLEEMULATIONPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaHaptics SofaOpenglVisual ${Boost_LIBRARIES})
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
