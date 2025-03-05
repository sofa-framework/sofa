cmake_minimum_required(VERSION 3.22)
project(sofaInitTimer)

find_package(SofaGeneral)
find_package(SofaMisc)

sofa_find_package(Boost QUIET OPTIONAL_COMPONENTS thread)

set(SOURCE_FILES
    sofaInitTimer.cpp
)

if(Boost_THREAD_FOUND)
    list(APPEND SOURCE_FILES ${SRC_ROOT}/system/thread/TimeoutWatchdog.cpp)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGeneral SofaMisc)
if(UNIX)
    target_link_libraries(${PROJECT_NAME} dl)
endif()

if(Boost_THREAD_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Boost::thread)
endif()
