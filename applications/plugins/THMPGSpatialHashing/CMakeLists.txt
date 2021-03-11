cmake_minimum_required(VERSION 3.12)
project(THMPGSpatialHashing VERSION 0.1)


set(HEADER_FILES
    THMPGHashTable.h
    THMPGSpatialHashing.h
    config.h
)

set(SOURCE_FILES
    THMPGHashTable.cpp
    THMPGSpatialHashing.cpp
    initTHMPGSpatialHashingPlugin.cpp
)

sofa_find_package(SofaBaseCollision REQUIRED)
sofa_find_package(SofaMeshCollision REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseCollision SofaMeshCollision)

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(THMPGSpatialHashing_test)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    PACKAGE_VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
