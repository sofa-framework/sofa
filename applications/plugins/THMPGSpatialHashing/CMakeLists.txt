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

find_package(SofaCommon REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseCollision SofaMeshCollision)

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(THMPGSpatialHashing_test)
endif()

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
