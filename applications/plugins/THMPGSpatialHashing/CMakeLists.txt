cmake_minimum_required(VERSION 2.8)
project(THMPGSpatialHashing)


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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_THMPGSPATIALHASHING")
target_link_libraries(${PROJECT_NAME} SofaBaseCollision SofaMeshCollision)

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(THMPGSpatialHashing_test)
endif()

install(TARGETS ${PROJECT_NAME}
    COMPONENT THMPGSpatialHashing_libraries
    EXPORT THMPGSpatialHashingTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
