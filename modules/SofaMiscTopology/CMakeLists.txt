cmake_minimum_required(VERSION 3.12)
project(SofaMiscTopology)

find_package(SofaBase REQUIRED) # SofaBaseTopology
sofa_find_package(ZLIB BOTH_SCOPES)

set(HEADER_FILES
config.h.in
initMiscTopology.h
)

set(SOURCE_FILES
initMiscTopology.cpp
)

list(APPEND HEADER_FILES
TopologicalChangeProcessor.h
)
list(APPEND SOURCE_FILES
TopologicalChangeProcessor.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
if(ZLIB_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
    if(CMAKE_SYSTEM_NAME STREQUAL Windows)
        sofa_install_libraries(TARGETS ZLIB::ZLIB)
    endif()
endif()

sofa_add_targets_to_package(
    PACKAGE_NAME SofaMisc
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
