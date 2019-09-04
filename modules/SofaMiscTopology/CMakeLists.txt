cmake_minimum_required(VERSION 3.1)
project(SofaMiscTopology)

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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS  "-DSOFA_BUILD_MISC_TOPOLOGY")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} "SofaMisc/${PROJECT_NAME}")
