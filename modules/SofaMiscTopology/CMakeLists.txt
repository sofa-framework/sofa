cmake_minimum_required(VERSION 2.8.12)
project(SofaMiscTopology)

set(HEADER_FILES
    TopologicalChangeProcessor.h
    config.h
    initMiscTopology.h
)

set(SOURCE_FILES
    TopologicalChangeProcessor.cpp
    initMiscTopology.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS  "-DSOFA_BUILD_MISC_TOPOLOGY -DPOINT_DATA_VECTOR_ACCESS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaMisc_install(${PROJECT_NAME})
