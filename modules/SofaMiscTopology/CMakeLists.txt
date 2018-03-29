cmake_minimum_required(VERSION 3.1)
project(SofaMiscTopology)

set(HEADER_FILES
config.h
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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS  "-DSOFA_BUILD_MISC_TOPOLOGY")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
