cmake_minimum_required(VERSION 2.8)
project(ManualMapping)


set(HEADER_FILES
    ManualLinearMapping.h
    ManualLinearMapping.inl
    config.h
)

set(SOURCE_FILES
    ManualLinearMapping.cpp
    initManualMapping.cpp
)

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_ManualMapping")
target_link_libraries(${PROJECT_NAME} SofaComponentBase)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
    COMPONENT ManualMapping_libraries
    EXPORT ManualMappingTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
