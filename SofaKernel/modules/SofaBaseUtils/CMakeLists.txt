cmake_minimum_required(VERSION 3.1)
project(SofaBaseUtils)

find_package(SofaFramework REQUIRED)

set(HEADER_FILES
    config.h
    initBaseUtils.h
)

set(SOURCE_FILES
    initBaseUtils.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_UTILS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaBase ${PROJECT_NAME} "SofaBase/${PROJECT_NAME}")
