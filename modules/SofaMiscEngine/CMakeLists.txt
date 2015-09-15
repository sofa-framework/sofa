cmake_minimum_required(VERSION 2.8.12)
project(SofaMiscEngine)

set(HEADER_FILES
    Distances.h
    Distances.inl
    ProjectiveTransformEngine.h
    ProjectiveTransformEngine.inl
    config.h
    initMiscEngine.h
)

set(SOURCE_FILES
    Distances.cpp
    ProjectiveTransformEngine.cpp
    initMiscEngine.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaNonUniformFem)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_ENGINE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaMisc_install(${PROJECT_NAME})
