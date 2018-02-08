cmake_minimum_required(VERSION 3.1)
project(SofaMiscEngine)

set(HEADER_FILES
    config.h
    initMiscEngine.h
    )

set(SOURCE_FILES
    initMiscEngine.cpp
    )

list(APPEND HEADER_FILES
    DisplacementMatrixEngine.h
    DisplacementMatrixEngine.inl
    Distances.h
    Distances.inl
    ProjectiveTransformEngine.h
    ProjectiveTransformEngine.inl
    )
list(APPEND SOURCE_FILES
    DisplacementMatrixEngine.cpp
    Distances.cpp
    ProjectiveTransformEngine.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaNonUniformFem)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_ENGINE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
