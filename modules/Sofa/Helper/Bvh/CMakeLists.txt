cmake_minimum_required(VERSION 3.1)

project(Sofa.Helper.Bvh VERSION 1.0)

find_package(SofaFramework)

file(GLOB CONFIG_FILES
    config/${PROJECT_NAME}.h
    config/${PROJECT_NAME}.cpp
    )

set(HEADER_FILES
    deprecated_layout/sofa/helper/io/bvh/BVHChannels.h
    deprecated_layout/sofa/helper/io/bvh/BVHOffset.h
    deprecated_layout/sofa/helper/io/bvh/BVHJoint.h
    deprecated_layout/sofa/helper/io/bvh/BVHLoader.h
    deprecated_layout/sofa/helper/io/bvh/BVHMotion.h
)

set(HEADER_FILES
    src/sofa/helper/bvh/BVHChannels.h
    src/sofa/helper/bvh/BVHOffset.h
    src/sofa/helper/bvh/BVHJoint.h
    src/sofa/helper/bvh/BVHLoader.h
    src/sofa/helper/bvh/BVHMotion.h
)
set(SOURCE_FILES
    src/sofa/helper/bvh/BVHJoint.cpp
    src/sofa/helper/bvh/BVHLoader.cpp
    src/sofa/helper/bvh/BVHMotion.cpp
)

set(EXTRA_FILES
)

add_library(${PROJECT_NAME} SHARED ${CONFIG_FILES} ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DBUILD_SOFA_HELPER_BVH")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/config>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/deprecated_layout>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>")

sofa_create_package(${PROJECT_NAME} ${PROJECT_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
