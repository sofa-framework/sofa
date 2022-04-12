cmake_minimum_required(VERSION 3.12)
project(SofaGuiCommon)

set(SOURCE_FILES
    empty.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

find_package(Sofa.GUI.Common REQUIRED)
find_package(Sofa.GUI.Batch REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Common)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Batch)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
)
