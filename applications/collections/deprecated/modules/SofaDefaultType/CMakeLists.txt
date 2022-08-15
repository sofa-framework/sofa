cmake_minimum_required(VERSION 3.12)
project(SofaDefaultType)

set(SOURCE_FILES
    empty.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.DefaultType instead.")

sofa_find_package(Sofa.DefaultType REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.DefaultType)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER SofaFramework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
