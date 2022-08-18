cmake_minimum_required(VERSION 3.12)
project(SofaGuiQt LANGUAGES CXX)

set(SOURCE_FILES
    empty.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.GUI.Qt instead.")

# forward to the new sofang module
sofa_find_package(Sofa.GUI.Qt REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Qt)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "collections"
)
