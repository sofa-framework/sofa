cmake_minimum_required(VERSION 3.22)
project(SofaHaptics LANGUAGES CXX)

set(HEADER_FILES
    src/SofaHaptics/config.h
    src/SofaHaptics/initSofaHaptics.h
    )

set(SOURCE_FILES
    src/SofaHaptics/initSofaHaptics.cpp
    )

set(EXTRA_FILES
    README.md
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Haptics instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Haptics REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Haptics)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
    )
