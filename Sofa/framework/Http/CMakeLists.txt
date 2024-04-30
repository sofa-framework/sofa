cmake_minimum_required(VERSION 3.22)
project(Sofa.Http LANGUAGES CXX)

set(SOFAHTTP_SRC "src/sofa/http")

set(HEADER_FILES
    ${SOFAHTTP_SRC}/config.h.in
    ${SOFAHTTP_SRC}/init.h
)

set(SOURCE_FILES
    ${SOFAHTTP_SRC}/init.cpp
)

find_package(Sofa.Config REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
