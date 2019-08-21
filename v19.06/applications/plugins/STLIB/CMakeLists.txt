cmake_minimum_required(VERSION 3.5)
project(STLIB VERSION 1.0)

find_package(SofaFramework REQUIRED)
find_package(SofaPython REQUIRED)

include(cmake/environment.cmake)

set(HEADER_FILES src/initPlugin.h)
set(SOURCE_FILES src/initPlugin.cpp)

file(GLOB_RECURSE RESOURCE_FILES  "*.md" "*.psl" "*.py" "*.pyscn" "*.scn" "*.rst")

# Config files and install rules for pythons scripts
sofa_install_pythonscripts(PLUGIN_NAME ${PROJECT_NAME} PYTHONSCRIPTS_SOURCE_DIR "python" PYTHONSCRIPTS_INSTALL_DIR "python")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${RESOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_STLIB")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
target_link_libraries(${PROJECT_NAME} SofaCore SofaPython)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

## Install rules for the library and headers; CMake package configurations files

sofa_generate_package(
    NAME STLIB 
    VERSION ${PROJECT_VERSION} 
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR STLIB
    INCLUDE_SOURCE_DIR "src"
    RELOCATABLE "plugins"
    )

install(DIRECTORY docs DESTINATION share/sofa/${PROJECT_NAME})
install(FILES "README.md" "LICENSE" DESTINATION .)

include(cmake/packaging.cmake)
