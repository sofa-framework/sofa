cmake_minimum_required(VERSION 3.1)

project(SofaEditor VERSION 1.0)

find_package(SofaFramework REQUIRED)

set(HEADER_FILES
    src/SofaEditor/config.h
    src/SofaEditor/BaseSofaEditor.h
)

set(SOURCE_FILES
    src/SofaEditor/BaseSofaEditor.cpp
)

set(EXTRA_FILES
    SofaEditorConfig.cmake.in
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAEDITOR")

target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

## When installing, copy the content of the example directory into module/SofaEditor
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples DESTINATION modules/SofaEditor COMPONENT resources)

## When installing, copy the target into module/SofaEditor
install(TARGETS ${PROJECT_NAME} DESTINATION modules/SofaEditor)

# The signature of sofa_create_package is
#     sofa_create_package(package_name, version, the_targets, <include_subdir>)
# This function assumes that there is a FooConfig.cmake.in file template in the source directory.
# This function is an all in one replacement for install(TARGET), install(EXPORT), and also add
# the example directory in the INSTALL set.
sofa_create_package(${PROJECT_NAME} ${PROJECT_VERSION} SofaEditor "SofaEditor")

### Tests
add_subdirectory(tests)

### Python binding
add_subdirectory(python)
