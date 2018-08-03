cmake_minimum_required(VERSION 3.1)

project(SofaGeometry VERSION 1.0)

find_package(SofaFramework REQUIRED)

set(HEADER_FILES
    src/SofaGeometry/config.h
    src/SofaGeometry/Plane.h
    src/SofaGeometry/Ray.h
    src/SofaGeometry/Constants.h
)

set(SOURCE_FILES
    src/SofaGeometry/Plane.cpp
    src/SofaGeometry/Ray.cpp
    src/SofaGeometry/Constants.cpp
)

set(EXTRA_FILES
    SofaGeometryConfig.cmake.in
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGEOMETRY")

target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(${PROJECT_NAME} ${PROJECT_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})

find_package(PythonLibs 2.7 REQUIRED)
if(PythonLibs_FOUND)
    add_subdirectory(python)
else()
    message("Python 2.7 is not available, disabling compilation of the binding for SofaGeometry")
endif()
add_subdirectory(tests)

## When installing, copy the content of the example directory into module/SofaEditor
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples DESTINATION modules/SofaGeometry COMPONENT resources)

## When installing, copy the target into module/SofaEditor
install(TARGETS ${PROJECT_NAME} DESTINATION modules/SofaGeometry)
