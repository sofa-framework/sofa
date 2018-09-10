cmake_minimum_required(VERSION 3.1)
project(SofaGeometry_test)

if(NOT SOFA_BUILD_TESTS)
    return()
endif()

####################################################################################################
### Module dependencies
####################################################################################################
find_package(SofaGeometry QUIET)
if(!SofaGeometry_FOUND)
    message("-- 'SofaGeometry_test' requires the module SofaGeometry (is the module activated?).")
    return()
endif()

####################################################################################################
### Building
####################################################################################################
set(SOURCE_FILES
    Plane_test.cpp
    Ray_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaGeometry)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

