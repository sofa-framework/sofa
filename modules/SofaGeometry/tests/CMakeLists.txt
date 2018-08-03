cmake_minimum_required(VERSION 3.1)

project(SofaGeometry_test)

#find_package(SofaEditor REQUIRED)

set(SOURCE_FILES
    Plane_test.cpp
    Ray_test.cpp
)

#find_package(SofaPython QUIET)
#if(SofaPython_FOUND)
#    list(APPEND SOURCE_FILES
#        CustomField_test.cpp)
#    add_definitions("-DDO_SOFAPYTHON_FEATURES=1")
#else()
#    add_definitions("-DDO_SOFAPYTHON_FEATURES=0")
#endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaGeometry)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

