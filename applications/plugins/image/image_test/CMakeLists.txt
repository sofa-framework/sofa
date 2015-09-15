cmake_minimum_required(VERSION 2.8.12)

project(image_test)

set(HEADER_FILES
    TestImageEngine.h
)

set(SOURCE_FILES
    TestImageEngine.cpp
    DataImage_test.cpp
    ImageEngine_test.cpp
)

add_definitions("-DIMAGETEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} image SofaTest SofaGTestMain)

# ???
if(WIN32 AND NOT CMAKE_CL_64)
    add_definitions("-DEIGEN_DONT_ALIGN")
endif()

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
