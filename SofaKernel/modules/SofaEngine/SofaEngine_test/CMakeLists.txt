cmake_minimum_required(VERSION 3.12)

project(SofaEngine_test)


set(HEADER_FILES
        )

set(SOURCE_FILES
    BoxROI_test.cpp
    )
    
find_package(SofaEngine REQUIRED)
find_package(SofaBaseMechanics REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaEngine SofaBaseMechanics)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
