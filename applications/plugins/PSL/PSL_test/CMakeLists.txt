cmake_minimum_required(VERSION 3.1)

project(PSL_test)


set(HEADER_FILES)

set(SOURCE_FILES
        PSL_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest PSL)
add_definitions("-DPSL_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/pslfiles/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
