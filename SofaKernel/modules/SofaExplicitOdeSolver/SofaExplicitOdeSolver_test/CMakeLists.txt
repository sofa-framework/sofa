cmake_minimum_required(VERSION 3.1)

project(SofaExplicitOdeSolver_test)

set(SOURCE_FILES
    )

find_package(SofaMiscForceField QUIET)
if(SofaMiscForceField_FOUND)
    list(APPEND SOURCE_FILES
        EulerExplicitSolverDynamic_test.cpp
        )
endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
