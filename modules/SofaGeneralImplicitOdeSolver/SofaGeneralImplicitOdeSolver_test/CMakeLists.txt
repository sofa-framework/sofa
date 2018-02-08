cmake_minimum_required(VERSION 3.1)

project(SofaGeneralImplicitOdeSolver_test)

set(SOURCE_FILES ""
    )

list(APPEND HEADER_FILES
    )
list(APPEND SOURCE_FILES
    VariationalSymplecticExplicitSolverDynamic_test.cpp
    VariationalSymplecticImplicitSolverDynamic_test.cpp
    )

if(SOFA_WITH_DEPRECATED_COMPONENTS)
    list(APPEND SOURCE_FILES
        VariationalSymplecticImplicitSolverNonLinearForceDynamic_test.cpp
        )
endif()

add_definitions("-DSOFAGENERALIMPLICITODESOLVER_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
