cmake_minimum_required(VERSION 3.1)

project(SofaGeneralImplicitOdeSolver_test)

message(STATUS "SofaGeneralImplicitOdeSolver_test:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(SOURCE_FILES ""
    )

############################## COMPONENTS HERE ARE THE LIGHT-SET ###################################
if(SOFA_BUILD_COMPONENTSET_LIGHT)
    list(APPEND HEADER_FILES

        )
    list(APPEND SOURCE_FILES

        )
endif()

############################## COMPONENTS HERE ARE THE STANDARD-SET ################################
if(SOFA_BUILD_COMPONENTSET_STANDARD)
    list(APPEND HEADER_FILES
        )
    list(APPEND SOURCE_FILES
        VariationalSymplecticExplicitSolverDynamic_test.cpp
        VariationalSymplecticImplicitSolverDynamic_test.cpp
        )

endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
    list(APPEND SOURCE_FILES
        VariationalSymplecticImplicitSolverNonLinearForceDynamic_test.cpp
        )

endif()


add_definitions("-DSOFAGENERALIMPLICITODESOLVER_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
